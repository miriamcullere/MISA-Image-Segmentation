import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import warnings
from tqdm import tqdm
import gc

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd,
    Spacingd, EnsureTyped, AsDiscrete, 
    ResizeWithPadOrCropd, # Usamos este para mantener la anatomía real
    CropForegroundd,      # Centra el cerebro para que no se corte
    Rand3DElasticd, RandFlipd, Invertd
)
from monai.data import Dataset, DataLoader, decollate_batch, MetaTensor
from monai.losses import DiceLoss
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# -------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Hardware: Aceleración Apple Metal (MPS) ACTIVADA.")
else:
    device = torch.device("cpu")
    print(">>> Hardware: CPU.")

set_determinism(seed=0)

data_dir = '/Users/miriamcullere/Desktop/MMIC/MISA/PROJECTE FINAL/TrainingValidationTestSets'

# Carpetas de salida
output_dir_val = os.path.join(data_dir, "ResUNet_Validation_Results")
output_dir_test = os.path.join(data_dir, "ResUNet_TEST_SUBMISSION")

os.makedirs(output_dir_val, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

# -------------------------------------------------------------------------
# 1. ARQUITECTURA RES-U-NET
# -------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        res = self.shortcut(x)
        out += res
        return self.relu(out)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.inc = ResidualConvBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# -------------------------------------------------------------------------
# 2. DATA PIPELINE (CORREGIDO: CROP + PAD 160)
# -------------------------------------------------------------------------
def get_files_dict(folder_name, has_labels=True):
    search_path = os.path.join(data_dir, folder_name, 'IBSR_*', '*.nii.gz')
    all_files = sorted(glob.glob(search_path))
    
    if has_labels:
        labels = [x for x in all_files if '_seg.nii.gz' in x]
        images = [x for x in all_files if '_seg.nii.gz' not in x]
        return [{"image": i, "label": l, "id": os.path.basename(os.path.dirname(i))} for i, l in zip(images, labels)]
    else:
        images = [x for x in all_files if '_seg.nii.gz' not in x]
        return [{"image": i, "id": os.path.basename(os.path.dirname(i))} for i in images]

train_files = get_files_dict('Training_Set', has_labels=True)
val_files = get_files_dict('Validation_Set', has_labels=True)
test_files = get_files_dict('Test_Set', has_labels=False)

print(f">>> Datos: {len(train_files)} Train | {len(val_files)} Validation | {len(test_files)} Test")

# --- TRANSFORMACIONES TRAIN / VAL ---
base_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    
    # 1. CENTRADO: Elimina fondo negro sobrante para centrar el cerebro
    CropForegroundd(keys=["image", "label"], source_key="image"),
    
    # 2. AJUSTE DE TAMAÑO: Aumentamos Z ( antes era 160) 
    # Usamos constant (
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(288, 288, 256), mode=("constant", "constant")),
    
    EnsureTyped(keys=["image", "label"]),
]

train_transforms = Compose(base_transforms + [
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.2, padding_mode="zeros", mode=("bilinear", "nearest")),
])

val_transforms = Compose(base_transforms)

# --- TRANSFORMACIONES TEST ---
test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    
    # Mismas transformaciones geométricas que el Train
    CropForegroundd(keys=["image"], source_key="image"),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=(288, 288, 256), mode=("constant")),
    
    EnsureTyped(keys=["image"]),
])

# DataLoaders
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# -------------------------------------------------------------------------
# 3. ENTRENAMIENTO (OPTIMIZADO)
# -------------------------------------------------------------------------
model = ResUNet(n_channels=1, n_classes=4).to(device)

# Pesos: Ajusta según frecuencia real de clases si es necesario
class_weights = torch.tensor([0.1, 3.0, 1.0, 1.0], dtype=torch.float32).to(device)

loss_dice_fn = DiceLoss(to_onehot_y=True, softmax=True)
loss_ce_fn = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Aumentamos batch size real y eliminamos acumulación manual compleja
SLICE_BATCH_SIZE = 8 
max_epochs = 40 

best_model_path = os.path.join(data_dir, "best_metric_model.pth")
last_model_path = os.path.join(data_dir, "last_checkpoint.pth")

if os.path.exists(best_model_path):
    print(f"\n>>> Modelo encontrado: {best_model_path}. Saltando entrenamiento.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
else:
    print(f"\n>>> Iniciando entrenamiento de {max_epochs} épocas...")
    best_dice = -1
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        total_steps = 0
        
        # Barra de progreso por volúmenes
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", unit="vol")
        
        for batch_data in progress_bar:
            inputs_3d = batch_data["image"].to(device)
            labels_3d = batch_data["label"].to(device)
            
            # 1. Convertir Volumen 3D a Lista de Slices 2D
            # [Batch_Vol, Channel, H, W, Depth] -> [Depth, Channel, H, W]
            inputs_2d_all = inputs_3d[0].permute(3, 0, 1, 2)
            labels_2d_all = labels_3d[0].permute(3, 0, 1, 2)
            
            # 2. Filtrado de Slices Vacíos (Background puro)
            # Aceleramos el entrenamiento ignorando cortes sin cerebro
            mask = labels_2d_all.sum(dim=(1, 2, 3)) > 0 
            inputs_2d_all = inputs_2d_all[mask]
            labels_2d_all = labels_2d_all[mask]
            
            if len(inputs_2d_all) == 0: continue

            # 3. SHUFFLE (IMPORTANTE): Mezclar slices para romper correlación espacial
            perm_indices = torch.randperm(len(inputs_2d_all))
            inputs_2d_all = inputs_2d_all[perm_indices]
            labels_2d_all = labels_2d_all[perm_indices]
            
            # 4. Batch Loop (Entrenamiento por lotes de slices)
            for i in range(0, len(inputs_2d_all), SLICE_BATCH_SIZE):
                inputs_mini = inputs_2d_all[i : i + SLICE_BATCH_SIZE]
                labels_mini = labels_2d_all[i : i + SLICE_BATCH_SIZE]
                
                optimizer.zero_grad()
                
                # Forward
                outputs_mini = model(inputs_mini)
                
                # Loss Calculation
                l_dice = loss_dice_fn(outputs_mini, labels_mini)
                l_ce = loss_ce_fn(outputs_mini, labels_mini.squeeze(1).long())
                loss = l_dice + l_ce
                
                # Backward
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_steps += 1
        
        avg_train_loss = epoch_loss / total_steps if total_steps > 0 else 1.0
        
        # ---------------- VALIDATION ----------------
        model.eval()
        val_dice_accum = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                
                # Inferencia slice a slice (sin shuffle, ordenado para reconstruir)
                val_in_slices = val_inputs[0].permute(3, 0, 1, 2)
                val_out_list = []
                
                # Procesamos validación en batches también para no saturar memoria
                for k in range(0, len(val_in_slices), SLICE_BATCH_SIZE):
                    batch_slice = val_in_slices[k : k + SLICE_BATCH_SIZE]
                    val_out_list.append(model(batch_slice))
                
                # Reconstrucción 3D: [Depth, 4, H, W] -> [1, 4, H, W, Depth]
                val_pred_vol = torch.cat(val_out_list, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
                # OJO: La permutación inversa debe coincidir con la entrada original
                # Original era (B, C, H, W, D). Slices era (D, C, H, W).
                # Cat dim 0 crea (Total_D, C, H, W). 
                # Queremos (B, C, H, W, D).
                # Permute correcto desde (D, C, H, W) -> (C, H, W, D) -> unsqueeze -> (1, C, H, W, D)
                val_pred_vol = torch.cat(val_out_list, dim=0).permute(1, 2, 3, 0).unsqueeze(0)

                # Métricas rápidas (Dice promedio soft)
                val_pred_hard = torch.argmax(val_pred_vol, dim=1, keepdim=True)
                
                pred_oh = torch.nn.functional.one_hot(val_pred_hard.squeeze(1).long(), num_classes=4).permute(0, 4, 1, 2, 3)
                label_oh = torch.nn.functional.one_hot(val_labels.squeeze(1).long(), num_classes=4).permute(0, 4, 1, 2, 3)
                
                dice_vals = compute_dice(pred_oh, label_oh, include_background=False)
                val_dice_accum += dice_vals.mean().item()
                val_steps += 1
        
        avg_val_dice = val_dice_accum / val_steps if val_steps > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        scheduler.step(avg_val_dice)
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f">>> ¡Nuevo Récord! Modelo guardado ({best_dice:.4f}).")
        
        torch.save(model.state_dict(), last_model_path)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

from monai.inferers import sliding_window_inference

# =========================================================================
# CLASE ADAPTADORA (WRAPPER) - NUEVO
# Permite usar una Red 2D dentro de Sliding Window 3D
# =========================================================================
class Model2DWrapper(nn.Module):
    def __init__(self, model_2d):
        super().__init__()
        self.model_2d = model_2d
    
    def forward(self, x_5d):
        # x_5d viene de sliding_window: [Batch, Canal, H, W, Depth]
        b, c, h, w, d = x_5d.shape
        
        # 1. Aplanamos Depth dentro del Batch para que la red 2D lo entienda
        # [B, C, H, W, D] -> [B, D, C, H, W] -> [B*D, C, H, W]
        x_2d = x_5d.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, h, w)
        
        # 2. Inferencia 2D normal
        y_2d = self.model_2d(x_2d) # [B*D, n_classes, H, W]
        
        # 3. Reconstruimos el volumen 3D
        # [B*D, n_classes, H, W] -> [B, D, n_classes, H, W] -> [B, n_classes, H, W, D]
        n_class = y_2d.shape[1]
        y_3d = y_2d.view(b, d, n_class, h, w).permute(0, 2, 3, 4, 1)
        
        return y_3d

# Instanciamos el wrapper con tu modelo cargado
model_wrapper = Model2DWrapper(model)
model_wrapper.eval() # Importante

# -------------------------------------------------------------------------
# 4. EVALUACIÓN CON SLIDING WINDOW + MÉTRICAS DETALLADAS
# -------------------------------------------------------------------------
print("\n>>> Generando métricas finales en Validation Set (Sliding Window)...")

# Configuración segura para memoria (MPS/Mac)
ROI_SIZE = (256, 256, 16) 
SW_BATCH_SIZE = 1 
OVERLAP = 0.25

# --- TRANSFORMACIONES LIMPIAS ---
inference_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

val_ds_clean = Dataset(data=val_files, transform=inference_transforms)
val_loader_clean = DataLoader(val_ds_clean, batch_size=1, shuffle=False, num_workers=0)

post_transforms_val = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=inference_transforms,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    ),
])

# Acumuladores para la media final
global_metrics = {'dice': [], 'hd95': [], 'avd': []}

with torch.no_grad():
    for i, batch_data in enumerate(val_loader_clean):
        subject_id = val_files[i]["id"]
        original_gt_path = val_files[i]["label"]
        
        # 1. Inferencia
        vol_input = batch_data["image"].to(device)
        
        val_outputs = sliding_window_inference(
            inputs=vol_input, 
            roi_size=ROI_SIZE, 
            sw_batch_size=SW_BATCH_SIZE, 
            predictor=model_wrapper, 
            overlap=OVERLAP,
            mode="gaussian"
        )
        
        seg_vol_tensor = torch.argmax(val_outputs, dim=1, keepdim=True)
        
        # 2. Inversión al espacio original
        batch_data["pred"] = seg_vol_tensor.cpu()
        batch_data["image"] = batch_data["image"].cpu()
        
        batch_inverted = [post_transforms_val(x) for x in decollate_batch(batch_data)]
        pred_final = batch_inverted[0]["pred"].cpu().numpy().astype(np.uint8)
        if pred_final.ndim == 4: pred_final = pred_final.squeeze(0)
        
        # 3. Cargar Ground Truth (GT)
        gt_nii = nib.load(original_gt_path)
        gt_final = gt_nii.get_fdata().astype(np.uint8)
        if gt_final.ndim == 4: gt_final = gt_final.squeeze(-1)

        # =================================================================
        # CÁLCULO DE MÉTRICAS (Dice, HD95, AVD)
        # =================================================================
        
        # Preparamos Tensores One-Hot para MONAI [B, C, H, W, D]
        # Transponemos numpy de (H,W,D) a (C, H,W,D) y añadimos Batch dim
        n_classes = 4
        pred_oh = np.eye(n_classes)[pred_final].transpose(3, 0, 1, 2)[np.newaxis, ...]
        gt_oh = np.eye(n_classes)[gt_final].transpose(3, 0, 1, 2)[np.newaxis, ...]
        
        # Convertimos a tensor (CPU es suficiente y seguro para métricas)
        pred_tensor = torch.tensor(pred_oh)
        gt_tensor = torch.tensor(gt_oh)

        # A. Dice & HD95 (MONAI)
        # include_background=False ignora la clase 0
        dice_vals = compute_dice(pred_tensor, gt_tensor, include_background=False).squeeze(0).numpy()
        hd95_vals = compute_hausdorff_distance(pred_tensor, gt_tensor, include_background=False, percentile=95).squeeze(0).numpy()
        
        # B. AVD (Average Volumetric Difference) - Cálculo Manual
        avd_vals = []
        for c in range(1, n_classes): # Clases 1, 2, 3
            vol_p = np.sum(pred_final == c)
            vol_g = np.sum(gt_final == c)
            if vol_g == 0:
                avd = 0.0 if vol_p == 0 else 100.0
            else:
                avd = abs(vol_p - vol_g) / vol_g * 100.0
            avd_vals.append(avd)
        
        # Guardar para promedios globales
        global_metrics['dice'].append(dice_vals)
        global_metrics['hd95'].append(hd95_vals)
        global_metrics['avd'].append(avd_vals)

        # =================================================================
        # IMPRIMIR TABLA POR SUJETO
        # =================================================================
        print(f"\nResultados para: {subject_id}")
        print(f"{'Metric':<12} | {'CSF (1)':<10} | {'GM (2)':<10} | {'WM (3)':<10} | {'Avg':<10}")
        print("-" * 65)
        
        # Dice Row
        d_avg = np.mean(dice_vals)
        print(f"{'Dice':<12} | {dice_vals[0]:.4f}     | {dice_vals[1]:.4f}     | {dice_vals[2]:.4f}     | {d_avg:.4f}")
        
        # HD95 Row
        h_avg = np.mean(hd95_vals)
        print(f"{'HD95':<12} | {hd95_vals[0]:.4f}     | {hd95_vals[1]:.4f}     | {hd95_vals[2]:.4f}     | {h_avg:.4f}")
        
        # AVD Row
        a_avg = np.mean(avd_vals)
        print(f"{'AVD (%)':<12} | {avd_vals[0]:.4f}     | {avd_vals[1]:.4f}     | {avd_vals[2]:.4f}     | {a_avg:.4f}")
        print("-" * 65)
        
        # Guardar NIfTI
        final_nifti = nib.Nifti1Image(pred_final, gt_nii.affine, gt_nii.header)
        save_path = os.path.join(output_dir_val, f"{subject_id}_PRED_FULL.nii.gz")
        nib.save(final_nifti, save_path)

# IMPRIMIR MEDIA TOTAL DEL VALIDATION SET
print("\n" + "="*65)
print("MEDIA GLOBAL (VALIDATION SET)")
print("="*65)
print(f"{'Metric':<12} | {'CSF (1)':<10} | {'GM (2)':<10} | {'WM (3)':<10} | {'Avg':<10}")
print("-" * 65)

# Calcular promedios de columnas
mean_dice = np.mean(global_metrics['dice'], axis=0)
mean_hd95 = np.mean(global_metrics['hd95'], axis=0)
mean_avd = np.mean(global_metrics['avd'], axis=0)

print(f"{'Dice':<12} | {mean_dice[0]:.4f}     | {mean_dice[1]:.4f}     | {mean_dice[2]:.4f}     | {np.mean(mean_dice):.4f}")
print(f"{'HD95':<12} | {mean_hd95[0]:.4f}     | {mean_hd95[1]:.4f}     | {mean_hd95[2]:.4f}     | {np.mean(mean_hd95):.4f}")
print(f"{'AVD (%)':<12} | {mean_avd[0]:.4f}     | {mean_avd[1]:.4f}     | {mean_avd[2]:.4f}     | {np.mean(mean_avd):.4f}")
print("-" * 65)

# -------------------------------------------------------------------------
# 5. TEST SET (CORREGIDO)
# -------------------------------------------------------------------------
print("\n>>> Generando Test Set sin cortes...")

test_transforms_clean = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
    ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])

test_ds_clean = Dataset(data=test_files, transform=test_transforms_clean)
test_loader_clean = DataLoader(test_ds_clean, batch_size=1, shuffle=False, num_workers=0)

post_transforms_test = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_transforms_clean,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    ),
])

with torch.no_grad():
    for i, batch_data in enumerate(test_loader_clean):
        subject_id = test_files[i]["id"]
        original_img_path = test_files[i]["image"]
        
        print(f"Inferencia Test: {subject_id}")
        
        vol_input = batch_data["image"].to(device)
        
        # Usamos el wrapper aquí también
        test_outputs = sliding_window_inference(
            inputs=vol_input, 
            roi_size=(256, 256, 16), # (Hardcoded o variable)
            sw_batch_size=1,         
            predictor=model_wrapper, 
            overlap=OVERLAP,
            mode="gaussian"
        )
        
        seg_vol_tensor = torch.argmax(test_outputs, dim=1, keepdim=True)
        
        batch_data["pred"] = seg_vol_tensor.cpu()
        batch_data["image"] = batch_data["image"].cpu()
        
        batch_inverted = [post_transforms_test(x) for x in decollate_batch(batch_data)]
        pred_final = batch_inverted[0]["pred"].cpu().numpy().astype(np.uint8)
        if pred_final.ndim == 4: pred_final = pred_final.squeeze(0)
        
        # Guardar
        orig_nii = nib.load(original_img_path)
        final_nifti = nib.Nifti1Image(pred_final, orig_nii.affine, orig_nii.header)
        save_path = os.path.join(output_dir_test, f"{subject_id}_PRED.nii.gz")
        nib.save(final_nifti, save_path)
        print(f"-> Guardado Test: {save_path}")
