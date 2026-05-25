"""
Paso 3: Data Augmentation para YOLO Segmentation.

Con solo ~21 imágenes por clase necesitamos aumentar el dataset.
Este script genera nuevas imágenes+labels aplicando transformaciones
que respetan los polígonos de segmentación.

Transformaciones aplicadas:
    - Flip horizontal
    - Rotación (±20°)
    - Escala (zoom in/out)
    - Brillo y contraste aleatorio
    - Blur gaussiano suave
    - Ruido aditivo
    - Ajuste de saturación/hue (HSV)

Resultado: Multiplica x5 el número de imágenes en el split de TRAIN únicamente.

Uso:
    python data_augmentation.py

IMPORTANTE: Ejecutar DESPUÉS de build_yolo_structure.py
Dependencias: pip install albumentations opencv-python
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
import albumentations as A

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
YOLO_DATASET_DIR = Path("yolo_dataset")   # Dataset generado por build_yolo_structure.py
AUGMENTATIONS_PER_IMAGE = 5               # Cuántas versiones generar por imagen original
RANDOM_SEED = 42

# Solo se hace augmentation en TRAIN, nunca en val/test
TRAIN_IMAGES_DIR = YOLO_DATASET_DIR / "images/train"
TRAIN_LABELS_DIR = YOLO_DATASET_DIR / "labels/train"

VALID_IMG_EXT = {".jpg", ".jpeg", ".png"}

# ──────────────────────────────────────────────


def build_augmentation_pipeline():
    """Define el pipeline de augmentación compatible con polígonos."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    ], keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False,   # Mantener puntos aunque salgan del frame
        angle_in_degrees=True
    ))


def read_yolo_label(txt_path: Path, img_w: int, img_h: int):
    """
    Lee un .txt de YOLO Segmentation y devuelve lista de shapes:
    [(class_id, [(x_abs, y_abs), ...]), ...]
    """
    shapes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # Mínimo: class + 2 puntos (4 coords)
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            # Convertir coordenadas normalizadas a absolutas
            points = []
            for i in range(0, len(coords), 2):
                x_abs = coords[i] * img_w
                y_abs = coords[i + 1] * img_h
                points.append((x_abs, y_abs))
            shapes.append((class_id, points))
    return shapes


def write_yolo_label(txt_path: Path, shapes: list, img_w: int, img_h: int):
    """Guarda shapes como .txt en formato YOLO Segmentation normalizado."""
    lines = []
    for class_id, points in shapes:
        coords = []
        for x_abs, y_abs in points:
            x_norm = max(0.0, min(1.0, x_abs / img_w))
            y_norm = max(0.0, min(1.0, y_abs / img_h))
            coords.extend([round(x_norm, 6), round(y_norm, 6)])
        coords_str = " ".join(str(c) for c in coords)
        lines.append(f"{class_id} {coords_str}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def augment_image_and_label(img_path: Path, txt_path: Path, transform, aug_idx: int):
    """
    Aplica augmentación a una imagen y su label, guarda los resultados.
    """
    # Leer imagen
    image = cv2.imread(str(img_path))
    if image is None:
        return False
    img_h, img_w = image.shape[:2]

    # Leer label
    shapes = read_yolo_label(txt_path, img_w, img_h)
    if not shapes:
        return False

    # Preparar keypoints (todos los puntos de todos los polígonos)
    all_keypoints = []
    shape_info = []  # (class_id, num_points)
    for class_id, points in shapes:
        shape_info.append((class_id, len(points)))
        for x, y in points:
            all_keypoints.append((x, y))

    # Aplicar transformación
    try:
        result = transform(image=image, keypoints=all_keypoints)
    except Exception as e:
        print(f"    [WARN] Error en augmentación: {e}")
        return False

    aug_image = result["image"]
    aug_keypoints = result["keypoints"]
    aug_h, aug_w = aug_image.shape[:2]

    # Reconstruir shapes con los keypoints transformados
    aug_shapes = []
    kp_idx = 0
    for class_id, num_points in shape_info:
        pts = []
        for _ in range(num_points):
            if kp_idx < len(aug_keypoints):
                kp = aug_keypoints[kp_idx]
                pts.append((kp[0], kp[1]))
            kp_idx += 1
        if pts:
            aug_shapes.append((class_id, pts))

    if not aug_shapes:
        return False

    # Guardar imagen aumentada
    stem = img_path.stem
    suffix = img_path.suffix.lower()
    aug_img_name = f"{stem}_aug{aug_idx:02d}{suffix}"
    aug_txt_name = f"{stem}_aug{aug_idx:02d}.txt"

    aug_img_path = img_path.parent / aug_img_name
    aug_txt_path = txt_path.parent / aug_txt_name

    cv2.imwrite(str(aug_img_path), aug_image)
    write_yolo_label(aug_txt_path, aug_shapes, aug_w, aug_h)

    return True


def main():
    print("=" * 60)
    print("  Data Augmentation — YOLO Segmentation")
    print("=" * 60)

    if not TRAIN_IMAGES_DIR.exists():
        print(f"[ERROR] No existe: {TRAIN_IMAGES_DIR}")
        print("        Ejecuta primero: python build_yolo_structure.py")
        return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Recopilar imágenes ORIGINALES (sin _aug en el nombre)
    original_images = [
        p for p in TRAIN_IMAGES_DIR.iterdir()
        if p.suffix.lower() in VALID_IMG_EXT and "_aug" not in p.stem
    ]

    print(f"\n[INFO] Imágenes originales en train: {len(original_images)}")
    print(f"[INFO] Augmentaciones por imagen:    {AUGMENTATIONS_PER_IMAGE}")
    print(f"[INFO] Total nuevas imágenes:        {len(original_images) * AUGMENTATIONS_PER_IMAGE}")
    print(f"[INFO] Total final en train:         {len(original_images) * (AUGMENTATIONS_PER_IMAGE + 1)}\n")

    transform = build_augmentation_pipeline()

    generated = 0
    failed = 0

    for i, img_path in enumerate(sorted(original_images), 1):
        txt_path = TRAIN_LABELS_DIR / (img_path.stem + ".txt")

        if not txt_path.exists():
            print(f"  [{i:>3}] [WARN] Sin label para: {img_path.name}")
            continue

        print(f"  [{i:>3}/{len(original_images)}] {img_path.name}")

        for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
            success = augment_image_and_label(img_path, txt_path, transform, aug_idx + 1)
            if success:
                generated += 1
            else:
                failed += 1

    # Contar total final
    final_count = sum(
        1 for p in TRAIN_IMAGES_DIR.iterdir()
        if p.suffix.lower() in VALID_IMG_EXT
    )

    print("\n" + "=" * 60)
    print(f"  Imágenes generadas: {generated}")
    print(f"  Fallidas:           {failed}")
    print(f"  Total en train:     {final_count}")
    print("=" * 60)
    print("\n[LISTO] Ahora ejecuta: python train_yolo.py")


if __name__ == "__main__":
    main()
