"""
Paso 2: Construye la estructura de carpetas que necesita YOLOv8 y genera dataset.yaml

Estructura resultante:
    yolo_dataset/
    ├── images/
    │   ├── train/    (80%)
    │   ├── val/      (10%)
    │   └── test/     (10%)
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── dataset.yaml

Uso:
    python build_yolo_structure.py

IMPORTANTE: Ejecutar DESPUÉS de labelme2yolo_converter.py
"""

import os
import random
import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
DATASET_DIR   = Path("dataset")         # Carpeta raíz del dataset
OUTPUT_DIR    = Path("yolo_dataset")    # Carpeta de salida para YOLO
CLASSES_FILE  = DATASET_DIR / "classes.txt"

TRAIN_RATIO   = 0.80
VAL_RATIO     = 0.10
TEST_RATIO    = 0.10

RANDOM_SEED   = 42   # Para reproducibilidad del split

VALID_IMG_EXT = {".jpg", ".jpeg", ".png"}

# ──────────────────────────────────────────────

def load_classes(classes_file: Path) -> list:
    """Lee classes.txt y retorna lista de clases en orden."""
    classes = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    return classes


def collect_pairs(dataset_dir: Path) -> list:
    """
    Busca todos los pares (imagen, label.txt) válidos en el dataset.
    Solo incluye imágenes que tienen su .txt correspondiente.
    """
    pairs = []
    # Buscar imágenes recursivamente
    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() not in VALID_IMG_EXT:
            continue
        # Buscar el .txt correspondiente
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((img_path, txt_path))
        else:
            print(f"  [WARN] Sin .txt para: {img_path.relative_to(dataset_dir)}")
    return pairs


def split_dataset(pairs: list) -> tuple:
    """Divide los pares en train/val/test de forma estratificada por clase."""
    random.seed(RANDOM_SEED)

    # Agrupar por clase (primer nivel de carpeta = clase)
    by_class = {}
    for img_path, txt_path in pairs:
        # La clase es el nombre de la carpeta directa bajo dataset/
        class_name = img_path.relative_to(DATASET_DIR).parts[0]
        by_class.setdefault(class_name, []).append((img_path, txt_path))

    train_pairs, val_pairs, test_pairs = [], [], []

    for class_name, class_pairs in sorted(by_class.items()):
        random.shuffle(class_pairs)
        n = len(class_pairs)
        n_val  = max(1, round(n * VAL_RATIO))
        n_test = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test

        train_pairs.extend(class_pairs[:n_train])
        val_pairs.extend(class_pairs[n_train:n_train + n_val])
        test_pairs.extend(class_pairs[n_train + n_val:])

        print(f"  {class_name:>3}: {n_train:>3} train | {n_val:>2} val | {n_test:>2} test  (total: {n})")

    return train_pairs, val_pairs, test_pairs


def copy_pairs(pairs: list, img_dir: Path, lbl_dir: Path, split_name: str):
    """Copia imágenes y labels a las carpetas de destino, renombrando si hay conflicto."""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, txt_path in pairs:
        # Nombre único: clase_voluntario_nombreoriginal
        relative = img_path.relative_to(DATASET_DIR)
        # Reemplazar separadores de carpeta por guion bajo
        unique_stem = "_".join(relative.parts[:-1]) + "_" + img_path.stem
        new_img_name = unique_stem + img_path.suffix.lower()
        new_txt_name = unique_stem + ".txt"

        dest_img = img_dir / new_img_name
        dest_txt = lbl_dir / new_txt_name

        shutil.copy2(img_path, dest_img)
        shutil.copy2(txt_path, dest_txt)


def generate_yaml(output_dir: Path, classes: list):
    """Genera el archivo dataset.yaml requerido por YOLOv8."""
    yaml_path = output_dir / "dataset.yaml"
    abs_output = output_dir.resolve()

    # Construir lista de clases en formato YAML
    names_str = "\n".join(f"  {i}: {cls}" for i, cls in enumerate(classes))

    yaml_content = f"""# Dataset de Lengua de Señas Colombiana
# Generado automáticamente por build_yolo_structure.py

path: {abs_output.as_posix()}   # Ruta absoluta al dataset
train: images/train
val: images/val
test: images/test

nc: {len(classes)}   # Número de clases

names:
{names_str}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n[INFO] dataset.yaml guardado en: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("  Build YOLO Dataset Structure")
    print("=" * 60)

    # Verificar dataset
    if not DATASET_DIR.exists():
        print(f"[ERROR] No se encontró: {DATASET_DIR}")
        return

    # Limpiar output anterior si existe
    if OUTPUT_DIR.exists():
        print(f"\n[INFO] Eliminando dataset anterior en {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    # Cargar clases
    classes = load_classes(CLASSES_FILE)
    print(f"\n[INFO] Clases ({len(classes)}): {classes}")

    # Recopilar pares imagen-label
    print("\n[INFO] Buscando pares imagen + label...")
    pairs = collect_pairs(DATASET_DIR)
    print(f"[INFO] Pares válidos encontrados: {len(pairs)}")

    if not pairs:
        print("[ERROR] No se encontraron pares imagen+label. Ejecuta primero labelme2yolo_converter.py")
        return

    # Dividir en splits
    print(f"\n[INFO] Dividiendo dataset ({TRAIN_RATIO*100:.0f}% train / "
          f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test):\n")
    train_pairs, val_pairs, test_pairs = split_dataset(pairs)

    # Copiar archivos
    print("\n[INFO] Copiando archivos...")
    copy_pairs(train_pairs, OUTPUT_DIR / "images/train", OUTPUT_DIR / "labels/train", "train")
    copy_pairs(val_pairs,   OUTPUT_DIR / "images/val",   OUTPUT_DIR / "labels/val",   "val")
    copy_pairs(test_pairs,  OUTPUT_DIR / "images/test",  OUTPUT_DIR / "labels/test",  "test")

    # Generar YAML
    yaml_path = generate_yaml(OUTPUT_DIR, classes)

    # Resumen
    print("\n" + "=" * 60)
    print(f"  Train: {len(train_pairs):>4} imágenes")
    print(f"  Val:   {len(val_pairs):>4} imágenes")
    print(f"  Test:  {len(test_pairs):>4} imágenes")
    print(f"  Total: {len(pairs):>4} imágenes")
    print("=" * 60)
    print(f"\n  Dataset YAML: {yaml_path}")
    print("\n[LISTO] Ahora ejecuta: python data_augmentation.py")
    print("        (o directamente: python train_yolo.py si tienes suficientes datos)")


if __name__ == "__main__":
    main()
