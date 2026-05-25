"""
Paso 1: Convierte anotaciones JSON de LabelMe a formato TXT de YOLO Segmentation.

Formato YOLO Segmentation (por línea en el .txt):
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    donde las coordenadas están normalizadas entre 0.0 y 1.0

Uso:
    python labelme2yolo_converter.py

Resultado:
    Para cada archivo .json en el dataset, se genera un .txt en la misma carpeta.
"""

import json
import os
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
DATASET_DIR = Path("dataset")          # Carpeta raíz del dataset
CLASSES_FILE = DATASET_DIR / "classes.txt"   # Archivo con lista de clases


def load_classes(classes_file: Path) -> dict:
    """Lee classes.txt y devuelve un dict {nombre_lowercase: índice}."""
    classes = {}
    with open(classes_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:  # Ignora líneas vacías
                classes[name.lower()] = idx
    print(f"[INFO] Clases cargadas: {classes}")
    return classes


def convert_json_to_txt(json_path: Path, classes: dict) -> bool:
    """
    Convierte un .json de LabelMe a .txt de YOLO Segmentation.
    Retorna True si fue exitoso, False si se saltó.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_width = data.get("imageWidth")
    img_height = data.get("imageHeight")

    if not img_width or not img_height:
        print(f"  [WARN] Sin dimensiones de imagen en: {json_path.name} — saltando")
        return False

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"  [WARN] Sin anotaciones en: {json_path.name} — saltando")
        return False

    txt_lines = []

    for shape in shapes:
        label = shape.get("label", "").lower().strip()
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        # Solo procesar polígonos
        if shape_type != "polygon":
            print(f"  [WARN] Shape type '{shape_type}' ignorado en {json_path.name}")
            continue

        # Verificar que la clase existe
        if label not in classes:
            print(f"  [WARN] Clase '{label}' no encontrada en classes.txt — saltando shape")
            continue

        class_id = classes[label]

        # Normalizar coordenadas entre 0.0 y 1.0
        normalized_points = []
        for x, y in points:
            x_norm = round(x / img_width, 6)
            y_norm = round(y / img_height, 6)
            # Clamp entre 0 y 1 por seguridad
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            normalized_points.extend([x_norm, y_norm])

        # Construir línea: <class_id> <x1> <y1> <x2> <y2> ...
        coords_str = " ".join(str(c) for c in normalized_points)
        txt_lines.append(f"{class_id} {coords_str}")

    if not txt_lines:
        print(f"  [WARN] Sin líneas válidas para: {json_path.name}")
        return False

    # Guardar .txt en la misma carpeta que el .json
    txt_path = json_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return True


def main():
    print("=" * 60)
    print("  LabelMe JSON -> YOLO TXT Converter")
    print("=" * 60)

    # Verificar que existe el dataset
    if not DATASET_DIR.exists():
        print(f"[ERROR] No se encontró la carpeta: {DATASET_DIR}")
        return

    # Cargar clases
    classes = load_classes(CLASSES_FILE)

    # Buscar todos los .json recursivamente en el dataset
    json_files = list(DATASET_DIR.rglob("*.json"))
    # Excluir el log de captura
    json_files = [f for f in json_files if f.name != "log_captura.json"]

    total = len(json_files)
    print(f"\n[INFO] Archivos JSON encontrados: {total}\n")

    converted = 0
    skipped = 0

    for i, json_path in enumerate(json_files, 1):
        print(f"[{i:>4}/{total}] {json_path.relative_to(DATASET_DIR)}", end=" -> ")
        success = convert_json_to_txt(json_path, classes)
        if success:
            print("OK")
            converted += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print(f"  Convertidos: {converted}")
    print(f"  Saltados:    {skipped}")
    print(f"  Total:       {total}")
    print("=" * 60)
    print("\n[LISTO] Ahora ejecuta: python build_yolo_structure.py")



if __name__ == "__main__":
    main()
