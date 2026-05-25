"""
Paso 4: Entrenamiento de YOLOv8 Segmentation con Transfer Learning.

Usa pesos pre-entrenados en COCO (yolov8s-seg.pt) como punto de partida
y los ajusta para reconocer señas de la Lengua de Señas Colombiana.

Uso:
    python train_yolo.py

Dependencias: pip install ultralytics
GPU: NVIDIA con CUDA (recomendado, se detecta automáticamente)
"""

from pathlib import Path
from ultralytics import YOLO
import torch

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────

# Modelo base para Transfer Learning
# Opciones: yolov8n-seg (nano, más rápido) | yolov8s-seg (small, recomendado) | yolov8m-seg (medium, más preciso)
# ⚠ RTX 2060 6GB: usar nano para evitar OOM. Cambiar a 's' solo si la VRAM aguanta.
BASE_MODEL = "yolov8n-seg.pt"

# Ruta al dataset.yaml generado por build_yolo_structure.py
DATASET_YAML = Path("yolo_dataset/dataset.yaml")

# Nombre del experimento (los resultados se guardan en runs/segment/<RUN_NAME>/)
RUN_NAME = "senas_colombianas_v1"

# ── Hiperparámetros ──
EPOCHS      = 150       # Número de épocas de entrenamiento
IMG_SIZE    = 640       # Resolución de entrada (640 recomendado para YOLOv8)
BATCH_SIZE  = 4         # RTX 2060 6GB: batch 4 conservador para segmentación a 640px
PATIENCE    = 30        # Early stopping: detiene si no mejora en N épocas
WORKERS     = 2         # ⬇ Reducido a 2 para liberar RAM del sistema

# ── Augmentation integrado de YOLOv8 ──
# (Complementa el augmentation manual hecho con albumentations)
DEGREES     = 10.0      # Rotación máxima (°)
FLIPUD      = 0.0       # Flip vertical (0 = desactivado, señas no se ven al revés)
FLIPLR      = 0.5       # Flip horizontal (50% de probabilidad)
MOSAIC      = 1.0       # Mosaic augmentation (muy efectivo con pocos datos)
MIXUP       = 0.1       # MixUp augmentation
HSV_H       = 0.015     # Variación de hue
HSV_S       = 0.7       # Variación de saturación
HSV_V       = 0.4       # Variación de valor/brillo
SCALE       = 0.5       # Variación de escala

# ──────────────────────────────────────────────


def check_gpu():
    """Verifica disponibilidad de GPU y muestra información."""
    print("\n[GPU] Información del dispositivo:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ VRAM: {vram:.1f} GB")
        device = "0"  # Usar primera GPU
    else:
        print("  ⚠ GPU no encontrada. Entrenando en CPU (muy lento).")
        device = "cpu"
    return device


def verify_dataset():
    """Verifica que el dataset.yaml existe y tiene el formato correcto."""
    if not DATASET_YAML.exists():
        print(f"[ERROR] No se encontró: {DATASET_YAML}")
        print("        Ejecuta primero:")
        print("        1. python labelme2yolo_converter.py")
        print("        2. python build_yolo_structure.py")
        print("        3. python data_augmentation.py")
        return False

    print(f"\n[INFO] Dataset YAML: {DATASET_YAML.resolve()}")

    # Contar imágenes por split
    yolo_dir = DATASET_YAML.parent
    for split in ["train", "val", "test"]:
        img_dir = yolo_dir / "images" / split
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            print(f"  {split:>5}: {count:>4} imágenes")

    return True


def train():
    """Entrena YOLOv8 Segmentation con Transfer Learning."""
    print("=" * 60)
    print("  YOLOv8 Segmentation — Transfer Learning")
    print("  Lengua de Señas Colombiana")
    print("=" * 60)

    # Verificar dataset
    if not verify_dataset():
        return

    # Verificar GPU
    device = check_gpu()

    print(f"\n[INFO] Modelo base: {BASE_MODEL}")
    print(f"[INFO] Épocas:      {EPOCHS}")
    print(f"[INFO] Batch size:  {BATCH_SIZE}")
    print(f"[INFO] Img size:    {IMG_SIZE}")
    print(f"[INFO] Run name:    {RUN_NAME}")
    print(f"\n[INFO] Iniciando entrenamiento...\n")

    # Limpiar estado de GPU y configurar uso de memoria antes de empezar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Limitar la fracción de VRAM que PyTorch puede reservar (evita OOM)
        torch.cuda.set_per_process_memory_fraction(0.90)

    # Cargar modelo con pesos pre-entrenados (Transfer Learning)
    # Si yolov8s-seg.pt no está descargado, se descarga automáticamente
    model = YOLO(BASE_MODEL)

    # Entrenar
    results = model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        name=RUN_NAME,
        patience=PATIENCE,
        workers=WORKERS,
        pretrained=True,       # Transfer Learning desde pesos COCO
        optimizer="AdamW",     # AdamW es más estable que SGD para datasets pequeños
        lr0=0.001,             # Learning rate inicial
        lrf=0.01,              # Learning rate final (fracción de lr0)
        weight_decay=0.0005,
        warmup_epochs=3.0,     # Épocas de warmup del LR
        amp=True,              # ✅ Precisión mixta FP16 — reduce VRAM ~50% sin perder precisión
        cache=False,           # ✅ No cachear imágenes en RAM (evita OOM en sistema)
        # ── Augmentation integrado ──
        degrees=DEGREES,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        scale=SCALE,
        # ── Guardado ──
        save=True,
        save_period=25,        # Guardar checkpoint cada N épocas
        plots=True,            # Generar gráficas de métricas
        verbose=True,
    )

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)

    best_model_path = Path(f"runs/segment/{RUN_NAME}/weights/best.pt")
    if best_model_path.exists():
        print(f"\n  Mejor modelo: {best_model_path}")
    else:
        print(f"\n  Resultados en: runs/segment/{RUN_NAME}/")

    # Evaluar en test set
    print("\n[INFO] Evaluando en test set...")
    metrics = model.val(data=str(DATASET_YAML.resolve()), split="test", workers=0)

    print("\n[MÉTRICAS EN TEST SET]")
    print(f"  mAP50 (box):  {metrics.box.map50:.4f}")
    print(f"  mAP50 (seg):  {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")

    print("\n[LISTO] Para hacer predicciones, ejecuta:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model_path}')")
    print(f"  model.predict(source=0, show=True)  # Webcam en tiempo real")


if __name__ == "__main__":
    train()
