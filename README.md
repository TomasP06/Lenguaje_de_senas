# 🤟 Detector de Lengua de Señas Colombiana (LSC)

Sistema de reconocimiento de señas estáticas de la **Lengua de Señas Colombiana (LSC)** en tiempo real mediante segmentación de instancias con **YOLOv8**.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura y Flujo de Trabajo](#arquitectura-y-flujo-de-trabajo)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso — Guía Paso a Paso](#uso--guía-paso-a-paso)
  - [Paso 0: Captura del Dataset](#paso-0-captura-del-dataset)
  - [Paso 1: Anotación con LabelMe](#paso-1-anotación-con-labelme)
  - [Paso 2: Conversión de Anotaciones](#paso-2-conversión-de-anotaciones)
  - [Paso 3: Construcción del Dataset YOLO](#paso-3-construcción-del-dataset-yolo)
  - [Paso 4: Data Augmentation](#paso-4-data-augmentation)
  - [Paso 5: Entrenamiento del Modelo](#paso-5-entrenamiento-del-modelo)
  - [Paso 6: Detección en Tiempo Real](#paso-6-detección-en-tiempo-real)
- [Descripción de los Scripts](#descripción-de-los-scripts)
- [Clases Reconocidas (Alfabeto LSC)](#clases-reconocidas-alfabeto-lsc)
- [Hiperparámetros de Entrenamiento](#hiperparámetros-de-entrenamiento)
- [Resultados](#resultados)
- [Autor](#autor)

---

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de visión por computadora para el reconocimiento de **señas estáticas** del alfabeto de la Lengua de Señas Colombiana. Comprende desde la captura y anotación de datos hasta el entrenamiento de un modelo de segmentación de instancias y su despliegue en tiempo real mediante webcam.

**Tecnologías utilizadas:**
- 🧠 **YOLOv8 Segmentation** — modelo de segmentación de instancias (Ultralytics)
- 📷 **OpenCV** — captura de video y visualización
- 🏷️ **LabelMe** — herramienta de anotación de polígonos
- 🔄 **Albumentations** — augmentación de datos para polígonos de segmentación
- ⚡ **PyTorch + CUDA** — aceleración GPU durante el entrenamiento

---

## Arquitectura y Flujo de Trabajo

```
Captura de imágenes         Anotación con LabelMe
  (capturar_dataset.py)  →  (run_labelme.bat)
           │                         │
           ▼                         ▼
     dataset/                  dataset/*.json
  (imágenes crudas)          (anotaciones de polígonos)
           │                         │
           └──────────┬──────────────┘
                      ▼
         Conversión JSON → TXT YOLO
          (labelme2yolo_converter.py)
                      │
                      ▼
         Construcción estructura YOLO
          (build_yolo_structure.py)
                      │
                      ▼
         yolo_dataset/ (train 80% / val 10% / test 10%)
                      │
                      ▼
         Data Augmentation (x5 imágenes en train)
          (data_augmentation.py)
                      │
                      ▼
         Entrenamiento con Transfer Learning
          (train_yolo.py)
                      │
                      ▼
         runs/segment/<run_name>/weights/best.pt
                      │
                      ▼
         Detección en Tiempo Real (Webcam)
          (test_webcam.py)
```

---

## Estructura del Repositorio

```
ProyectoVision/
│
├── 📄 README.md                      # Este archivo
├── 📄 .gitignore                     # Archivos excluidos del repositorio
│
├── 🐍 capturar_dataset.py            # Paso 0: Captura de imágenes con webcam
├── 🐍 labelme2yolo_converter.py      # Paso 1: Conversión LabelMe JSON → YOLO TXT
├── 🐍 build_yolo_structure.py        # Paso 2: Organización del dataset (train/val/test)
├── 🐍 data_augmentation.py           # Paso 3: Aumento del dataset de entrenamiento
├── 🐍 train_yolo.py                  # Paso 4: Entrenamiento del modelo YOLOv8
├── 🐍 test_webcam.py                 # Paso 5: Inferencia en tiempo real con webcam
│
├── 🦇 run_labelme.bat                # Atajo para lanzar LabelMe en Windows
│
├── 📁 dataset/                       # Dataset crudo (imágenes + JSON de LabelMe)
│   ├── classes.txt                   # Lista de clases en orden (A, B, C, ...)
│   ├── log_captura.json              # Log automático de capturas
│   └── <LETRA>/                      # Subcarpeta por letra del alfabeto LSC
│       └── <VOLUNTARIO>/             # Subcarpeta por voluntario (V01, V02, ...)
│           ├── *.jpg                 # Imágenes capturadas
│           └── *.json                # Anotaciones de LabelMe
│
├── 📁 yolo_dataset/                  # Dataset preparado para YOLOv8 (generado)
│   ├── dataset.yaml                  # Configuración del dataset para YOLO
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── 📁 runs/                          # Resultados de entrenamiento (generado)
│   └── segment/
│       └── <run_name>/
│           ├── weights/
│           │   ├── best.pt           # Mejor modelo guardado
│           │   └── last.pt           # Último checkpoint
│           └── *.png                 # Gráficas de métricas
│
├── 📦 yolov8n-seg.pt                 # Pesos base YOLOv8 Nano (Transfer Learning)
├── 📦 yolov8s-seg.pt                 # Pesos base YOLOv8 Small (Transfer Learning)
└── 📁 venv_py311/                    # Entorno virtual Python 3.11 (no incluido en git)
```

> **Nota:** Las carpetas `dataset/`, `runs/`, `venv/` y los archivos `.pt` están excluidos del repositorio en `.gitignore` por su tamaño. Ver la sección [Instalación](#instalación) para obtenerlos.

---

## Requisitos

### Hardware
| Componente | Mínimo | Recomendado |
|---|---|---|
| CPU | Cualquier x64 moderno | Intel/AMD 6+ núcleos |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU lento) | NVIDIA con CUDA ≥ 6 GB VRAM |
| Cámara | Webcam 640×480 | Webcam 1080p |

### Software
- **Python 3.11** (probado con esta versión)
- **CUDA Toolkit** (opcional, para aceleración GPU — recomendado)
- **Git**

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/TomasP06/Lenguaje_de_senas.git
cd Lenguaje_de_senas
```

### 2. Crear y activar el entorno virtual

```bash
# Windows
python -m venv venv_py311
venv_py311\Scripts\activate

# Linux / macOS
python3.11 -m venv venv_py311
source venv_py311/bin/activate
```

### 3. Instalar dependencias

```bash
# Instalar PyTorch con soporte CUDA (recomendado para NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar el resto de dependencias
pip install ultralytics opencv-python albumentations labelme
```

> **Sin GPU:** Si no tienes GPU NVIDIA, instala PyTorch estándar:
> ```bash
> pip install torch torchvision torchaudio
> pip install ultralytics opencv-python albumentations labelme
> ```

### 4. Verificar instalación

```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Uso — Guía Paso a Paso

> Sigue los pasos en orden. Cada script valida que el paso anterior se haya completado.

---

### Paso 0: Captura del Dataset

Captura imágenes de cada seña con tu webcam.

```bash
python capturar_dataset.py
```

**Controles del programa:**

| Tecla | Acción |
|---|---|
| `ESPACIO` | Capturar foto |
| `N` | Siguiente letra |
| `P` | Letra anterior |
| `R` | Cambiar voluntario (V01 → V02 → ...) |
| `D` | Eliminar última foto capturada |
| `Q` / `ESC` | Salir |

Las imágenes se guardan en `dataset/<LETRA>/<VOLUNTARIO>/`.

**Configuración** (editar al inicio de `capturar_dataset.py`):
```python
FOTOS_POR_LETRA = 3     # Número de fotos objetivo por letra por voluntario
CAMARA_INDEX = 0        # Índice de la cámara (0 = cámara por defecto)
```

---

### Paso 1: Anotación con LabelMe

Abre LabelMe para anotar los polígonos de cada mano en las imágenes capturadas.

```bash
# Windows (doble clic o desde terminal):
run_labelme.bat

# O directamente:
labelme
```

**Instrucciones de anotación:**
1. Abre la carpeta `dataset/` en LabelMe.
2. Para cada imagen, dibuja un **polígono** alrededor de la mano.
3. Asigna el nombre de la letra como etiqueta (ej. `A`, `B`, `C`...).
4. Guarda cada anotación — se genera un archivo `.json` junto a cada imagen.

> Asegúrate de que `dataset/classes.txt` contiene todas las clases en el orden correcto (una por línea).

---

### Paso 2: Conversión de Anotaciones

Convierte los archivos JSON de LabelMe al formato TXT que requiere YOLOv8.

```bash
python labelme2yolo_converter.py
```

**Resultado:** Para cada `imagen.json` se genera `imagen.txt` con las coordenadas del polígono normalizadas.

**Formato YOLO Segmentation:**
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Donde todas las coordenadas están normalizadas entre `0.0` y `1.0`.

---

### Paso 3: Construcción del Dataset YOLO

Organiza el dataset en las carpetas `train/val/test` y genera el archivo `dataset.yaml`.

```bash
python build_yolo_structure.py
```

**División del dataset:**
- **80%** → Train
- **10%** → Validación
- **10%** → Test

La división es **estratificada por clase** (garantiza representación de todas las letras en cada split) y reproducible con semilla `42`.

**Resultado:** carpeta `yolo_dataset/` con la estructura requerida por YOLOv8.

---

### Paso 4: Data Augmentation

Aumenta artificialmente el número de imágenes de entrenamiento (×5 por defecto).

```bash
python data_augmentation.py
```

**Transformaciones aplicadas:**
| Transformación | Probabilidad |
|---|---|
| Flip horizontal | 50% |
| Rotación ± 20° + escala ± 15% | 80% |
| Brillo y contraste aleatorio | 70% |
| Ajuste Hue/Saturación/Valor | 50% |
| Blur gaussiano o de movimiento | 30% |
| Ruido gaussiano | 30% |
| CLAHE (mejora de contraste local) | 20% |

> El augmentation **solo se aplica al split de train**, nunca a val ni test.

---

### Paso 5: Entrenamiento del Modelo

Entrena el modelo YOLOv8 con Transfer Learning desde pesos preentrenados en COCO.

```bash
python train_yolo.py
```

**Configuración principal** (editar en `train_yolo.py`):

```python
BASE_MODEL  = "yolov8n-seg.pt"          # nano (rápido) | s (equilibrado) | m (preciso)
RUN_NAME    = "senas_colombianas_v1"    # Nombre del experimento
EPOCHS      = 150                        # Número máximo de épocas
BATCH_SIZE  = 4                          # Ajustar según VRAM disponible
IMG_SIZE    = 640                        # Resolución de entrada
PATIENCE    = 30                         # Early stopping
```

**Selección del modelo según VRAM:**
| Modelo | VRAM recomendada | Velocidad | Precisión |
|---|---|---|---|
| `yolov8n-seg.pt` | ≥ 4 GB | ⚡ Rápido | Buena |
| `yolov8s-seg.pt` | ≥ 6 GB | ✅ Equilibrado | Mejor |
| `yolov8m-seg.pt` | ≥ 10 GB | 🐢 Lento | Óptima |

Los resultados se guardan en `runs/segment/<RUN_NAME>/`:
- `weights/best.pt` — mejor modelo (mayor mAP en validación)
- `weights/last.pt` — último checkpoint
- Gráficas de métricas (curvas de pérdida, mAP, etc.)

---

### Paso 6: Detección en Tiempo Real

Ejecuta el detector en tiempo real usando tu webcam.

```bash
python test_webcam.py
```

El script carga automáticamente el modelo más reciente disponible en `runs/segment/`.
- La imagen se muestra con **efecto espejo** para mayor naturalidad.
- Presiona `Q` en la ventana de video para salir.

**Personalizar el modelo a cargar** (editar `test_webcam.py`):
```python
model_path = "runs/segment/senas_colombianas_v15/weights/best.pt"
```

---

## Descripción de los Scripts

| Script | Descripción |
|---|---|
| `capturar_dataset.py` | Interfaz con webcam para capturar imágenes del dataset organizadas por letra y voluntario. Incluye barra de progreso, log JSON y controles de teclado. |
| `labelme2yolo_converter.py` | Convierte anotaciones de polígonos en formato LabelMe (`.json`) a formato YOLO Segmentation (`.txt`) con coordenadas normalizadas. |
| `build_yolo_structure.py` | Recopila pares imagen+label, hace split estratificado train/val/test y genera `dataset.yaml` para YOLOv8. |
| `data_augmentation.py` | Aplica un pipeline de 7 transformaciones a las imágenes de train respetando los polígonos de segmentación (usa Albumentations con keypoints). |
| `train_yolo.py` | Entrena YOLOv8 Segmentation con Transfer Learning. Configura hiperparámetros, augmentation interno de YOLO, early stopping y evalúa en test set al terminar. |
| `test_webcam.py` | Carga el modelo entrenado e infiere en tiempo real sobre la webcam con efecto espejo, dibujando máscaras y bounding boxes. |
| `run_labelme.bat` | Script de Windows para activar el entorno virtual y lanzar LabelMe con un doble clic. |

---

## Clases Reconocidas (Alfabeto LSC)

El modelo reconoce las siguientes **21 señas estáticas** del alfabeto LSC:

```
A  B  C  D  E  F  I  K
L  M  N  O  P  Q  R  T
U  V  W  X  Y
```

> Las letras **G, H, J, Ñ, S y Z** fueron excluidas porque son señas **dinámicas** (requieren movimiento) o no tienen representación estática aplicable.

---

## Hiperparámetros de Entrenamiento

| Parámetro | Valor | Descripción |
|---|---|---|
| `optimizer` | AdamW | Más estable que SGD para datasets pequeños |
| `lr0` | 0.001 | Learning rate inicial |
| `lrf` | 0.01 | Learning rate final (fracción de lr0) |
| `weight_decay` | 0.0005 | Regularización L2 |
| `warmup_epochs` | 3 | Épocas de calentamiento del LR |
| `amp` | True | Precisión mixta FP16 (reduce VRAM ~50%) |
| `mosaic` | 1.0 | Mosaico de 4 imágenes (muy efectivo con pocos datos) |
| `fliplr` | 0.5 | Flip horizontal aleatorio |
| `degrees` | 10° | Rotación máxima |
| `cache` | False | No cachear en RAM (evita OOM) |

---

## Resultados

Los resultados del entrenamiento (métricas, curvas, matrices de confusión) se guardan automáticamente en:

```
runs/segment/<RUN_NAME>/
├── results.png          # Curvas de pérdida y mAP
├── confusion_matrix.png # Matriz de confusión
├── PR_curve.png         # Curva Precisión-Recall
└── weights/
    ├── best.pt          # Mejor modelo
    └── last.pt          # Último checkpoint
```

Las métricas principales reportadas al finalizar el entrenamiento son:
- **mAP50 (box)**: mAP de las bounding boxes al umbral IoU 0.50
- **mAP50 (seg)**: mAP de las máscaras de segmentación al umbral IoU 0.50
- **mAP50-95 (seg)**: mAP de segmentación promediado sobre IoU 0.50–0.95

---

## Autor

**Tomás Patiño**
Proyecto de Visión por Computadora — Lengua de Señas Colombiana

---

*Desarrollado con ❤️ para mejorar la accesibilidad de la comunicación en Lengua de Señas Colombiana.*
