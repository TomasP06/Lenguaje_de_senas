"""
=============================================================
  ANOTADOR DE BOUNDING BOXES - Lengua de Señas Colombiana
  Formato de salida: YOLO  (clase cx cy w h  normalizados)
=============================================================

Controles (con la ventana de imagen en foco):
  CLIC + ARRASTRAR  → Dibujar bounding box sobre la mano
  ENTER / ESPACIO   → Guardar anotación y pasar a la siguiente imagen
  Z                 → Deshacer el box actual (volver a dibujar)
  S                 → Saltar imagen sin anotar
  Q / ESC           → Guardar y salir
=============================================================
"""

import cv2
import os
import glob

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
DATASET_DIR = "dataset"

LETRAS_LSC = [
    'A', 'B', 'C', 'D', 'E', 'F', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

# Colores
COLOR_BOX    = (0, 220, 100)
COLOR_TEMP   = (0, 180, 255)
COLOR_PANEL  = (40, 40, 40)
COLOR_BLANCO = (255, 255, 255)
COLOR_AMARILLO = (0, 220, 220)
COLOR_ROJO   = (0, 80, 220)

# ─────────────────────────────────────────────
#  ESTADO GLOBAL DEL MOUSE
# ─────────────────────────────────────────────
mouse_x, mouse_y = 0, 0
dibujando   = False
x1, y1      = -1, -1
x2, y2      = -1, -1
box_listo   = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, dibujando, x1, y1, x2, y2, box_listo

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        box_listo = False
        x1, y1 = x, y
        x2, y2 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dibujando:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False
        x2, y2 = x, y
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            box_listo = True

# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def yolo_coords(img_w, img_h, bx1, by1, bx2, by2):
    """Convierte coordenadas pixel a formato YOLO normalizado."""
    cx = ((bx1 + bx2) / 2) / img_w
    cy = ((by1 + by2) / 2) / img_h
    w  = abs(bx2 - bx1) / img_w
    h  = abs(by2 - by1) / img_h
    return cx, cy, w, h


def guardar_yolo(img_path, clase_idx, bx1, by1, bx2, by2, img_w, img_h):
    """Guarda el archivo .txt YOLO junto a la imagen."""
    cx, cy, w, h = yolo_coords(img_w, img_h, bx1, by1, bx2, by2)
    txt_path = os.path.splitext(img_path)[0] + ".txt"
    with open(txt_path, 'w') as f:
        f.write(f"{clase_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return txt_path


def ya_anotada(img_path):
    """Verifica si la imagen ya tiene su archivo .txt."""
    return os.path.exists(os.path.splitext(img_path)[0] + ".txt")


def recopilar_imagenes():
    """Recopila todas las imágenes del dataset con su clase correspondiente."""
    imagenes = []
    for letra in LETRAS_LSC:
        patron = os.path.join(DATASET_DIR, letra, "**", "*.jpg")
        for img_path in glob.glob(patron, recursive=True):
            imagenes.append((img_path, letra))
    return imagenes


def dibujar_panel(frame, letra, clase_idx, img_idx, total, anotada, bx1, by1, bx2, by2):
    """Dibuja la barra de información inferior."""
    h, w = frame.shape[:2]
    panel_h = 90
    cv2.rectangle(frame, (0, h - panel_h), (w, h), COLOR_PANEL, -1)

    # Clase / letra
    cv2.putText(frame, f"Clase: {letra} (idx {clase_idx})", (10, h - panel_h + 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_AMARILLO, 2)

    # Progreso
    cv2.putText(frame, f"Imagen {img_idx + 1} / {total}", (10, h - panel_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_BLANCO, 1)

    # Estado anotacion
    if anotada:
        estado = "[YA ANOTADA]"
        color_e = COLOR_AMARILLO
    elif box_listo:
        estado = "[BOX LISTO - ENTER para guardar]"
        color_e = COLOR_BOX
    else:
        estado = "[Dibuja el box con el mouse]"
        color_e = (180, 180, 180)
    cv2.putText(frame, estado, (w // 2 - 160, h - panel_h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_e, 1)

    # Coords del box
    if box_listo or dibujando:
        coords = f"Box: ({min(bx1,bx2)},{min(by1,by2)}) -> ({max(bx1,bx2)},{max(by1,by2)})"
        cv2.putText(frame, coords, (10, h - panel_h + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    # Controles
    ctrl = "ENTER=Guardar  Z=Rehacer  S=Saltar  Q=Salir"
    cv2.putText(frame, ctrl, (w - 430, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)


# ─────────────────────────────────────────────
#  PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────

def main():
    global dibujando, x1, y1, x2, y2, box_listo

    imagenes = recopilar_imagenes()
    if not imagenes:
        print("[ERROR] No se encontraron imagenes en dataset/")
        return

    total = len(imagenes)
    print(f"[INFO] {total} imagenes encontradas para anotar.")

    cv2.namedWindow("Anotador LSC", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Anotador LSC", 900, 650)
    cv2.setMouseCallback("Anotador LSC", mouse_callback)

    img_idx = 0

    while img_idx < total:
        img_path, letra = imagenes[img_idx]
        clase_idx = LETRAS_LSC.index(letra)
        anotada = ya_anotada(img_path)

        # Cargar imagen original
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            print(f"[SKIP] No se pudo leer: {img_path}")
            img_idx += 1
            continue

        img_h, img_w = img_orig.shape[:2]

        # Si ya estaba anotada, leer el box existente para mostrarlo
        if anotada:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path) as f:
                parts = f.read().strip().split()
            if len(parts) == 5:
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * img_w)
                y1 = int((cy - bh / 2) * img_h)
                x2 = int((cx + bw / 2) * img_w)
                y2 = int((cy + bh / 2) * img_h)
                box_listo = True

        # Resetear si es imagen nueva sin anotar
        else:
            x1, y1, x2, y2 = -1, -1, -1, -1
            box_listo = False
            dibujando = False

        while True:
            display = img_orig.copy()

            # Dibujar box actual (temporal o confirmado)
            if dibujando or box_listo:
                bx_min, bx_max = min(x1, x2), max(x1, x2)
                by_min, by_max = min(y1, y2), max(y1, y2)
                color = COLOR_TEMP if dibujando else COLOR_BOX
                cv2.rectangle(display, (bx_min, by_min), (bx_max, by_max), color, 2)
                # Cruz en el centro
                cx_px = (bx_min + bx_max) // 2
                cy_px = (by_min + by_max) // 2
                cv2.drawMarker(display, (cx_px, cy_px), color,
                               cv2.MARKER_CROSS, 15, 1)

            # Panel informativo
            dibujar_panel(display, letra, clase_idx, img_idx, total,
                          anotada, x1, y1, x2, y2)

            # Título ventana
            cv2.setWindowTitle("Anotador LSC",
                f"Anotador LSC | {letra} | {img_idx+1}/{total} | {os.path.basename(img_path)}")

            cv2.imshow("Anotador LSC", display)
            key = cv2.waitKey(30) & 0xFF

            # GUARDAR → ENTER (13) o ESPACIO (32)
            if key in (13, 32):
                if box_listo:
                    bx_min, bx_max = min(x1, x2), max(x1, x2)
                    by_min, by_max = min(y1, y2), max(y1, y2)
                    txt = guardar_yolo(img_path, clase_idx,
                                       bx_min, by_min, bx_max, by_max,
                                       img_w, img_h)
                    print(f"[GUARDADO] {os.path.basename(txt)}")
                    img_idx += 1
                    break
                else:
                    print("[AVISO] Dibuja primero un bounding box con el mouse.")

            # DESHACER BOX → Z
            elif key in (ord('z'), ord('Z')):
                x1, y1, x2, y2 = -1, -1, -1, -1
                box_listo = False
                anotada = False
                print("[REHACER] Box eliminado, vuelve a dibujar.")

            # SALTAR IMAGEN → S
            elif key in (ord('s'), ord('S')):
                print(f"[SALTAR] {os.path.basename(img_path)}")
                img_idx += 1
                break

            # IMAGEN ANTERIOR → A (flecha iz) o P
            elif key in (ord('p'), ord('P')):
                if img_idx > 0:
                    img_idx -= 1
                break

            # SALIR → Q o ESC
            elif key in (ord('q'), ord('Q'), 27):
                print("\n[INFO] Saliendo del anotador.")
                cv2.destroyAllWindows()

                # Resumen
                anotadas = sum(1 for p, _ in imagenes if ya_anotada(p))
                print(f"\n{'='*40}")
                print(f"  RESUMEN: {anotadas} / {total} imagenes anotadas")
                print(f"{'='*40}\n")
                return

    cv2.destroyAllWindows()

    # Resumen final
    anotadas = sum(1 for p, _ in imagenes if ya_anotada(p))
    print(f"\n{'='*40}")
    print(f"  COMPLETADO: {anotadas} / {total} imagenes anotadas")
    print(f"  Los archivos .txt YOLO estan en las mismas carpetas que las imagenes.")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
