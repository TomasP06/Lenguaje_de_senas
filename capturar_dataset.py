"""
=============================================================
  CAPTURA DE DATASET - Lengua de Señas Colombiana (LSC)
  Proyecto: Traductor LSC con YOLO
  Autor: Tomas Patiño
=============================================================

Controles:
  ESPACIO     → Capturar foto
  N           → Siguiente letra
  P           → Letra anterior
  R           → Cambiar voluntario
  D           → Eliminar última foto capturada
  Q           → Salir
"""

import cv2
import os
import time
import json
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"          # Carpeta raíz donde se guardan las imágenes
FOTOS_POR_LETRA = 3                # Cuántas fotos se quieren por letra por voluntario
CAMARA_INDEX  = 0                  # Índice de la cámara (0 = cámara por defecto)
RESOLUCION    = (640, 480)         # Resolución de captura

# Alfabeto LSC estático (excluidas: G, H, J, Ñ, S, Z por ser dinámicas o no aplicar)
LETRAS_LSC = [
    'A', 'B', 'C', 'D', 'E', 'F', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

# Colores (BGR)
COLOR_VERDE   = (0, 220, 100)
COLOR_ROJO    = (0, 60, 220)
COLOR_AZUL    = (220, 140, 0)
COLOR_BLANCO  = (255, 255, 255)
COLOR_NEGRO   = (0, 0, 0)
COLOR_GRIS    = (60, 60, 60)
COLOR_AMARILLO= (0, 220, 220)

# ─────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def crear_estructura_carpetas(voluntario):
    """Crea las carpetas del dataset para todas las letras."""
    for letra in LETRAS_LSC:
        ruta = os.path.join(DATASET_DIR, letra, voluntario)
        os.makedirs(ruta, exist_ok=True)


def contar_fotos(letra, voluntario):
    """Cuenta cuántas fotos ya existen para una letra y voluntario."""
    ruta = os.path.join(DATASET_DIR, letra, voluntario)
    if not os.path.exists(ruta):
        return 0
    return len([f for f in os.listdir(ruta) if f.endswith('.jpg')])


def guardar_log(voluntario, letra, archivo):
    """Guarda un log JSON con metadatos de captura."""
    log_path = os.path.join(DATASET_DIR, "log_captura.json")
    log = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "voluntario": voluntario,
        "letra": letra,
        "archivo": archivo
    })
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def dibujar_interfaz(frame, letra_actual, idx_letra, voluntario, fotos_tomadas, ultima_captura):
    """Dibuja la interfaz visual sobre el frame de la cámara."""
    h, w = frame.shape[:2]

    # ── Panel superior ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 80), COLOR_GRIS, -1)

    # Letra actual (grande, centrada)
    texto_letra = f"SENAL:  {letra_actual}"
    cv2.putText(frame, texto_letra, (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, COLOR_VERDE, 3)

    # Progreso letras
    progreso_letra = f"Letra {idx_letra + 1} / {len(LETRAS_LSC)}"
    cv2.putText(frame, progreso_letra, (w - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_BLANCO, 1)

    # Voluntario
    cv2.putText(frame, f"Voluntario: {voluntario}", (w - 230, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_AMARILLO, 1)

    # ── Panel inferior ──────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 120), (w, h), COLOR_GRIS, -1)

    # Barra de progreso de fotos
    max_barra = w - 40
    progreso_px = int((fotos_tomadas / FOTOS_POR_LETRA) * max_barra)
    progreso_px = min(progreso_px, max_barra)

    cv2.rectangle(frame, (20, h - 100), (20 + max_barra, h - 75), (80, 80, 80), -1)
    color_barra = COLOR_VERDE if fotos_tomadas < FOTOS_POR_LETRA else COLOR_AMARILLO
    cv2.rectangle(frame, (20, h - 100), (20 + progreso_px, h - 75), color_barra, -1)

    pct = min(int((fotos_tomadas / FOTOS_POR_LETRA) * 100), 100)
    texto_fotos = f"Fotos: {fotos_tomadas} / {FOTOS_POR_LETRA}  ({pct}%)"
    cv2.putText(frame, texto_fotos, (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_BLANCO, 1)

    # Controles
    controles = "[ESPACIO] Capturar   [N] Siguiente   [P] Anterior   [R] Voluntario   [D] Borrar   [Q] Salir"
    cv2.putText(frame, controles, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # ── Mensaje de última captura ───────────────────────────────
    if ultima_captura and (time.time() - ultima_captura["tiempo"]) < 1.5:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - 140, h//2 - 30), (w//2 + 140, h//2 + 30), COLOR_VERDE, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"[OK] Foto #{ultima_captura['num']} guardada",
                    (w//2 - 130, h//2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_BLANCO, 2)

    # ── Aviso si ya se completó la letra ───────────────────────
    if fotos_tomadas >= FOTOS_POR_LETRA:
        cv2.putText(frame, ">> Completada! Presiona N para continuar",
                    (20, h - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_AMARILLO, 2)

    return frame




def eliminar_ultima_foto(letra, voluntario):
    """Elimina la última foto tomada para la letra/voluntario actuales."""
    ruta = os.path.join(DATASET_DIR, letra, voluntario)
    fotos = sorted([f for f in os.listdir(ruta) if f.endswith('.jpg')])
    if fotos:
        os.remove(os.path.join(ruta, fotos[-1]))
        print(f"[ELIMINADA] {fotos[-1]}")
        return True
    return False


# ─────────────────────────────────────────────
#  PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────

def main():
    voluntario = "V01"  # Cambia esto o usa la tecla R para cambiar de voluntario
    crear_estructura_carpetas(voluntario)

    print(f"\n[INFO] Voluntario: {voluntario}")
    print(f"[INFO] Dataset guardado en: {os.path.abspath(DATASET_DIR)}")
    print(f"[INFO] Fotos objetivo por letra: {FOTOS_POR_LETRA}")
    print("\n[INFO] Abriendo cámara... (presiona Q para salir)\n")

    cap = cv2.VideoCapture(CAMARA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUCION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCION[1])

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifica el índice CAMARA_INDEX.")
        return

    idx_letra     = 0
    ultima_captura = None

    cv2.namedWindow("LSC Dataset Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LSC Dataset Capture", 800, 620)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame de la cámara.")
            break

        # Espejo horizontal para mayor naturalidad
        frame = cv2.flip(frame, 1)

        letra_actual = LETRAS_LSC[idx_letra]
        fotos_tomadas = contar_fotos(letra_actual, voluntario)

        # Dibujar interfaz
        frame = dibujar_interfaz(frame, letra_actual, idx_letra,
                                  voluntario, fotos_tomadas, ultima_captura)

        cv2.imshow("LSC Dataset Capture", frame)

        key = cv2.waitKey(30) & 0xFF

        # ── CAPTURAR (ESPACIO) ──────────────────────────────────
        if key == ord(' '):
            fotos_actuales = contar_fotos(letra_actual, voluntario)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
            nombre_archivo = f"{letra_actual}_{voluntario}_{timestamp}.jpg"
            ruta_guardado = os.path.join(DATASET_DIR, letra_actual, voluntario, nombre_archivo)

            # Guardar sin overlay de interfaz (frame limpio)
            ret2, frame_limpio = cap.read()
            if ret2:
                frame_limpio = cv2.flip(frame_limpio, 1)
                cv2.imwrite(ruta_guardado, frame_limpio)
                guardar_log(voluntario, letra_actual, nombre_archivo)
                ultima_captura = {"tiempo": time.time(), "num": fotos_actuales + 1}
                print(f"[GUARDADA] {nombre_archivo}")

        # ── SIGUIENTE LETRA (N) ─────────────────────────────────
        elif key == ord('n') or key == ord('N'):
            if idx_letra < len(LETRAS_LSC) - 1:
                idx_letra += 1
                ultima_captura = None
                print(f"[LETRA] -> {LETRAS_LSC[idx_letra]}")
            else:
                print("[INFO] Ya estás en la última letra.")

        # ── LETRA ANTERIOR (P) ─────────────────────────────────
        elif key == ord('p') or key == ord('P'):
            if idx_letra > 0:
                idx_letra -= 1
                ultima_captura = None
                print(f"[LETRA] <- {LETRAS_LSC[idx_letra]}")
            else:
                print("[INFO] Ya estás en la primera letra.")

        # ── CAMBIAR VOLUNTARIO (R) ──────────────────────────────
        elif key == ord('r') or key == ord('R'):
            cap.release()
            cv2.destroyAllWindows()
            ids = [f"V{str(i).zfill(2)}" for i in range(1, 11)]
            actual = ids.index(voluntario) if voluntario in ids else 0
            voluntario = ids[(actual + 1) % len(ids)]
            crear_estructura_carpetas(voluntario)
            cap = cv2.VideoCapture(CAMARA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUCION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUCION[1])
            ultima_captura = None
            print(f"[INFO] Cambiado a voluntario: {voluntario}")

        # ── ELIMINAR ÚLTIMA FOTO (D) ────────────────────────────
        elif key == ord('d') or key == ord('D'):
            if eliminar_ultima_foto(letra_actual, voluntario):
                ultima_captura = None

        # ── SALIR (Q o ESC) ─────────────────────────────────────
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("\n[INFO] Cerrando captura.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Resumen final ────────────────────────────────────────────
    print("\n" + "="*50)
    print("  RESUMEN DE CAPTURA")
    print("="*50)
    total = 0
    for letra in LETRAS_LSC:
        n = contar_fotos(letra, voluntario)
        estado = "OK" if n >= FOTOS_POR_LETRA else f"{n}/{FOTOS_POR_LETRA}"
        print(f"  {letra}: {estado}")
        total += n
    print(f"\n  Total fotos capturadas: {total}")
    print(f"  Guardadas en: {os.path.abspath(DATASET_DIR)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
