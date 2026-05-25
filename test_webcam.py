import cv2
from ultralytics import YOLO
import os

# Cargar el modelo entrenado (se usa senas_colombianas_v15 como el último entrenamiento completado)
model_path = "runs/segment/senas_colombianas_v15/weights/best.pt"

if not os.path.exists(model_path):
    # Intentar buscar automáticamente el último entrenamiento si la v15 no existe
    import glob
    posibles_modelos = sorted(glob.glob("runs/segment/*/weights/best.pt"), key=os.path.getmtime)
    if posibles_modelos:
        model_path = posibles_modelos[-1]
        print(f"⚠ No se encontró el modelo en v15. Usando el más reciente encontrado: {model_path}")
    else:
        print(f"❌ Error: No se encontró ningún modelo best.pt en runs/segment/.")

print(f"Cargando modelo desde: {model_path}")
model = YOLO(model_path)

# Iniciar captura de la webcam (cámara 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: No se pudo acceder a la webcam.")
    exit()

print("\n" + "="*60)
print("  INICIANDO WEBCAM CON EFECTO ESPEJO 🪞")
print("  Presiona la tecla 'q' en la ventana de video para salir.")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: No se pudo recibir la imagen de la cámara.")
        break

    # 1. Aplicar efecto espejo (voltear horizontalmente: 1)
    mirror_frame = cv2.flip(frame, 1)

    # 2. Hacer predicciones sobre el frame espejo
    # verbose=False para no saturar la consola de texto
    results = model.predict(source=mirror_frame, conf=0.15, verbose=False)

    # 3. Obtener el frame con las segmentaciones y cajas dibujadas por YOLOv8
    annotated_frame = results[0].plot()

    # 4. Mostrar la ventana con el frame anotado y el efecto espejo
    cv2.imshow("Detector de Senas (Espejo)", annotated_frame)

    # 5. Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
print("Webcam cerrada. ¡Prueba finalizada!")


