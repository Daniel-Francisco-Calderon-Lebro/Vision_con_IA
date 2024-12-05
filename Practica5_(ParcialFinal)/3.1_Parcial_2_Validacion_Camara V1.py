import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo guardado
model = load_model(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica5_(ParcialFinal)\Modelos_Generados\CrackDetectorV1.h5')

# Definir las etiquetas
labels = ['Con Grieta', 'Sin Grieta']

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

# Definir las coordenadas de las 4 regiones fijas para una cámara de 640x360 (x, y, ancho, alto)
regions = [
    (20, 20, 128, 128),      # Cuadro superior izquierdo
    (492, 20, 128, 128),     # Cuadro superior derecho
    (20, 212, 128, 128),     # Cuadro inferior izquierdo
    (492, 212, 128, 128)     # Cuadro inferior derecho
]

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Voltear el frame horizontalmente
    
    if not ret:
        break
    
    for (x, y, w, h) in regions:
        # Extraer la región de interés (ROI)
        roi = frame[y:y + h, x:x + w]
        
        # Preprocesar la ROI para el modelo
        roi_resized = cv2.resize(roi, (128, 128))
        roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_resized = roi_resized.astype("float32") / 255.0
        roi_resized = np.expand_dims(roi_resized, axis=0)  # Añadir una dimensión para el lote
        
        # Hacer la predicción
        predictions_prob = model.predict(roi_resized)
        predicted_class = (predictions_prob > 0.5).astype("int32")[0][0]
        prediction_confidence = predictions_prob[0][0] * 100  # Convertir a porcentaje
        
        # Dibujar el cuadro y etiqueta en el frame original
        color = (0, 0, 255) if predicted_class == 0 else (0, 255, 0)  # Rojo si hay grieta, verde si no
        label = f'{labels[predicted_class]}: {prediction_confidence:.2f}%'
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Mostrar el frame resultante
    cv2.imshow('Frame', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
