import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo guardado
model = load_model(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica5_(ParcialFinal)\Modelos_Generados\CrackDetectorV1.h5')

# Definir las etiquetas
labels = ['Con Grieta', 'Sin Grieta']

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocesar la imagen
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Asegurarse de que la imagen esté en RGB
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Añadir una dimensión para el lote
    
    # Hacer la predicción
    predictions_prob = model.predict(img)
    predicted_class = (predictions_prob > 0.5).astype("int32")[0][0]
    prediction_confidence = predictions_prob[0][0] * 100  # Convertir a porcentaje
    
    # Determinar el color y la etiqueta en función de la predicción
    if predicted_class == 0:  # 'Con Grieta'
        label = f'{labels[predicted_class]}: {prediction_confidence:.2f}%'
        color = (0, 0, 255)  # Rojo
    else:  # 'Sin Grieta'
        label = f'{labels[predicted_class]}: {prediction_confidence:.2f}%'
        color = (255, 0, 0)  # Azul
    
    # Dibujar el cuadro y mostrar el porcentaje
    cv2.rectangle(frame, (10, 10), (400, 60), color, -1)
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Mostrar el frame resultante
    cv2.imshow('Frame', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()