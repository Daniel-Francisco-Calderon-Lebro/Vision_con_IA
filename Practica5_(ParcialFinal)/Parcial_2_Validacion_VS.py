from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Configuración de la ruta de la base de datos
base_path = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica5_(ParcialFinal)\Base_datos\Grietas_NoGrietas'
folder = r'\Validacion'  # Agregar 'r' para indicar que es una ruta en crudo (raw string)
sub_folders = [r'\Validacion_Grietas', r'\Validacion_No_Grietas']  # Corregir rutas con 'r' para manejar correctamente las barras invertidas

# Cargar datos e inicializar etiquetas
data = []
labels = []

for idx, subfolder in enumerate(sub_folders):
    # Asegurarse de que la ruta está bien concatenada
    subfolder_path = base_path + folder + subfolder  # Concatenamos la ruta base con la carpeta y subcarpeta
    
    files = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
    for file in files:
        # Cargar imagen y procesarla
        img_path = os.path.join(subfolder_path, file)  # Usar os.path.join para concatenar rutas correctamente
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Ajustar tamaño de imagen al tamaño de entrada del modelo
        img = img.astype('float32') / 255.0  # Normalización
        data.append(img)
        labels.append(idx)  # Asignar 0 para 'Sin_Grietas' y 1 para 'Con_Grietas'

# Convertir listas a arreglos de numpy
data = np.array(data)
labels = np.array(labels)

# Imprimir información sobre los datos cargados
print(f"Cantidad de imágenes cargadas: {len(data)}")
print(f"Dimensiones de las imágenes: {data[0].shape}")
print(f"Etiquetas únicas: {np.unique(labels)}")

# Ya las imágenes están listas para ser utilizadas en la validación


# Configuración de la ruta del modelo
path_model = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica5_(ParcialFinal)\Modelos_Generados\CrackDetector2.h5'

# Intentar cargar el modelo
try:
    model = load_model(path_model)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")


# Realizar las predicciones
y_pred = model.predict(np.array(data))  # Asegúrate de convertir los datos en un numpy array
y_pred = (y_pred > 0.5).astype(int)  # Convertir las probabilidades a clases binarias

# Mostrar reporte de clasificación
from sklearn.metrics import classification_report
print(classification_report(labels, y_pred))

