import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Ruta de la imagen
ruta = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica3_(Parcial)\Parte_1\imagen_filtrada.tiff'

# Cargar la imagen en escala de grises
imagen = cv.imread(ruta, 0)
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.show()

# Reducir el tamaño de la imagen
factor_reduccion = 0.20
imagen_reducida = cv.resize(imagen, (0, 0), fx=factor_reduccion, fy=factor_reduccion)

# Aplanar la imagen
imagen_reshaped = imagen_reducida.reshape(-1, 1)

# Aplicar KMeans con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=5)
kmeans.fit(imagen_reshaped)

# Obtener las etiquetas de los clusters
etiquetas = kmeans.labels_

# Mantener solo la clase "agua" (etiqueta 1) y eliminar otras clases
agua_imagen = np.zeros_like(imagen_reshaped)  # Crear un array del mismo tamaño que imagen_reshaped

# Asignar negro a los píxeles clasificados como agua (etiqueta 1) y blanco a los demás
agua_imagen[etiquetas == 1] = 0  # Asignar negro a los píxeles clasificados como agua
agua_imagen[etiquetas != 1] = 255  # Asignar blanco a los demás píxeles

# Reshape de agua_imagen para que coincida con las dimensiones de imagen_reducida
agua_imagen = agua_imagen.reshape(imagen_reducida.shape)  # Reshape para que coincida con la imagen reducida

# Mostrar la imagen segmentada con solo la clase "agua"
plt.figure(figsize=(15, 8))
plt.axis('off')
plt.imshow(agua_imagen, cmap='gray')
plt.title('Clasificación Agua/No Agua')
plt.show()

# Redimensionar la imagen segmentada al tamaño original
agua_reescalada = cv.resize(agua_imagen, (imagen.shape[1], imagen.shape[0]))

# Mostrar la imagen reescalada
plt.figure(figsize=(15, 15))
plt.imshow(agua_reescalada, cmap='gray')
plt.axis('off')
plt.title('Clasificación Agua/No Agua - Tamaño Original')
plt.show()

