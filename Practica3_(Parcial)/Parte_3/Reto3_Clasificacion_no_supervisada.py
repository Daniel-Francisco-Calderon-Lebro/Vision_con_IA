import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Ruta de la imagen
ruta = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica3_(Parcial)\Parte_1\imagen_filtrada.tiff'

imagen = cv.imread(ruta,0)
# plt.figure(figsize=(15, 15))
# plt.axis('off')
# plt.imshow(imagen, cmap='gray')
# print(imagen.shape)
# plt.show()

factor_reduccion = 0.20
imagen_reducida = cv.resize(imagen, (0, 0), fx=factor_reduccion, fy=factor_reduccion)

# plt.figure(figsize=(15, 15))
# plt.axis('off')
# plt.imshow(imagen_reducida, cmap='gray')
# print(imagen_reducida.shape)
# plt.show()

# Se aplana la imagen
imagen_reshaped = imagen_reducida.reshape(-1, 1)
print(imagen_reshaped.shape)

# Aplicar KMeans con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=5)
kmeans.fit(imagen_reshaped)
print('Centroides de cada cluster (Grises):', kmeans.cluster_centers_)

# Obtener las etiquetas de los clusters (asignación de cada píxel a un cluster)
etiquetas = kmeans.labels_
print('ETIQUETAS | Vegetación = 0 | Agua = 1 | Montaña = 2 | ', etiquetas)

# Convertir las etiquetas de vuelta a la forma original de la imagen (2D)
etiquetas_imagen = etiquetas.reshape(imagen_reducida.shape[:2])

# Crear una nueva imagen para visualizar los clusters en escala de grises
colores = np.zeros_like(imagen_reducida)

# Asignar niveles de gris a cada cluster según las etiquetas
# Cluster 0: Vegetación (blanco)
# Cluster 1: Agua (negro)
# Cluster 2: Montañas (gris claro)
colores[etiquetas_imagen == 0] = 255   # Vegetación (blanco)
colores[etiquetas_imagen == 1] = 0     # Agua (negro)
colores[etiquetas_imagen == 2] = 100    # Montañas (gris intermedio)

# Mostrar la imagen segmentada con los diferentes clusters
plt.figure(figsize=(15, 8))  # Ajustar el tamaño para que la imagen se vea mejor
plt.axis('off')

# Mostrar la imagen con la segmentación
img = plt.imshow(colores, cmap='gray')

# # Agregar una barra de colores para visualizar los clusters
# cbar = plt.colorbar(img, orientation='vertical', fraction=0.046, pad=0.04)
# cbar.set_ticks([0, 100, 255])  # Ajustar los niveles de gris para la barra de color
# cbar.set_ticklabels(['Agua', 'Montañas', 'Vegetación'])  # Etiquetas personalizadas
# plt.show()


# Redimensionar la imagen segmentada al tamaño original
colores_reescalados = cv.resize(colores, (imagen.shape[1], imagen.shape[0]))

# # Mostrar la imagen reescalada usando matplotlib
# plt.figure(figsize=(15, 15))  # Tamaño de la figura ajustado para mejor visualización

# # No es necesario convertir a RGB, ya que es una imagen en escala de grises
# plt.imshow(colores_reescalados, cmap='gray')  # Usar cmap 'gray' para imágenes en escala de grises
# plt.axis('off')  # Ocultar los ejes
# plt.title('Imagen Segmentada por K-Means (K=3)')
# plt.show()

# Gráfica de dispersión (puedes cambiar imagen_reshaped si usas imagen en color)
plt.figure(figsize=(10, 6))
plt.scatter(imagen_reshaped[:, 0], imagen_reshaped[:, 0], c=etiquetas, cmap='viridis', alpha=0.5)  # Usar el mismo canal para visualización
plt.title('Gráfica de Puntos de Colores de Píxeles')
plt.xlabel('Valor de Grises')
plt.ylabel('Valor de Grises')
plt.colorbar(label='Clusters')
plt.grid()
plt.show()
# Guardar el modelo entrenado de K-Means
filename = 'kmeans_model.pkl'
joblib.dump(kmeans, filename)
print(f'Modelo K-Means guardado en {filename}')
