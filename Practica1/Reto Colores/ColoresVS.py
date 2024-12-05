#importamos las librerías necesarias
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Cargamos la imagen

img = cv2.imread(r"C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica1\Reto Colores\pico-y-placa-2022.jpg",1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Extraemos 5 zonas de la imagen

#agregamos un subplot para mostrar las imagenes

# Crear una figura con 6 subplots en una cuadrícula de 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Definir y mostrar cada ROI en un subplot diferente
roi1 = img[130:200, 300:400]
axs[0, 0].imshow(roi1)
axs[0, 0].axis('off')  # Opcional: oculta los ejes
#obtenemos la media de R G y B
avg1R = float('%.1f' % roi1[:, :, 0].mean())
avg1G = float('%.1f' % roi1[:, :, 1].mean())
avg1B = float('%.1f' % roi1[:, :, 2].mean())

axs[0, 0].set_title('R: ' + str(avg1R) + ' G: ' + str(avg1G) + ' B: ' + str(avg1B))


roi2 = img[340:390,140:250]
axs[0, 1].imshow(roi2)
axs[0, 1].axis('off')  # Opcional: oculta los ejes
#obtenemos la media de R G y B
avg2R = float('%.1f' % roi2[:, :, 0].mean())
avg2G = float('%.1f' % roi2[:, :, 1].mean())
avg2B = float('%.1f' % roi2[:, :, 2].mean())

axs[0, 1].set_title('R: ' + str(avg2R) + ' G: ' + str(avg2G) + ' B: ' + str(avg2B))

roi3 = img[30:80, 140:185]
axs[1, 0].imshow(roi3)
axs[1, 0].axis('off')  # Opcional: oculta los ejes
avg3R = float('%.1f' % roi3[:, :, 0].mean())
avg3G = float('%.1f' % roi3[:, :, 1].mean())
avg3B = float('%.1f' % roi3[:, :, 2].mean())

axs[1, 0].set_title('R: ' + str(avg3R) + ' G: ' + str(avg3G) + ' B: ' + str(avg3B))


roi4 = img[20:100, 1000:1150]
axs[1, 1].imshow(roi4)
axs[1, 1].axis('off')  # Opcional: oculta los ejes
avg4R = float('%.1f' % roi4[:, :, 0].mean())
avg4G = float('%.1f' % roi4[:, :, 1].mean())
avg4B = float('%.1f' % roi4[:, :, 2].mean())

axs[1, 1].set_title('R: ' + str(avg4R) + ' G: ' + str(avg4G) + ' B: ' + str(avg4B))


roi5 = img[560:580, 205:280]
axs[2, 0].imshow(roi5)
axs[2, 0].axis('off')  # Opcional: oculta los ejes
avg5R = float('%.1f' % roi5[:, :, 0].mean())
avg5G = float('%.1f' % roi5[:, :, 1].mean())
avg5B = float('%.1f' % roi5[:, :, 2].mean())

axs[2, 0].set_title('R: ' + str(avg5R) + ' G: ' + str(avg5G) + ' B: ' + str(avg5B))


# El último subplot (axs[2, 1]) queda vacío
axs[2, 1].axis('off')  # Opcional: oculta los ejes para el subplot vacío

# Mostrar la figura con los subplots
plt.show()

#Dibujamos un rectangulo sobre la imagen y le agregamos el texto

cv2.rectangle(img, (300, 130), (400, 200), (0, 255, 0), 3)
cv2.putText(img, 'Media R: ' + str(avg1R), (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media G: ' + str(avg1G), (300, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media B: ' + str(avg1B), (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.rectangle(img, (140, 340), (250, 390), (0, 255, 0), 3)
cv2.putText(img, 'Media R: ' + str(avg2R), (140, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media G: ' + str(avg2G), (140, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media B: ' + str(avg2B), (140, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.rectangle(img, (140, 30), (185, 80), (0, 255, 0), 3)
cv2.putText(img, 'Media R: ' + str(avg3R), (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media G: ' + str(avg3G), (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media B: ' + str(avg3B), (140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.rectangle(img, (1000, 20), (1150, 100), (0, 255, 0), 3)
cv2.putText(img, 'Media R: ' + str(avg4R), (1000, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media G: ' + str(avg4G), (1000, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media B: ' + str(avg4B), (1000, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.rectangle(img, (205, 560), (280, 580), (0, 255, 0), 3)
cv2.putText(img, 'Media R: ' + str(avg5R), (205, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media G: ' + str(avg5G), (205, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv2.putText(img, 'Media B: ' + str(avg5B), (205, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

plt.imshow(img)
plt.show()
