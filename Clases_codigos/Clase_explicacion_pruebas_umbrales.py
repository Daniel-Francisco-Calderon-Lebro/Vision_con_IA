import cv2

image = cv2.imread(r'Practica1\Reto Colores\pico-y-placa-2022.jpg')

print("shape:", image.shape)
print("Minimo:", image.min())
print("Maximo:", image.max())
print("Media:", image.mean())
print("Desviacion Estandar:", image.std())

print("pixel:", image[10, 10])

umbral = cv2.inRange(image, (0, 0, 0), (150, 150, 150))#pueden ser valores de 0 a 255 con maximo y minimo

countours, hierarchy = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("countours:", len(countours))

print("hierarchy:", len(hierarchy))

#DIBUJAR LOS CONTORNOS EN LA IMAGEN

cv2.drawContours(image, countours, -1, (0, 255, 0), 3)
cv2.imshow("image", image)
cv2.waitKey(0)

#REDIMENSIONAR LA IMAGEN

# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow("image", image)
# cv2.waitKey(0)

#COLOCAR TEXTO EN LA IMAGEN GRANDE EN LA IMAGEN REDIMENSIONADA

# cv2.putText(image, "Pico y Placa", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# cv2.imshow("image", image)
# cv2.waitKey(0)


