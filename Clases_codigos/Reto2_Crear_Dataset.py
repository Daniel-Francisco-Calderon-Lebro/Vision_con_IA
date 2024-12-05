import cv2 as cv
import os
import numpy as np

# Ruta base donde se encuentran las imágenes y donde se guardarán las subimágenes
basepath = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica3_(Parcial)'
path1 = '\Parte_1'
# Rutas de las imágenes filtradas
ruta1 = os.path.join(basepath + path1, 'imagen1_filtrada.png')
ruta2 = os.path.join(basepath + path1, 'imagen2_filtrada.png')
ruta3 = os.path.join(basepath + path1, 'imagen3_filtrada.png')

# Función para dividir y guardar las subimágenes de 512x512 en una sola carpeta
def dividir_y_guardar(imagen, nombre_base, directorio_base):
    alto, ancho = imagen.shape[:2]
    size = 512
    contador = 0

    # Crear directorio para las subimágenes si no existe
    if not os.path.exists(directorio_base):
        os.makedirs(directorio_base)

    for i in range(0, alto, size):
        for j in range(0, ancho, size):
            subimagen = imagen[i:i+size, j:j+size]
            # Solo guardar si la subimagen tiene el tamaño correcto
            if subimagen.shape[0] == size and subimagen.shape[1] == size:
                nombre_subimagen = f'{nombre_base}_{contador}.png'
                cv.imwrite(os.path.join(directorio_base, nombre_subimagen), subimagen)
                contador += 1

# Cargar imágenes filtradas
imagen1 = cv.imread(ruta1)
imagen2 = cv.imread(ruta2)
imagen3 = cv.imread(ruta3)

print('Tamaños originales:')
print('El tamaño de la imagen1 es', imagen1.shape, 'pixeles')
print('El tamaño de la imagen2 es', imagen2.shape, 'pixeles')
print('El tamaño de la imagen3 es', imagen3.shape, 'pixeles')

# Función para redimensionar la imagen a múltiplos exactos de 512
def ajustar_a_multiplo_de_512(imagen):
    alto, ancho = imagen.shape[:2]
    nuevo_alto = (alto // 512) * 512
    nuevo_ancho = (ancho // 512) * 512
    return cv.resize(imagen, (nuevo_ancho, nuevo_alto))

# Redimensionar imágenes a múltiplos de 512
imagen1 = ajustar_a_multiplo_de_512(imagen1)
imagen2 = ajustar_a_multiplo_de_512(imagen2)
imagen3 = ajustar_a_multiplo_de_512(imagen3)

print('Tamaños ajustados:')
print('El tamaño de la imagen1 es', imagen1.shape, 'pixeles')
print('El tamaño de la imagen2 es', imagen2.shape, 'pixeles')
print('El tamaño de la imagen3 es', imagen3.shape, 'pixeles')


# Definir un único directorio donde se guardarán todas las subimágenes
path2 = '\Parte_2'
directorio_subimagenes = os.path.join(basepath + path2, 'Subimagenes')

# Dividir y guardar las subimágenes de las tres imágenes en la misma carpeta
dividir_y_guardar(imagen1, 'imagen1', directorio_subimagenes)
dividir_y_guardar(imagen2, 'imagen2', directorio_subimagenes)
dividir_y_guardar(imagen3, 'imagen3', directorio_subimagenes)

print("Subimágenes guardadas correctamente en una sola carpeta.")
