import random
import matplotlib.pyplot as plt
import time


def programa_sin_numpy():
    """-Crear una matriz de 1000x1000 en Python de números aleatorios de 0 a 255. Pueden ser del tipo enteros sin signo de 8 bits (uint8). """
    def crear_imagen_de_0_255(filas, columnas):
        return [[random.randint(0, 255) for _ in range(columnas)] for _ in range(filas)]
    """-Mostrar la imagen en una ventana con Matplotlib esto es opcional"""
    def mostrar_imagen(imagen):
        plt.imshow(imagen)
        plt.axis('off')
        plt.show()
    """-Hacer una función que permita obtener el mínimo, máximo, media y desviación estándar de todos los valores de esta matriz usando fórmulas matemáticas."""
    def datos_estadisticos(imagen):
        datos = []

        for fila in imagen:
            for pixel in fila:
                datos.append(pixel)

        return min(datos), max(datos), sum(datos) / len(datos), sum((x - sum(datos) / len(datos)) ** 2 for x in datos) ** 0.5


    """-Aplanar la matriz (eliminar una dimensión, que sea un vector)"""
    def aplanar_imagen(imagen):
        imagenaplanada = []

        for fila in imagen:
            for pixel in fila:
                imagenaplanada.append(pixel)

        return imagenaplanada
    
    """-Guardar en archivo de texto plano en el PC (.csv, .txt, .out)."""
    def generador_archivos(imagenaplanada):
        # Generar un Archivo de Texto plano .txt
        with open("imagenaplanada.txt", "w") as archivo:
            for pixel in imagenaplanada:
                # Escribir el pixel separado por comas
                if pixel != imagenaplanada[-1]:
                    archivo.write(str(pixel) + ",")
                else:
                    archivo.write(str(pixel))

        # Generar un Archivo de Texto plano .csv
        with open("imagenaplanada.csv", "w") as archivo:
            for pixel in imagenaplanada:
                # Escribir el pixel separado por comas
                if pixel != imagenaplanada[-1]:
                    archivo.write(str(pixel) + ",")
                else:
                    archivo.write(str(pixel))

        # Generar un Archivo de Texto plano .tsv
        with open("imagenaplanada.tsv", "w") as archivo:
            for pixel in imagenaplanada:
                # Escribir el pixel separado por comas
                if pixel != imagenaplanada[-1]:
                    archivo.write(str(pixel) + "\t")
                else:
                    archivo.write(str(pixel))

        # Generar un Archivo de Texto plano .out
        with open("imagenaplanada.out", "w") as archivo:
            for pixel in imagenaplanada:
                # Escribir el pixel separado por comas
                if pixel != imagenaplanada[-1]:
                    archivo.write(str(pixel) + " ")
                else:
                    archivo.write(str(pixel))

    # Crear la imagen tipo matriz en formato de 0 a 255 de 1000x1000
    imagen = crear_imagen_de_0_255(1000, 1000)
    
    # Mostrar la imagen esto es opcional
    #mostrar_imagen(imagen)

    # Mostrar valores
    dato1, dato2, dato3, dato4 = datos_estadisticos(imagen)
    print("Menor valor:", dato1)
    print("Mayor valor:", dato2)
    print("Promedio:", dato3)
    print("Desviación estándar:", dato4)

    # Imprimir la imagen aplanada
    imagenaplanada = aplanar_imagen(imagen)
    
    # print(imagenaplanada)
    # print(len(imagenaplanada)) # Imprimir la longitud de la imagen
    
    # Generar Archivos
    generador_archivos(imagenaplanada)



if __name__ == "__main__":
    #calcular el tiempo de ejecución
    start_time = time.time()
    programa_sin_numpy()
    print("--- %s seconds ---" % (time.time() - start_time))
    
