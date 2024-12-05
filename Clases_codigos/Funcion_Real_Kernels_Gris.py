import cv2
import numpy as np

def convolution(image, K):
    (filas, columnas) = image.shape[:2]#obtener las dimensiones de la imagen de entrada
    (kfilas, kcolumnas) = K.shape[:2]#obtener las dimensiones del kernel
    pad = (kfilas - 1) // 2#obtener el tamaño del padding
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)#agregar padding
    output = np.zeros((filas, columnas), dtype="float")#crear la imagen de salida
    for x in np.arange(pad, filas + pad):#recorrer la imagen de entrada desde el padding hasta el final
        for y in np.arange(pad, columnas + pad):#recorrer la imagen de entrada desde el padding hasta el final
            roi = image[x - pad:x + pad + 1, y - pad:y + pad + 1]
            k = (roi * K).sum()
            output[x - pad, y - pad] = k

    #normalizamos la imagen de salida entre 0 y 255
    output = 255 * (output - np.amin(output)) / (np.amax(output) - np.amin(output))
    output = output.astype("uint8")
    # output = output[pad:filas + pad, pad:columnas + pad]
    return output

if __name__ == "__main__":
    image = cv2.imread(r'Practica1\Reto Colores\pico-y-placa-2022.jpg')
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    cv2.imshow("image", convolution(image, kernel))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


