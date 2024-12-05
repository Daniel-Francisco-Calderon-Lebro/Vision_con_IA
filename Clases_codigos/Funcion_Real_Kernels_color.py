#
from skimage.exposure import rescale_intensity
import numpy as np
import cv2


def convolution(image, kernel):
    (iH, iW) = image.shape[:2]  # obtiene las dimensiones de la imagen de entrada   
    (kH, kW) = kernel.shape[:2]  # obtiene las dimensiones del kernel

    pad = (kH - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

small_blur = np.array([ [0.04, 0.04, 0.04],
                        [0.04, 0.04, 0.04],
                        [0.04, 0.04, 0.04]])
large_blur = np.array([ [0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                        [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]])
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
sobelX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobelY = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
identity = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

kernelbank = (('small_blur', small_blur),
              ('large_blur', large_blur),
              ('sharpen', sharpen),
              ('laplacian', laplacian),
              ('sobelX', sobelX),
              ('sobelY', sobelY),
              ('identity', identity))


image = cv2.imread(r'Practica1\Reto Colores\pico-y-placa-2022.jpg', 0)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
for (kernelName, kernel) in kernelbank:
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolution(image, kernel)
    cv2.imshow("image", convolveOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()