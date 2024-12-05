import cv2
import numpy as np
basepath = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica3_(Parcial)\Parte_1' #Path of folder where the image is located

#List of images
img1path = '\s1a-iw-grd-vv-20240826t111135-20240826t111152-055383-06c133-001_scaled.tiff' #Image not aligined (registered) because it was used as the reference to register the rest
img2path = '\s1a-iw-grd-vv-20240907t111135-20240907t111152-055558-06c7d4-001_registered.tiff'
img3path = '\s1a-iw-grd-vv-20240919t111136-20240919t111153-055733-06cebc-001_registered.tiff'
# img4path = 's1a-iw-grd-vv-20220929t230848-20220929t230913-045226-0567e1-001_scaled_registered.tiff'
# img5path = 's1a-iw-grd-vv-20221011t230847-20221011t230912-045401-056dca-001_scaled_registered.tiff'
# img6path = 's1a-iw-grd-vv-20221023t230848-20221023t230913-045576-0572f6-001_scaled_registered.tiff'
# img7path = 's1a-iw-grd-vv-20221104t230847-20221104t230912-045751-0578dc-001_scaled_registered.tiff'
# img8path = 's1a-iw-grd-vv-20221116t230847-20221116t230912-045926-057ecb-001_scaled_registered.tiff'
# img9path = 's1a-iw-grd-vv-20221128t230847-20221128t230912-046101-0584b4-001_scaled_registered.tiff'
# img10path = 's1a-iw-grd-vv-20221222t230845-20221222t230910-046451-0590a5-001_scaled_registered.tiff'

pathlist = [basepath + img1path,
               basepath + img2path,
               basepath + img3path,
            #    basepath + img4path,
            #    basepath + img5path,
            #    basepath + img6path,
            #    basepath + img7path,
            #    basepath + img8path,
            #    basepath + img9path,
            #    basepath + img10path
               ]

#print(rutaimglist[0])
img = cv2.imread(pathlist[0])  #Image load
height, width, depth = img.shape
imgacum = np.zeros((height, width), np.single)
imgscalar = 10.0 * np.ones((height, width), np.single)

for imgpath in pathlist:
    print(imgpath)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)  # Image load
    img = img.astype(np.single)
    imgacum = cv2.add(imgacum, img)
avgGT = cv2.divide(imgacum, imgscalar)
avgGT = avgGT.astype(np.uint8)
cv2.imwrite(basepath + 'AverageGT.tiff', avgGT)