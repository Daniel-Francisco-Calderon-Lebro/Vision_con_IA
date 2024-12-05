import cv2
import numpy as np
import os

basepath = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica3_(Parcial)\Parte_2' #Path of folder where the image is locatedrutaimg1 = 'prom10.tiff'
noisypath = '\s1a-iw-grd-vv-20240826t111135-20240826t111152-055383-06c133-001_scaled.png' #Reference image of one of the registered ones. In this case reference was used.
GTpath = '\imagen_filtrada.png' #Image obtained by averaging several images with Generate_GTruth.py

#Load images
imgnoisy = cv2.imread(basepath + noisypath)
imgGT = cv2.imread(basepath + GTpath)

#Parameters
height, width = imgnoisy.shape[:2]
size = 512
step = 512
counter = 0

#Create directories
if not os.path.isdir(basepath + 'Noisy'): #Check if directory "Noisy" exists.
    os.mkdir(basepath + 'Noisy') #If does not exist then create the folder
if not os.path.isdir(basepath + 'Gtruth'): #Check if directory "Gtruth" exists. If not then create
    os.mkdir(basepath + 'Gtruth') #If does not exist then create the folder

for i in range(0, height-size, size):
    for j in range(0, width-size, size):
        #Noisy
        name_noisy = basepath +'Noisy/' + str(i) + '_' + str(j) + '.png'
        imgcrop_noisy = imgnoisy[i:i+size, j:j+size]
        cv2.imwrite(name_noisy, imgcrop_noisy)
        #Ground Truth
        name_GT = basepath + 'Gtruth/' + str(i) + '_' + str(j) + '.png'
        imgcrop_GT = imgGT[i:i + size, j:j + size]
        cv2.imwrite(name_GT, imgcrop_GT)
        counter += 1
        print(counter)
