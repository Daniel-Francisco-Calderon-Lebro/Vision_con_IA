from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt  # Importamos matplotlib para mostrar las imágenes

# Carga de datos
base_path = r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica5_(ParcialFinal)\Base_datos'
folder = '\Grietas_NoGrietas'
sub_folders = ['\Con_Grietas', '\Sin_Grietas']

data = []
labels = []
for subfolder in sub_folders:
    files = [f for f in os.listdir(base_path + folder + subfolder) if f.endswith('.jpg')]
    for file in files:
        img = cv2.imread(base_path + folder + subfolder + '/' + file, cv2.IMREAD_COLOR)
        img = img.astype('float32') / 255.0
        data.append(img)
        labels.append(subfolder)

# Procesamiento de etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels).flatten()

(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.1)

# Definición del modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(227, 227, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))

# Salida binaria
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()

# Compilación y entrenamiento
opt = RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)

# Predicciones y reporte
predictions = (model.predict(testX) > 0.5).astype("int32")
print(classification_report(testY, predictions, target_names=['Sin Grieta', 'Con Grieta']))

# Ejemplo de imagen con predicción
rand_pos = random.randint(0, len(testX))
rand_img = testX[rand_pos]
plt.imshow(rand_img)  # Usamos matplotlib para mostrar la imagen
plt.axis('off')  # Para que no se muestren los ejes
plt.show()

print('Ground truth class: ', lb.classes_[np.argmax(testY[rand_pos])])
print('Predicted class: ', lb.classes_[np.argmax(predictions[rand_pos])])

# Bucle para mostrar 5 imágenes aleatorias con sus predicciones
for _ in range(5):
    rand_pos = random.randint(0, len(testX) - 1)
    rand_img = testX[rand_pos]
    
    # Mostrar la imagen con matplotlib
    plt.imshow(rand_img)  
    plt.axis('off')  # Ocultar los ejes
    plt.show()
    
    # Mostrar la clase real y predicha
    print('Ground truth class: ', lb.classes_[testY[rand_pos]])
    print('Predicted class: ', lb.classes_[predictions[rand_pos][0]])

    # Verificar si la predicción fue incorrecta
    if testY[rand_pos] != predictions[rand_pos][0]:
        print('Wrong prediction#####################################################')
