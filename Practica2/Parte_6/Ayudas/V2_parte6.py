import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas y carpetas
base_path = r'Practica2\Parte_6\Costas,Forest and Higway'
folders = ['coast', 'forest', 'highway']

# Inicialización de variables
files_names = []
files_labels = []

# Obtener la lista de archivos en cada carpeta
for folder in folders:
    dirname = os.path.join(base_path, folder)
    files = [f for f in os.listdir(dirname) if f.endswith('.jpg')]
    print(f'Total de archivos en carpeta {folder}: {len(files)}')
    for file in files:
        files_names.append(os.path.join(dirname, file))
        files_labels.append(folder)

cant = len(files_names)
print('Total de archivos leídos: ', cant)

# Inicialización de listas para almacenar características y etiquetas
features = []
labels = []

# Asignar etiquetas basadas en el nombre del archivo
for i in range(cant):
    img = cv.imread(files_names[i])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Calcular medias y varianzas por canal
    avgR = img[:, :, 0].mean()
    avgG = img[:, :, 1].mean()
    avgB = img[:, :, 2].mean()
    varR = img[:, :, 0].std()
    varG = img[:, :, 1].std()
    varB = img[:, :, 2].std()

    # Concatenar las medias y varianzas en una sola lista
    features_img = [avgR, avgG, avgB, varR, varG, varB]
    features.append(features_img)

    # Asignar etiqueta basada en el nombre del archivo
    if 'coast' in files_names[i]:
        labels.append(0)  # coast -> 0
    elif 'forest' in files_names[i]:
        labels.append(1)  # forest -> 1
    elif 'highway' in files_names[i]:
        labels.append(2)  # highway -> 2

# Convertir las características y etiquetas en arrays numpy
features = np.array(features)
print(features.shape)
labels = np.array(labels) # Asegurarse de que las etiquetas sean una columna
print(labels.shape)


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Crear el modelo Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
# Entrenar el modelo
gb.fit(X_train, y_train)

# Realizar predicciones
y_pred_gb = gb.predict(X_test)

# Evaluar el modelo
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
recall_gb = recall_score(y_test, y_pred_gb, average='weighted')
precision_gb = precision_score(y_test, y_pred_gb, average='weighted')

# Imprimir las métricas de evaluación
print("Exactitud (Accuracy) - Gradient Boosting:", accuracy_gb)
print("F1 Score - Gradient Boosting:", f1_gb)
print("Recall - Gradient Boosting:", recall_gb)
print("Precisión - Gradient Boosting:", precision_gb)

# Calcular la matriz de confusión
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Mostrar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Gradient Boosting')
plt.show()

results = np.array([y_test, y_pred_gb])
print('Results: ', results)

image_test = cv.imread(r'C:\Users\Daniel Calderon\Desktop\2024-2 Poli\Vision Con IA\Practica2\Parte_6\Costas,Forest and Higway\highway\highway_a866042.jpg')
print(image_test.shape)
B, G, R = cv.split(image_test)
features = []
image_features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
features.append(image_features)
image_test_pred = gb.predict(features)
print('Pred: ', image_test_pred)






























































































# # División del conjunto de datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test, files_train, files_test, labels_train, labels_test = train_test_split(
#     features, labels, files_names, labels, test_size=0.30, random_state=1000, stratify=labels
# )




"""
# Crear el modelo Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
gb.fit(X_train, y_train.ravel())

# Realizar predicciones
y_pred_gb = gb.predict(X_test)

# Evaluar el modelo
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
recall_gb = recall_score(y_test, y_pred_gb, average='weighted')
precision_gb = precision_score(y_test, y_pred_gb, average='weighted')

# Imprimir las métricas de evaluación
print("Exactitud (Accuracy) - Gradient Boosting:", accuracy_gb)
print("F1 Score - Gradient Boosting:", f1_gb)
print("Recall - Gradient Boosting:", recall_gb)
print("Precisión - Gradient Boosting:", precision_gb)

# Calcular la matriz de confusión
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Mostrar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Gradient Boosting')
plt.show()

# Comparar imagen real vs imagen predicha
def plot_image_comparison(images_real_paths, pred_labels):
    num_images = len(images_real_paths)
    num_rows = num_images // 5
    if num_images % 5 != 0:
        num_rows += 1
    
    plt.figure(figsize=(20, num_rows * 5))
    for i in range(num_images):
        img_real = cv.imread(images_real_paths[i])
        img_real = cv.cvtColor(img_real, cv.COLOR_BGR2RGB)
        
        plt.subplot(num_rows, 5, i + 1)
        plt.imshow(img_real)
        plt.title(f'Pred: {folders[pred_labels[i]]}')
        plt.axis('off')
    
    plt.show()

# Obtener las imágenes incorrectamente clasificadas
incorrect_images = [(files_test[i], y_pred_gb[i]) for i in range(len(X_test)) if y_pred_gb[i] != y_test[i][0]]

# Limitar a 20 imágenes incorrectas
incorrect_images = incorrect_images[:20]

# Extraer rutas y predicciones
images_real_paths, pred_labels = zip(*incorrect_images)

# Mostrar comparación de imágenes
plot_image_comparison(images_real_paths, pred_labels)
"""