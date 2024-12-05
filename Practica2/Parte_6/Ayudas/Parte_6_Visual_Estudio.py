# -*- coding: utf-8 -*-
"""Parte_6_Visual_Estudio.ipynb"""

import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import expon
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

# for i in range(len(files_names)):
#     cv.imshow(files_labels[i], cv.imread(files_names[i]))
#     print(files_labels[i])
#     cv.waitKey(0)
#     cv.destroyAllWindows()

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
labels = np.array(labels).reshape(-1, 1)  # Asegurarse de que las etiquetas sean una columna

# Combinar características y etiquetas
data = np.hstack((features, labels))
# Crear los nombres de las columnas
columns = ['avgR', 'avgG', 'avgB', 'varR', 'varG', 'varB', 'label']

# Guardar los datos en un archivo .txt usando np.savetxt
filename = 'Base_de_Datos_Tar.txt'

# Guardar los datos en un archivo de texto con diferentes formatos para características y etiquetas
header = ','.join(columns)
formats = '%.6e %.6e %.6e %.6e %.6e %.6e %d'  # Usar %d para la etiqueta
np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt=formats.split())

print(f"Archivo '{filename}' guardado exitosamente.")

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=1000)

########################################################################################################################################

# 1. Redes Neuronales
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train.ravel())  # Aplanar y_train para evitar advertencias
y_pred_mlp = mlp.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred_mlp)
f1 = f1_score(y_test, y_pred_mlp, average='weighted')
recall = recall_score(y_test, y_pred_mlp, average='weighted')
precision = precision_score(y_test, y_pred_mlp, average='weighted')

# Mostrar los resultados
print("Exactitud (Accuracy): ", accuracy)
print("F1 Score: ", f1)
print("Recall: ", recall)
print("Precisión: ", precision)

# Mostrar matriz de confusión
cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - MLP Classifier')
plt.show()


#########################################################################################################################################

# 2. SVM con kernel RBF
# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo SVM
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# Definir los parámetros de búsqueda con distribución exponencial para `C` y `gamma`
param_distributions = {
    'C': expon(scale=100),
    'gamma': expon(scale=1)
}

# Aplicar RandomizedSearchCV
random_search = RandomizedSearchCV(svm, param_distributions, n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train_scaled, y_train.ravel())  # Aplanar y_train para evitar advertencias

# Mejor modelo encontrado
best_svm = random_search.best_estimator_

# Realizar predicciones
y_pred = best_svm.predict(X_test_scaled)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Imprimir las métricas de evaluación
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)
print(f"Exactitud (Accuracy): {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precisión: {precision}")

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Imprimir la matriz de confusión
print("Matriz de confusión:")
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Coast', 'Forest', 'Highway'], yticklabels=['Coast', 'Forest', 'Highway'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión SVM con kernel RBF")
plt.show()

#########################################################################################################################################


from sklearn.ensemble import RandomForestClassifier

# Crear el modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf.fit(X_train, y_train.ravel())

# Realizar predicciones
y_pred_rf = rf.predict(X_test)

# Evaluar el modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')

# Imprimir las métricas de evaluación
print("Exactitud (Accuracy) - Random Forest:", accuracy_rf)
print("F1 Score - Random Forest:", f1_rf)
print("Recall - Random Forest:", recall_rf)
print("Precisión - Random Forest:", precision_rf)

# Calcular la matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Mostrar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Random Forest')
plt.show()


########################################################################################################################################



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



#######################################################################################################################################

# Crear el modelo K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo
knn.fit(X_train, y_train.ravel())

# Realizar predicciones
y_pred_knn = knn.predict(X_test)

# Evaluar el modelo
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')

# Imprimir las métricas de evaluación
print("Exactitud (Accuracy) - KNN:", accuracy_knn)
print("F1 Score - KNN:", f1_knn)
print("Recall - KNN:", recall_knn)
print("Precisión - KNN:", precision_knn)

# Calcular la matriz de confusión
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Mostrar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - K-Nearest Neighbors')
plt.show()


#########################################################################################################################################


# Crear el modelo de Regresión Logística
lr = LogisticRegression(max_iter=1000, random_state=42)

# Entrenar el modelo
lr.fit(X_train, y_train.ravel())

# Realizar predicciones
y_pred_lr = lr.predict(X_test)

# Evaluar el modelo
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')

# Imprimir las métricas de evaluación
print("Exactitud (Accuracy) - Regresión Logística:", accuracy_lr)
print("F1 Score - Regresión Logística:", f1_lr)
print("Recall - Regresión Logística:", recall_lr)
print("Precisión - Regresión Logística:", precision_lr)

# Calcular la matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Mostrar matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Costa', 'Bosque', 'Autopista'], yticklabels=['Costa', 'Bosque', 'Autopista'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()
