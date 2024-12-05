import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, roc_curve, roc_auc_score, classification_report)
from sklearn.preprocessing import label_binarize

# Cargar datos del conjunto de datos Iris
data = load_iris()
X = data.data
y = data.target
class_names = data.target_names

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=42)

# 1. **Máquinas de Vectores de Soporte con Kernel RBF**
print("Evaluación de SVM con Kernel RBF")

# Definir el modelo con kernel RBF
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)

# Definir los parámetros a buscar
param_grid_svm_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

# Realizar la búsqueda en malla para encontrar los mejores hiperparámetros
grid_search_svm_rbf = GridSearchCV(svm_rbf, param_grid_svm_rbf, cv=5, scoring='accuracy')
grid_search_svm_rbf.fit(X_train, y_train)

# Mejor modelo y resultados
best_svm_rbf = grid_search_svm_rbf.best_estimator_
y_pred_svm_rbf = best_svm_rbf.predict(X_test)

# Calcular la matriz de confusión para SVM RBF
cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)

# Mostrar la matriz de confusión con Seaborn para SVM RBF
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm_rbf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión para SVM con Kernel RBF')
plt.show()

# Evaluar SVM con Kernel RBF
accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
f1_svm_rbf = f1_score(y_test, y_pred_svm_rbf, average='macro')
recall_svm_rbf = recall_score(y_test, y_pred_svm_rbf, average='macro')
precision_svm_rbf = precision_score(y_test, y_pred_svm_rbf, average='macro')

print(f"\nMétricas para SVM con Kernel RBF:")
print(f"Exactitud (Accuracy): {accuracy_svm_rbf:.2f}")
print(f"F1-Score: {f1_svm_rbf:.2f}")
print(f"Recall (Sensibilidad): {recall_svm_rbf:.2f}")
print(f"Precisión: {precision_svm_rbf:.2f}")

# Binarizar las etiquetas para la curva ROC
y_true_binarized = label_binarize(y_test, classes=[0, 1, 2])
y_pred_svm_rbf_proba = best_svm_rbf.predict_proba(X_test)

# Curva ROC y AUC para SVM con Kernel RBF
plt.figure(figsize=(12, 6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_svm_rbf_proba[:, i])
    auc = roc_auc_score(y_true_binarized[:, i], y_pred_svm_rbf_proba[:, i])
    plt.plot(fpr, tpr, label=f'SVM RBF - {class_names[i]} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para SVM con Kernel RBF')
plt.legend(loc='best')
plt.show()

# 2. **Gradient Boosting**
print("Evaluación de Gradient Boosting")

# Crear y ajustar el modelo de Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Validación cruzada K-Fold
kf = 5
scores_gb = cross_val_score(gb, X, y, cv=kf, scoring='accuracy')

# Ajustar el modelo con los datos de entrenamiento
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Calcular la matriz de confusión para Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Mostrar la matriz de confusión con Seaborn para Gradient Boosting
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión para Gradient Boosting')
plt.show()

# Resultados de validación cruzada
print(f"\nExactitud promedio del Gradient Boosting (K-Fold): {np.mean(scores_gb):.4f}")

# Evaluar Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='macro')
recall_gb = recall_score(y_test, y_pred_gb, average='macro')
precision_gb = precision_score(y_test, y_pred_gb, average='macro')

print(f"\nMétricas para Gradient Boosting:")
print(f"Exactitud (Accuracy): {accuracy_gb:.2f}")
print(f"F1-Score: {f1_gb:.2f}")
print(f"Recall (Sensibilidad): {recall_gb:.2f}")
print(f"Precisión: {precision_gb:.2f}")

# Binarizar las etiquetas para la curva ROC
y_pred_gb_proba = gb.predict_proba(X_test)

# Curva ROC y AUC para Gradient Boosting
plt.figure(figsize=(12, 6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_gb_proba[:, i])
    auc = roc_auc_score(y_true_binarized[:, i], y_pred_gb_proba[:, i])
    plt.plot(fpr, tpr, label=f'Gradient Boosting - {class_names[i]} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para Gradient Boosting')
plt.legend(loc='best')
plt.show()
