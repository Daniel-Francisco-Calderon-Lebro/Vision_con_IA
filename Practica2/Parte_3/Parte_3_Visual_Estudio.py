import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos del conjunto de datos Iris
data = load_iris()
X = data.data
y_true = data.target
n_classes = len(np.unique(y_true))

# Binarizar las etiquetas para el cálculo de la curva ROC y AUC
y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])

# Entrenar K-Means (con 3 clústeres para las 3 clases)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Asignar las etiquetas a los clústeres resultantes
labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask])[0]

# Calcular Accuracy, F1-Score, Recall y Precisión
accuracy = accuracy_score(y_true, labels)
f1 = f1_score(y_true, labels, average='macro')
recall = recall_score(y_true, labels, average='macro')
precision = precision_score(y_true, labels, average='macro')

# Calcular la especificidad para cada clase: TN / (TN + FP)
cm = confusion_matrix(y_true, labels)
specificity = []
for i in range(3):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))
specificity = np.mean(specificity)

# Imprimir las métricas para K-Means
print(f"Métricas para K-Means:")
print(f"Exactitud (Accuracy): {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Recall (Sensibilidad): {recall:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"Especificidad: {specificity:.2f}")

# Mostrar la matriz de confusión con Seaborn para K-Means
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Etiquetas Predichas")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión para K-Means")
plt.show()

# Entrenar el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(3, 3,), max_iter=1000, random_state=42)
mlp.fit(X, y_true)

# Evaluar el modelo MLP
y_pred_mlp = mlp.predict(X)

# Calcular Accuracy, F1-Score, Recall y Precisión para MLP
accuracy_mlp = accuracy_score(y_true, y_pred_mlp)
f1_mlp = f1_score(y_true, y_pred_mlp, average='macro')
recall_mlp = recall_score(y_true, y_pred_mlp, average='macro')
precision_mlp = precision_score(y_true, y_pred_mlp, average='macro')

# Calcular la especificidad para cada clase: TN / (TN + FP)
cm_mlp = confusion_matrix(y_true, y_pred_mlp)
specificity = []
for i in range(3):
    tn = cm_mlp.sum() - (cm_mlp[i, :].sum() + cm_mlp[:, i].sum() - cm_mlp[i, i])
    fp = cm_mlp[:, i].sum() - cm_mlp[i, i]
    specificity.append(tn / (tn + fp))
specificity_mlp = np.mean(specificity)

# Imprimir las métricas para MLP
print(f"\nMétricas para MLP:")
print(f"Exactitud (Accuracy): {accuracy_mlp:.2f}")
print(f"F1-Score: {f1_mlp:.2f}")
print(f"Recall (Sensibilidad): {recall_mlp:.2f}")
print(f"Precisión: {precision_mlp:.2f}")
print(f"Especificidad: {specificity_mlp:.2f}")

# Mostrar la matriz de confusión con Seaborn para MLP
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Etiquetas Predichas")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión para MLP")
plt.show()

# ======= Generar Curva ROC y calcular AUC para MLP =======
y_score_mlp = mlp.predict_proba(X)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score_mlp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Mostrar las curvas ROC para cada clase
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para MLP')
plt.legend(loc="lower right")
plt.show()

