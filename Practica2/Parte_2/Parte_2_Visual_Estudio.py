from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# Cargar la base de datos Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
df.head()
sns.pairplot(df, hue='target')
plt.show()

# Separar datos en train y test
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1000)

# Función para calcular especificidad
def calcular_especificidad(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    especificidades = []
    
    for i in range(len(iris.target_names)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        if (tn + fp) != 0:
            especificidad = tn / (tn + fp)
        else:
            especificidad = np.nan
        especificidades.append(especificidad)
    
    return np.nanmean(especificidades)

# Función para evaluar el modelo
def evaluar_modelo(nombre, y_test, y_pred, modelo, X_test):
    print(f'\nResultados para {nombre}:')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    especificidad = calcular_especificidad(y_test, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall (Sensibilidad): {recall}')
    print(f'F1-score: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'Especificidad: {especificidad}')

    # Curva ROC y AUC
    y_prob = modelo.predict_proba(X_test)
    fpr = {}
    tpr = {}
    auc = {}

    for i in range(len(iris.target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
        auc[i] = roc_auc_score(y_test == i, y_prob[:, i])
    
        plt.plot(fpr[i], tpr[i], label=f'Clase {iris.target_names[i]} (AUC = {auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {nombre}')
    plt.legend(loc="lower right")
    plt.show()

# 1. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
evaluar_modelo("Naive Bayes", y_test, y_pred_nb, nb, X_test)

# 2. Máquinas de Vectores de Soporte
svm = SVC(probability=True, kernel='linear')  # kernel lineal para simplificar
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
evaluar_modelo("Máquinas de Vectores de Soporte", y_test, y_pred_svm, svm, X_test)

# Graficar las matrices de confusión
def graficar_matriz_confusion(y_test, y_pred, modelo_nombre):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Matriz de Confusión - {modelo_nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# Graficar las matrices de confusión de cada modelo
graficar_matriz_confusion(y_test, y_pred_nb, "Naive Bayes")
graficar_matriz_confusion(y_test, y_pred_svm, "Máquinas de Vectores de Soporte")
