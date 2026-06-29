import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluar_clusters(X, labels):

    score = silhouette_score(X, labels)
    print(f'Silouette Score: {score:.4f}')

    return score

def plot_clusters_2d(X_pca, labels, title='Visualización de Clústeres'):


    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis")
    plt.title(title)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Clúster')
    plt.show()



def evaluar_modelo_de_clasificacion(pipeline, X_train, y_train, X_test, y_test, model_name="Modelo"):

    print(f"\n{'='*50}")
    print(f"EVALUACIÓN DEL MODELO: {model_name.upper()}")
    print(f"{'='*50}")
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n1. Validación Cruzada (Accuracy de 5 Folds):")
    print(f"   Scores: {cv_scores}")
    print(f"   Media: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n2. Reporte de Clasificación en Set de Prueba:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.show()
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n3. ROC-AUC Score: {auc:.4f}")
    print("-" * 50)