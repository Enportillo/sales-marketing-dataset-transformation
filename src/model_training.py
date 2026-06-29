from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def aplicar_pca(X, n_components=2, ramdom_state=42):

    pca = PCA(n_components=n_components, random_state=ramdom_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def entrenar_Kmeans(X, n_clusters, random_state=42, n_init=10):
    
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    model.fit(X)
    return model

def construir_pipeline_clasificacion(tipo_de_modelo='rf', random_state=42):

    scaler = StandardScaler()
    if tipo_de_modelo == 'rf':
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif tipo_de_modelo == 'lr':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    else:
        raise ValueError("Modelo no soportado. Usa 'rf': (Random Forest) o 'lr' (Logistic regression)" )
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])

    return pipeline