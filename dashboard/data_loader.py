"""
Módulo de carga de datos y entrenamiento de modelos.
Se encarga de cargar los datasets y entrenar los modelos ML,
almacenando los resultados en caché para evitar re-cómputos.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
)

# ── Rutas base ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_RAW = BASE_DIR / "data" / "raw"

# ── Columnas relevantes ───────────────────────────────────────────────────────
COLS_COMPORTAMIENTO = [
    "total_spent",
    "avg_order_value",
    "last_3_month_purchase_freq",
    "total_visits",
    "pages_per_session",
    "support_tickets",
]

COLS_NUMERICAS = [
    "age",
    "total_spent",
    "avg_order_value",
    "lifetime_value",
    "last_3_month_purchase_freq",
    "marketing_spend_per_user",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "email_open_rate",
    "email_click_rate",
    "support_tickets",
    "delivery_delay_days",
    "satisfaction_score",
    "nps_score",
]

COLS_CATEGORICAS = [
    "gender",
    "country",
    "acquisition_channel",
    "subscription_type",
    "payment_method",
]

# ── Caché global ──────────────────────────────────────────────────────────────
_cache: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────
def get_df_encoded() -> pd.DataFrame:
    if "encoded" not in _cache:
        _cache["encoded"] = pd.read_csv(
            DATA_PROCESSED / "Sales_Marketing_Clean_(Codificado).csv"
        )
    return _cache["encoded"]


def get_df_clean() -> pd.DataFrame:
    if "clean" not in _cache:
        try:
            _cache["clean"] = pd.read_excel(
                DATA_PROCESSED / "Sales_Marketing_Clean.xlsx"
            )
        except Exception:
            _cache["clean"] = pd.DataFrame()
    return _cache["clean"]


def get_df_raw() -> pd.DataFrame:
    if "raw" not in _cache:
        try:
            _cache["raw"] = pd.read_excel(
                DATA_RAW / "Dirty_Sales_Marketing_Dataset.xlsx"
            )
        except Exception:
            _cache["raw"] = pd.DataFrame()
    return _cache["raw"]


# ──────────────────────────────────────────────────────────────────────────────
# Clustering (KMeans)
# ──────────────────────────────────────────────────────────────────────────────
def get_cluster_results() -> dict:
    if "cluster" not in _cache:
        df = get_df_encoded()

        # Filtrar columnas disponibles
        cols = [c for c in COLS_COMPORTAMIENTO if c in df.columns]
        X_cluster = df[cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Codo y Silhouette
        k_range = list(range(2, 7))
        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km.labels_))

        # Modelo final K=3
        km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km_final.fit_predict(X_scaled)

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Centroides
        df_cl = X_cluster.copy()
        df_cl["cluster"] = labels
        centroids = df_cl.groupby("cluster")[cols].mean().round(2)
        counts = df_cl["cluster"].value_counts().rename("n_clientes")
        centroids = centroids.join(counts)

        _cache["cluster"] = {
            "k_range": k_range,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "labels": labels,
            "X_pca": X_pca,
            "pca_var": pca.explained_variance_ratio_,
            "centroids": centroids,
            "cols": cols,
        }
    return _cache["cluster"]


# ──────────────────────────────────────────────────────────────────────────────
# Modelos Supervisados (Clasificación + Regresión)
# ──────────────────────────────────────────────────────────────────────────────
def get_classification_results() -> dict:
    if "classification" not in _cache:
        df = get_df_encoded()
        nombre_target = "subscription_type"

        y = df[nombre_target]
        X = df.drop(columns=[nombre_target], errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        def train_pipe(model):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            pipe.fit(X_train, y_train)
            return pipe

        # Random Forest
        pipe_rf = train_pipe(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        y_pred_rf = pipe_rf.predict(X_test)
        probs_rf = pipe_rf.predict_proba(X_test)[:, 1]

        # Regresión Logística
        pipe_lr = train_pipe(
            LogisticRegression(random_state=42, max_iter=1000)
        )
        y_pred_lr = pipe_lr.predict(X_test)
        probs_lr = pipe_lr.predict_proba(X_test)[:, 1]

        # ── Regresión (predecir total_spent) ─────────────────────────────────
        target_reg = "total_spent"
        y_reg = df[target_reg]
        X_reg = df.drop(columns=[target_reg], errors="ignore")
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        lr_reg = LinearRegression().fit(Xr_tr, yr_tr)
        rf_reg = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ).fit(Xr_tr, yr_tr)

        def reg_metrics(model, Xte, yte):
            yp = model.predict(Xte)
            return {
                "RMSE": float(mean_squared_error(yte, yp) ** 0.5),
                "R2": float(r2_score(yte, yp)),
            }

        # ── Upselling ─────────────────────────────────────────────────────────
        clientes_basicos_X = X_test[y_test == 0]
        probs_upsell = pipe_rf.predict_proba(clientes_basicos_X)[:, 1]

        _cache["classification"] = {
            "cm_rf": confusion_matrix(y_test, y_pred_rf),
            "cm_lr": confusion_matrix(y_test, y_pred_lr),
            "auc_rf": float(roc_auc_score(y_test, probs_rf)),
            "auc_lr": float(roc_auc_score(y_test, probs_lr)),
            "report_rf": classification_report(y_test, y_pred_rf, output_dict=True),
            "report_lr": classification_report(y_test, y_pred_lr, output_dict=True),
            "probs_rf": probs_rf,
            "probs_lr": probs_lr,
            "y_test": y_test,
            "reg_results": {
                "Regresión Lineal": reg_metrics(lr_reg, Xr_te, yr_te),
                "Random Forest Regressor": reg_metrics(rf_reg, Xr_te, yr_te),
            },
            "probs_upsell": probs_upsell,
            "n_basicos": len(clientes_basicos_X),
            "pipe_rf": pipe_rf,
        }
    return _cache["classification"]


# ──────────────────────────────────────────────────────────────────────────────
# Optimización de Hiperparámetros (GridSearchCV simplificado)
# ──────────────────────────────────────────────────────────────────────────────
def get_optimization_results() -> dict:
    if "optimization" not in _cache:
        df = get_df_encoded()
        nombre_target = "subscription_type"
        y = df[nombre_target]
        X = df.drop(columns=[nombre_target], errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cols_v2 = [c for c in COLS_COMPORTAMIENTO if c in df.columns]

        # Grilla reducida para velocidad razonable
        param_grid = {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5],
        }
        base_pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))]
        )

        def run_gs(Xtr, Xte, ytr, yte, label):
            gs = GridSearchCV(
                base_pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=1
            )
            gs.fit(Xtr, ytr)
            best = gs.best_estimator_
            probs = best.predict_proba(Xte)[:, 1]
            ypred = best.predict(Xte)
            return {
                "label": label,
                "auc": float(roc_auc_score(yte, probs)),
                "best_params": gs.best_params_,
                "cm": confusion_matrix(yte, ypred),
                "report": classification_report(yte, ypred, output_dict=True),
            }

        v1 = run_gs(X_train, X_test, y_train, y_test, "v1 - Todas las features")
        v2 = run_gs(
            X_train[cols_v2], X_test[cols_v2], y_train, y_test,
            "v2 - Solo comportamiento"
        )

        # v3: invertir target (Annual como positivo)
        y_inv = (y == 1).astype(int)
        y_inv_train, y_inv_test = (
            y_inv.iloc[y_train.index],
            y_inv.iloc[y_test.index],
        )
        v3 = run_gs(
            X_train[cols_v2], X_test[cols_v2],
            y_inv.loc[y_train.index], y_inv.loc[y_test.index],
            "v3 - Annual como positivo"
        )

        _cache["optimization"] = {
            "versions": [v1, v2, v3],
            "y_test": y_test,
            "cols_v2": cols_v2,
        }
    return _cache["optimization"]
