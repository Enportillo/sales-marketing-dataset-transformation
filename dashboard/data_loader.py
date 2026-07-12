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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Configuracion de rendimiento para dashboard interactivo
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
MAX_ROWS_CLASSIFICATION = 12000
MAX_ROWS_OPTIMIZATION = 6000


def _sample_if_large(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42):
    """Reduce el dataset de entrenamiento cuando es grande para mejorar latencia UI."""
    if len(X) <= max_rows:
        return X, y
    sampled = pd.concat([X, y.rename("__target__")], axis=1).sample(
        n=max_rows,
        random_state=random_state,
    )
    Xs = sampled.drop(columns=["__target__"])
    ys = sampled["__target__"]
    return Xs, ys


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────
def get_df_encoded() -> pd.DataFrame:
    if "encoded" not in _cache:
        df = pd.read_csv(
            DATA_PROCESSED / "Sales_Marketing_Clean_(Codificado).csv"
        )
        if "client_id" not in df.columns:
            # ID tecnico estable para trazabilidad de usuarios en dashboard.
            df.insert(0, "client_id", [f"C-{i + 1:05d}" for i in range(len(df))])
        _cache["encoded"] = df
    return _cache["encoded"]


def get_df_clean() -> pd.DataFrame:
    if "clean" not in _cache:
        try:
            df = pd.read_excel(
                DATA_PROCESSED / "Sales_Marketing_Clean.xlsx"
            )
            if "client_id" not in df.columns:
                # Mantiene correspondencia fila a fila con el dataset codificado.
                df.insert(0, "client_id", [f"C-{i + 1:05d}" for i in range(len(df))])
            _cache["clean"] = df
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
        X = df.drop(columns=[nombre_target, "client_id"], errors="ignore")

        X, y = _sample_if_large(X, y, max_rows=MAX_ROWS_CLASSIFICATION)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Random Forest
        pipe_rf = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=N_JOBS,
        )
        pipe_rf.fit(X_train, y_train)
        y_pred_rf = pipe_rf.predict(X_test)
        probs_rf = pipe_rf.predict_proba(X_test)[:, 1]

        # Regresión Logística
        pipe_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=600)),
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        probs_lr = pipe_lr.predict_proba(X_test)[:, 1]

        # ── Regresión (predecir total_spent) ─────────────────────────────────
        target_reg = "total_spent"
        y_reg = df[target_reg]
        X_reg = df.drop(columns=[target_reg, "client_id"], errors="ignore")
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        lr_reg = LinearRegression().fit(Xr_tr, yr_tr)
        rf_reg = RandomForestRegressor(
            n_estimators=120, random_state=42, n_jobs=N_JOBS
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

        # Scoring global para vista overview a nivel usuario.
        X_all = df.drop(columns=[nombre_target, "client_id"], errors="ignore")
        probs_all = pipe_rf.predict_proba(X_all)[:, 1]

        campaign_cols = [
            "client_id",
            "subscription_type",
            "total_spent",
            "avg_order_value",
            "last_3_month_purchase_freq",
            "total_visits",
            "pages_per_session",
            "support_tickets",
            "gender",
            "country",
            "acquisition_channel",
            "payment_method",
        ]
        campaign_cols = [c for c in campaign_cols if c in df.columns]
        df_campaign = df[campaign_cols].copy()

        # Reemplaza etiquetas codificadas por valores legibles del dataset limpio.
        df_clean = get_df_clean()
        if not df_clean.empty and "client_id" in df_clean.columns:
            readable_cols = [
                "client_id",
                "gender",
                "country",
                "acquisition_channel",
                "subscription_type",
                "payment_method",
            ]
            readable_cols = [c for c in readable_cols if c in df_clean.columns]

            if len(readable_cols) > 1:
                df_campaign = df_campaign.merge(
                    df_clean[readable_cols],
                    on="client_id",
                    how="left",
                    suffixes=("", "_readable"),
                )

                for col in [
                    "gender",
                    "country",
                    "acquisition_channel",
                    "subscription_type",
                    "payment_method",
                ]:
                    readable_col = f"{col}_readable"
                    if readable_col not in df_campaign.columns:
                        continue
                    if col in df_campaign.columns:
                        df_campaign[col] = df_campaign[readable_col].where(
                            df_campaign[readable_col].notna(),
                            df_campaign[col],
                        )
                    else:
                        df_campaign[col] = df_campaign[readable_col]
                    df_campaign.drop(columns=[readable_col], inplace=True)

        df_campaign["conversion_score"] = probs_all
        df_campaign["score_percentil"] = (
            df_campaign["conversion_score"].rank(pct=True, method="average") * 100
        )

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
            "df_campaign_scores": df_campaign,
        }
    return _cache["classification"]


def get_campaign_scores() -> pd.DataFrame:
    """Retorna scoring por cliente para campañas de conversión."""
    res = get_classification_results()
    return res.get("df_campaign_scores", pd.DataFrame()).copy()


# ──────────────────────────────────────────────────────────────────────────────
# Optimización de Hiperparámetros (GridSearchCV simplificado)
# ──────────────────────────────────────────────────────────────────────────────
def get_optimization_results() -> dict:
    if "optimization" not in _cache:
        df = get_df_encoded()
        nombre_target = "subscription_type"
        y = df[nombre_target]
        X = df.drop(columns=[nombre_target, "client_id"], errors="ignore")

        X, y = _sample_if_large(X, y, max_rows=MAX_ROWS_OPTIMIZATION)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cols_v2 = [c for c in COLS_COMPORTAMIENTO if c in df.columns]

        # Espacio de busqueda ligero para respuesta rapida en dashboard
        param_dist = {
            "clf__n_estimators": [80, 120, 160],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 4, 8],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.6, None],
        }
        base_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=N_JOBS)),
            ]
        )

        def run_gs(Xtr, Xte, ytr, yte, label):
            gs = RandomizedSearchCV(
                base_pipe,
                param_distributions=param_dist,
                n_iter=8,
                cv=2,
                scoring="roc_auc",
                n_jobs=N_JOBS,
                random_state=42,
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
