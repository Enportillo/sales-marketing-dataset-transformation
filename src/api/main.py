"""Servicio API REST para prediccion y targets de campana."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from dashboard.data_loader import get_campaign_scores, get_classification_results, get_df_encoded
from src.translator_simple import get_cache_stats

app = FastAPI(
    title="Sales & Marketing ML API",
    version="1.0.0",
    description="API REST para scoring de conversion y seleccion de targets de campana.",
)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, description="Registros a evaluar")
    threshold: float = Field(0.34, ge=0.0, le=1.0, description="Umbral de decision para target")


class PredictResponseItem(BaseModel):
    index: int
    conversion_score: float
    is_target: bool


class PredictResponse(BaseModel):
    threshold: float
    count: int
    results: List[PredictResponseItem]


def _feature_defaults() -> tuple[list[str], pd.Series]:
    df = get_df_encoded()
    if df.empty or "subscription_type" not in df.columns:
        raise RuntimeError("Dataset codificado no disponible para inferencia")

    feature_frame = df.drop(columns=["subscription_type", "client_id"], errors="ignore")
    feature_frame = feature_frame.select_dtypes(include=[np.number]).copy()
    feature_cols = list(feature_frame.columns)
    defaults = df[feature_cols].median(numeric_only=True)
    defaults = defaults.reindex(feature_cols)
    defaults = defaults.fillna(0.0)
    return feature_cols, defaults


def _prepare_features(records: List[Dict[str, Any]]) -> pd.DataFrame:
    feature_cols, defaults = _feature_defaults()
    frame = pd.DataFrame(records)

    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan

    frame = frame[feature_cols].copy()

    # Modelo usa features codificadas; coerce numerico para robustez de payload.
    for col in feature_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame[col].fillna(float(defaults[col]))

    return frame


@app.get("/health")
def health() -> Dict[str, Any]:
    cls = get_classification_results()
    has_model = cls.get("pipe_rf") is not None
    return {
        "status": "ok",
        "model_ready": bool(has_model),
        "auc_rf": float(cls.get("auc_rf", 0.0)) if has_model else None,
        "translation_cache": get_cache_stats(),
    }


@app.post("/predict/conversion", response_model=PredictResponse)
def predict_conversion(payload: PredictRequest) -> PredictResponse:
    cls = get_classification_results()
    model = cls.get("pipe_rf")
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    X = _prepare_features(payload.records)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error de inferencia: {exc}") from exc

    results = [
        PredictResponseItem(
            index=i,
            conversion_score=float(score),
            is_target=bool(score >= payload.threshold),
        )
        for i, score in enumerate(probs)
    ]

    return PredictResponse(threshold=payload.threshold, count=len(results), results=results)


@app.get("/campaign/targets")
def campaign_targets(
    threshold: float = Query(0.34, ge=0.0, le=1.0),
    top_n: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
    df = get_campaign_scores()
    if df.empty:
        raise HTTPException(status_code=503, detail="Scoring de campana no disponible")

    scored = df.copy()
    scored = scored[scored["conversion_score"] >= threshold]
    scored = scored.sort_values("conversion_score", ascending=False).head(top_n)

    cols = [
        "client_id",
        "conversion_score",
        "score_percentil",
        "subscription_type",
        "acquisition_channel",
        "country",
        "total_spent",
    ]
    cols = [c for c in cols if c in scored.columns]

    return {
        "threshold": threshold,
        "top_n": top_n,
        "count": int(len(scored)),
        "items": scored[cols].to_dict("records"),
    }
