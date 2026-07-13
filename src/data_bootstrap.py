"""Generacion automatica de artefactos processed para ejecucion reproducible del dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.external_api_enrichment import get_usd_to_clp_rate

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "Dirty_Sales_Marketing_Dataset.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "Sales_Marketing_Clean.xlsx"
ENCODED_FILE = PROCESSED_DIR / "Sales_Marketing_Clean_(Codificado).csv"

CAT_COLS = [
    "gender",
    "country",
    "acquisition_channel",
    "subscription_type",
    "payment_method",
]

NUMERIC_COLS = [
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

WINSOR_COLS = [
    "age",
    "total_spent",
    "avg_order_value",
    "lifetime_value",
    "total_visits",
    "avg_session_time",
    "pages_per_session",
    "support_tickets",
    "delivery_delay_days",
]

INT_COLS = ["age", "total_visits", "support_tickets", "delivery_delay_days"]


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
    if "gender" in df.columns:
        gender_map = {
            "male": "Male",
            "m": "Male",
            "female": "Female",
            "f": "Female",
            "unknown": "Unknown",
        }
        lower_gender = df["gender"].astype(str).str.lower().str.strip()
        df["gender"] = lower_gender.map(gender_map).fillna(df["gender"].astype(str).str.title())
    return df


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:
        df["age"] = df["age"].fillna(df["age"].median())
    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    if "total_spent" in df.columns:
        if "subscription_type" in df.columns:
            grouped = df.groupby("subscription_type")["total_spent"].transform(lambda s: s.fillna(s.median()))
            df["total_spent"] = grouped
        df["total_spent"] = df["total_spent"].fillna(df["total_spent"].median())

    for col in CAT_COLS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def _winsor_iqr(df: pd.DataFrame) -> pd.DataFrame:
    for col in WINSOR_COLS:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        df[col] = df[col].clip(low, high)
    return df


def _cast_ints(df: pd.DataFrame) -> pd.DataFrame:
    for col in INT_COLS:
        if col in df.columns:
            df[col] = np.rint(df[col]).astype("int64")
    return df


def _encode(df_clean: pd.DataFrame) -> pd.DataFrame:
    encoded = df_clean.copy()
    # Codifica cualquier feature categórica para garantizar matriz numérica estable.
    for col in encoded.columns:
        if col == "client_id":
            continue
        if pd.api.types.is_object_dtype(encoded[col]) or pd.api.types.is_categorical_dtype(encoded[col]):
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))
    return encoded


def build_processed_from_raw(raw_path: Path = RAW_FILE) -> Dict[str, str]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontro dataset crudo en {raw_path}. Debes incluir el archivo raw para generar processed."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(raw_path)
    df = _normalize_text_columns(df)
    df = _to_numeric(df)
    df = _impute(df)
    df = _winsor_iqr(df)
    df = _cast_ints(df)

    # Enriquecimiento REST: tasa de cambio para normalizar montos a USD.
    fx_usd_to_clp = float(get_usd_to_clp_rate())
    df["fx_usd_to_clp"] = fx_usd_to_clp
    if "total_spent" in df.columns:
        df["total_spent_usd"] = (df["total_spent"] / fx_usd_to_clp).round(2)
    if "avg_order_value" in df.columns:
        df["avg_order_value_usd"] = (df["avg_order_value"] / fx_usd_to_clp).round(2)

    df.to_excel(CLEAN_FILE, index=False)
    _encode(df).to_csv(ENCODED_FILE, index=False)

    return {
        "clean": str(CLEAN_FILE),
        "encoded": str(ENCODED_FILE),
    }


def ensure_processed_data(force: bool = False) -> Dict[str, str]:
    """Garantiza artefactos de processed; si faltan, los genera desde raw."""
    if not force and CLEAN_FILE.exists() and ENCODED_FILE.exists():
        return {"clean": str(CLEAN_FILE), "encoded": str(ENCODED_FILE)}

    if CLEAN_FILE.exists() and (force or not ENCODED_FILE.exists()):
        df_clean = pd.read_excel(CLEAN_FILE)
        _encode(df_clean).to_csv(ENCODED_FILE, index=False)
        return {"clean": str(CLEAN_FILE), "encoded": str(ENCODED_FILE)}

    return build_processed_from_raw(RAW_FILE)


if __name__ == "__main__":
    outputs = ensure_processed_data(force=False)
    print("Processed listo:", outputs)
