"""Integracion de API REST externa para enriquecimiento ETL con cache local.

Este modulo consume una API publica de tipo de cambio para agregar variables
auxiliares al dataset procesado. Incluye cache en disco, TTL y fallback seguro.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "outputs" / "cache" / "external_api_cache.json"
FX_ENDPOINT = "https://open.er-api.com/v6/latest/USD"
FX_TTL_HOURS = 24
FALLBACK_USD_TO_CLP = 900.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(payload: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Cache best-effort: nunca rompe ETL.
        return


def _is_fresh(iso_timestamp: str, ttl_hours: int) -> bool:
    try:
        cached_at = datetime.fromisoformat(iso_timestamp)
    except Exception:
        return False
    return _utc_now() - cached_at <= timedelta(hours=ttl_hours)


def get_usd_to_clp_rate(ttl_hours: int = FX_TTL_HOURS) -> float:
    """Obtiene tasa USD->CLP desde API REST publica con cache y fallback.

    Returns:
        float: tasa de conversion USD a CLP.
    """
    cache = _load_cache()
    fx_payload = cache.get("fx_usd_clp", {}) if isinstance(cache, dict) else {}

    if isinstance(fx_payload, dict):
        if _is_fresh(str(fx_payload.get("cached_at", "")), ttl_hours):
            rate = fx_payload.get("rate")
            if isinstance(rate, (int, float)) and rate > 0:
                return float(rate)

    last_exc: Exception | None = None
    for _ in range(3):
        try:
            response = requests.get(FX_ENDPOINT, timeout=5)
            response.raise_for_status()
            payload = response.json()
            rates = payload.get("rates", {}) if isinstance(payload, dict) else {}
            rate = rates.get("CLP")
            if not isinstance(rate, (int, float)) or rate <= 0:
                raise ValueError("Respuesta API sin tasa CLP valida")

            value = float(rate)
            cache["fx_usd_clp"] = {
                "rate": value,
                "source": FX_ENDPOINT,
                "cached_at": _utc_now().isoformat(),
            }
            _save_cache(cache)
            return value
        except Exception as exc:
            last_exc = exc

    # Fallback seguro: usa cache previo aunque este vencido, o constante.
    stale = fx_payload.get("rate") if isinstance(fx_payload, dict) else None
    if isinstance(stale, (int, float)) and stale > 0:
        return float(stale)

    _ = last_exc  # silencia variable no usada para linters.
    return FALLBACK_USD_TO_CLP
