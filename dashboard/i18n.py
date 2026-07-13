"""Utilidades i18n para el dashboard."""

from __future__ import annotations

import ast
from pathlib import Path

from src.translator_simple import translate_text, prewarm_translations

DEFAULT_LANG = "es"

LANG_OPTIONS = [
    {"label": "Español", "value": "es"},
    {"label": "English", "value": "en"},
]

_WARMUP_STARTED = False

# Textos de alto impacto en UX para que el primer cambio a EN responda mas rapido.
WARMUP_TEXTS = [
    "Sales & Marketing\nML Dashboard",
    "Proyecto de Ciencia de Datos",
    "IDIOMA",
    "VISTAS DE NEGOCIO",
    "Vista Ejecutiva",
    "Decisiones estratégicas",
    "Vista Operativa",
    "Ejecución de campañas",
    "Vista Técnica",
    "Arquitectura y detalle ML",
    "Distribución del Score de Conversión",
    "Impacto por Canal",
    "Impacto por País",
    "Matrices de Confusión",
    "Curvas ROC",
    "Comparativa de Métricas de Clasificación",
    "Simulación de Campaña de Upselling",
    "Análisis Exploratorio de Datos (EDA)",
    "Transformación de Datos",
    "Comparación de Resultados: Sucio vs Limpio",
    "Modelado – Clustering KMeans",
    "Evaluación de Modelos Supervisados",
    "Optimización de Hiperparámetros",
    "Dashboard Interactivo – Sales & Marketing ML",
]


def _is_human_text(text: str) -> bool:
    """Heuristica simple para extraer literales traducibles del codigo."""
    t = text.strip()
    if len(t) < 4 or len(t) > 220:
        return False
    if t.startswith(("http://", "https://", "/", "#")):
        return False
    if "{" in t or "}" in t:
        return False
    if "_" in t and " " not in t:
        return False
    if sum(ch.isalpha() for ch in t) < 3:
        return False
    return True


def _discover_warmup_texts(max_texts: int = 450) -> list[str]:
    """Descubre textos traducibles en app/pages para precalentamiento."""
    dashboard_dir = Path(__file__).resolve().parent
    candidate_files = [dashboard_dir / "app.py", *sorted((dashboard_dir / "pages").glob("*.py"))]

    found: list[str] = []
    seen: set[str] = set()

    for py_file in candidate_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.strip()
                if not _is_human_text(text):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                found.append(text)
                if len(found) >= max_texts:
                    return found
    return found


def normalize_lang(lang: str | None) -> str:
    """Normaliza el idioma de entrada a un valor soportado."""
    if lang in {"es", "en"}:
        return lang
    return DEFAULT_LANG


def tr(text: str, lang: str | None) -> str:
    """Traduce texto ES->EN cuando corresponde."""
    safe_lang = normalize_lang(lang)
    if safe_lang == "en":
        return translate_text(text, "es", "en")
    return text


def start_i18n_warmup() -> None:
    """Inicia una sola vez el precalentamiento asincrono de traducciones."""
    global _WARMUP_STARTED
    if _WARMUP_STARTED:
        return

    discovered = _discover_warmup_texts()
    merged: list[str] = []
    seen: set[str] = set()
    for text in [*WARMUP_TEXTS, *discovered]:
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)

    prewarm_translations(merged, "es", "en")
    _WARMUP_STARTED = True
