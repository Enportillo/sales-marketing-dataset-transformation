"""Utilidades i18n para el dashboard."""

from __future__ import annotations

from src.translator_simple import translate_text

DEFAULT_LANG = "es"

LANG_OPTIONS = [
    {"label": "Español", "value": "es"},
    {"label": "English", "value": "en"},
]


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
