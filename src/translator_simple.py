"""Servicio simple de traduccion con API externa y cache en memoria.

Este modulo encapsula una integracion minima con API para cumplir requisito de
consumo de servicio externo sin complejizar la arquitectura.
"""

from __future__ import annotations

from typing import Dict, Tuple

import requests

_TRANSLATION_CACHE: Dict[Tuple[str, str, str], str] = {}
_TRANSLATION_URL = "https://api.mymemory.translated.net/get"


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    """Traduce texto entre idiomas con fallback silencioso.

    Si la API falla o responde inesperadamente, retorna el texto original.

    Args:
        text: Texto a traducir.
        source_lang: Codigo ISO de idioma origen.
        target_lang: Codigo ISO de idioma destino.

    Returns:
        Texto traducido cuando es posible; en otro caso, el texto original.
    """
    if not text or source_lang == target_lang:
        return text

    key = (text, source_lang, target_lang)
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    try:
        response = requests.get(
            _TRANSLATION_URL,
            params={"q": text, "langpair": f"{source_lang}|{target_lang}"},
            timeout=4,
        )
        response.raise_for_status()
        payload = response.json()
        translated = payload.get("responseData", {}).get("translatedText")
        if not translated:
            return text
        _TRANSLATION_CACHE[key] = translated
        return translated
    except Exception:
        return text
