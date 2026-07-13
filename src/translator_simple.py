"""Servicio de traduccion con API externa, cache persistente y precalentamiento.

Incluye:
- Cache en memoria para respuestas rapidas dentro de la sesion.
- Cache en disco para reutilizar traducciones entre ejecuciones.
- Cola de warm-up en segundo plano para precargar textos frecuentes.
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests

_TRANSLATION_URL = "https://api.mymemory.translated.net/get"
_CACHE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "cache" / "translations_es_en.json"

_TRANSLATION_CACHE: Dict[Tuple[str, str, str], str] = {}
_CACHE_LOCK = threading.Lock()

_WARMUP_QUEUE: queue.Queue[Tuple[str, str, str]] = queue.Queue()
_QUEUED_KEYS: set[Tuple[str, str, str]] = set()
_WORKER_STARTED = False


def _cache_key(text: str, source_lang: str, target_lang: str) -> Tuple[str, str, str]:
    return text, source_lang, target_lang


def _serialize_key(key: Tuple[str, str, str]) -> str:
    text, src, tgt = key
    return f"{src}|{tgt}|{text}"


def _deserialize_key(raw: str) -> Tuple[str, str, str] | None:
    parts = raw.split("|", 2)
    if len(parts) != 3:
        return None
    return parts[2], parts[0], parts[1]


def _load_cache_from_disk() -> None:
    if not _CACHE_PATH.exists():
        return
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        with _CACHE_LOCK:
            for raw_key, translated in payload.items():
                if not isinstance(translated, str):
                    continue
                key = _deserialize_key(raw_key)
                if key is None:
                    continue
                _TRANSLATION_CACHE[key] = translated
    except Exception:
        # Cache corrupta o no parseable: se ignora y se reconstruye.
        return


def _save_cache_to_disk() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_LOCK:
            payload = {
                _serialize_key(key): value
                for key, value in _TRANSLATION_CACHE.items()
                if isinstance(value, str)
            }
        _CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Fallo de persistencia no debe romper la app.
        return


def _translate_via_api(text: str, source_lang: str, target_lang: str, timeout: int = 4) -> str:
    response = requests.get(
        _TRANSLATION_URL,
        params={"q": text, "langpair": f"{source_lang}|{target_lang}"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    translated = payload.get("responseData", {}).get("translatedText")
    return translated if isinstance(translated, str) and translated else text


def _warmup_worker() -> None:
    while True:
        text, src, tgt = _WARMUP_QUEUE.get()
        key = _cache_key(text, src, tgt)
        try:
            with _CACHE_LOCK:
                if key in _TRANSLATION_CACHE:
                    continue
            translated = _translate_via_api(text, src, tgt, timeout=4)
            if translated and translated != text:
                with _CACHE_LOCK:
                    _TRANSLATION_CACHE[key] = translated
                _save_cache_to_disk()
        except Exception:
            # El warm-up es best-effort.
            pass
        finally:
            with _CACHE_LOCK:
                _QUEUED_KEYS.discard(key)
            _WARMUP_QUEUE.task_done()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    thread = threading.Thread(target=_warmup_worker, name="translator-warmup", daemon=True)
    thread.start()
    _WORKER_STARTED = True


def prewarm_translations(
    texts: Iterable[str],
    source_lang: str = "es",
    target_lang: str = "en",
) -> None:
    """Encola traducciones para precalentamiento asincrono.

    No bloquea la ejecucion principal.
    """
    _ensure_worker_started()
    for text in texts:
        if not text:
            continue
        key = _cache_key(text, source_lang, target_lang)
        with _CACHE_LOCK:
            if key in _TRANSLATION_CACHE or key in _QUEUED_KEYS:
                continue
            _QUEUED_KEYS.add(key)
        _WARMUP_QUEUE.put(key)


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    """Traduce texto entre idiomas con cache persistente y fallback silencioso."""
    if not text or source_lang == target_lang:
        return text

    key = _cache_key(text, source_lang, target_lang)
    with _CACHE_LOCK:
        cached = _TRANSLATION_CACHE.get(key)
    if cached:
        return cached

    try:
        translated = _translate_via_api(text, source_lang, target_lang, timeout=4)
        if translated and translated != text:
            with _CACHE_LOCK:
                _TRANSLATION_CACHE[key] = translated
            _save_cache_to_disk()
            return translated
        return text
    except Exception:
        return text


def get_cache_stats() -> dict:
    """Entrega estadisticas simples del cache para diagnostico."""
    with _CACHE_LOCK:
        return {
            "items": len(_TRANSLATION_CACHE),
            "queued": len(_QUEUED_KEYS),
            "cache_file": str(_CACHE_PATH),
        }


_load_cache_from_disk()
atexit.register(_save_cache_to_disk)
