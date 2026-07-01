"""Load user-defined OpenAI-compatible providers from local config."""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ModelOption = Tuple[str, str]

logger = logging.getLogger(__name__)

CUSTOM_MODELS_FILE = Path.home() / ".tradingagents" / "custom_models.json"
_PROVIDER_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _as_model_option(item: Any) -> Optional[ModelOption]:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[0]), str(item[1])

    if isinstance(item, dict):
        display = item.get("display") or item.get("name") or item.get("label")
        value = item.get("value") or item.get("model") or item.get("id")
        if display and value:
            return str(display), str(value)

    return None


def _normalize_models(models: Any) -> Optional[Dict[str, List[ModelOption]]]:
    if not isinstance(models, dict):
        return None

    normalized: Dict[str, List[ModelOption]] = {}

    for mode in ("quick", "deep"):
        raw_options = models.get(mode, [])
        options: List[ModelOption] = []

        if isinstance(raw_options, list):
            for item in raw_options:
                option = _as_model_option(item)
                if option:
                    options.append(option)

        if not options:
            options = [("Custom model ID", "custom")]
        elif all(value != "custom" for _, value in options):
            options.append(("Custom model ID", "custom"))

        normalized[mode] = options

    return normalized


def _normalize_provider(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        logger.warning(
            "Ignoring custom provider in %s: entry is not a JSON object.",
            CUSTOM_MODELS_FILE,
        )
        return None

    display_name = raw.get("display_name") or raw.get("name")
    provider_key = raw.get("provider_key") or raw.get("key")
    base_url = raw.get("base_url")
    api_key_env = raw.get("api_key_env")
    api_type = raw.get("api_type", "openai_compatible")

    if not display_name or not provider_key or not base_url:
        logger.warning(
            "Ignoring custom provider %r in %s: 'display_name', "
            "'provider_key', and 'base_url' are all required.",
            provider_key or display_name,
            CUSTOM_MODELS_FILE,
        )
        return None

    provider_key = str(provider_key).strip().lower()
    if not _PROVIDER_KEY_RE.match(provider_key):
        logger.warning(
            "Ignoring custom provider %r in %s: provider_key must match %s.",
            provider_key,
            CUSTOM_MODELS_FILE,
            _PROVIDER_KEY_RE.pattern,
        )
        return None

    if api_type != "openai_compatible":
        logger.warning(
            "Ignoring custom provider %r in %s: unsupported api_type %r "
            "(only 'openai_compatible' is supported).",
            provider_key,
            CUSTOM_MODELS_FILE,
            api_type,
        )
        return None

    models = _normalize_models(raw.get("models"))
    if not models:
        logger.warning(
            "Ignoring custom provider %r in %s: 'models' must be an object "
            "with 'quick' and/or 'deep' model lists.",
            provider_key,
            CUSTOM_MODELS_FILE,
        )
        return None

    return {
        "display_name": str(display_name),
        "provider_key": provider_key,
        "base_url": str(base_url),
        "api_key_env": str(api_key_env) if api_key_env else None,
        "api_type": api_type,
        "models": models,
    }


@functools.lru_cache(maxsize=4)
def _load_custom_providers_cached(
    path: str, mtime: float
) -> Tuple[Dict[str, Any], ...]:
    """Parse and normalize the config file.

    Memoized on ``(path, mtime)`` so the file is re-read and re-parsed only
    when it changes on disk. ``is_custom_openai_compatible_provider`` runs on
    the client-factory hot path, so the previous read-and-parse-on-every-call
    behavior was wasteful. Returns a tuple (immutable, hashable cache value);
    the public wrapper converts it back to a list of dicts.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Could not parse custom provider config %s: %s. "
            "No custom providers will be available.",
            path,
            exc,
        )
        return ()

    raw_providers = data.get("providers", []) if isinstance(data, dict) else None
    if not isinstance(raw_providers, list):
        logger.warning(
            "Custom provider config %s has no 'providers' list; ignoring.",
            path,
        )
        return ()

    providers: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in raw_providers:
        provider = _normalize_provider(raw)
        if not provider:
            continue

        key = provider["provider_key"]
        if key in seen:
            logger.warning(
                "Ignoring duplicate custom provider %r in %s (first wins).",
                key,
                path,
            )
            continue

        seen.add(key)
        providers.append(provider)

    return tuple(providers)


def load_custom_providers() -> List[Dict[str, Any]]:
    if not CUSTOM_MODELS_FILE.exists():
        return []

    try:
        mtime = CUSTOM_MODELS_FILE.stat().st_mtime
    except OSError:
        # File vanished between exists() and stat(); treat as absent.
        return []

    # Return copies so callers cannot mutate the cached entries.
    return [dict(p) for p in _load_custom_providers_cached(str(CUSTOM_MODELS_FILE), mtime)]


def clear_cache() -> None:
    """Drop the memoized config.

    The loader is keyed on the file's mtime, so a normal on-disk edit is
    picked up automatically. This helper exists for the edge case of a
    same-second rewrite (mtime unchanged) — notably in tests — and for
    callers that want to force a re-read.
    """
    _load_custom_providers_cached.cache_clear()


def get_custom_provider_choices() -> List[Tuple[str, str, str]]:
    return [
        (
            provider["display_name"],
            provider["provider_key"],
            provider["base_url"],
        )
        for provider in load_custom_providers()
    ]


def get_custom_api_key_env(provider: str) -> Optional[str]:
    provider = provider.lower()

    for item in load_custom_providers():
        if item["provider_key"] == provider:
            return item.get("api_key_env")

    return None


def get_custom_model_options(provider: str, mode: str) -> Optional[List[ModelOption]]:
    provider = provider.lower()
    mode = mode.lower()

    for item in load_custom_providers():
        if item["provider_key"] == provider:
            return item["models"].get(mode)

    return None


def get_all_custom_model_options() -> Dict[str, Dict[str, List[ModelOption]]]:
    return {
        item["provider_key"]: item["models"]
        for item in load_custom_providers()
    }


def is_custom_openai_compatible_provider(provider: str) -> bool:
    provider = provider.lower()

    return any(
        item["provider_key"] == provider
        and item.get("api_type") == "openai_compatible"
        for item in load_custom_providers()
    )
