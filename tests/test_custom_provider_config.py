"""Tests for user-defined OpenAI-compatible provider config.

Covers normalization edge cases, the public accessors, malformed-file
handling, the built-in-shadowing guard in get_known_models, and an
end-to-end client path asserting the configured api_key_env is honored at
runtime (the headline capability of the custom-provider feature).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tradingagents.llm_clients.custom_provider_config as cpc


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    """Point CUSTOM_MODELS_FILE at a writable tmp file and clear the cache.

    Returns a writer callable; each write bumps the cache via clear_cache so
    same-second rewrites are reflected.
    """
    cfg_path = tmp_path / "custom_models.json"
    monkeypatch.setattr(cpc, "CUSTOM_MODELS_FILE", cfg_path)

    def write(data) -> Path:
        if isinstance(data, str):
            cfg_path.write_text(data, encoding="utf-8")
        else:
            cfg_path.write_text(json.dumps(data), encoding="utf-8")
        cpc.clear_cache()
        return cfg_path

    cpc.clear_cache()
    yield write
    cpc.clear_cache()


def _provider(**overrides):
    base = {
        "display_name": "Xiaomi",
        "provider_key": "xiaomi",
        "base_url": "https://api.xiaomi.example/v1",
        "api_key_env": "XIAOMI_API_KEY",
        "models": {
            "quick": [["MiMo Quick", "mimo-quick"]],
            "deep": [["MiMo Deep", "mimo-deep"]],
        },
    }
    base.update(overrides)
    return base


# ---- _normalize_models ----------------------------------------------------


def test_normalize_models_accepts_list_pairs(custom_config):
    custom_config({"providers": [_provider()]})
    opts = cpc.get_custom_model_options("xiaomi", "quick")
    assert ("MiMo Quick", "mimo-quick") in opts
    # 'custom' fallback appended when not already present.
    assert ("Custom model ID", "custom") in opts


def test_normalize_models_accepts_dict_items(custom_config):
    custom_config(
        {
            "providers": [
                _provider(
                    models={
                        "quick": [{"display": "D", "value": "v1"}],
                        "deep": [{"name": "N", "model": "v2"}],
                    }
                )
            ]
        }
    )
    quick = cpc.get_custom_model_options("xiaomi", "quick")
    deep = cpc.get_custom_model_options("xiaomi", "deep")
    assert ("D", "v1") in quick
    assert ("N", "v2") in deep


def test_normalize_models_empty_gets_custom_fallback(custom_config):
    custom_config({"providers": [_provider(models={"quick": [], "deep": []})]})
    quick = cpc.get_custom_model_options("xiaomi", "quick")
    assert quick == [("Custom model ID", "custom")]


def test_normalize_models_skips_invalid_items(custom_config):
    custom_config(
        {
            "providers": [
                _provider(
                    models={
                        "quick": [["only-one"], 42, {"display": "ok", "value": "good"}],
                        "deep": [["D", "v"]],
                    }
                )
            ]
        }
    )
    quick = cpc.get_custom_model_options("xiaomi", "quick")
    values = [v for _, v in quick]
    assert "good" in values
    assert "only-one" not in values


def test_does_not_duplicate_existing_custom_option(custom_config):
    custom_config(
        {
            "providers": [
                _provider(
                    models={
                        "quick": [["X", "custom"]],
                        "deep": [["D", "v"]],
                    }
                )
            ]
        }
    )
    quick = cpc.get_custom_model_options("xiaomi", "quick")
    assert [v for _, v in quick].count("custom") == 1


# ---- _normalize_provider validation --------------------------------------


def test_missing_required_field_dropped(custom_config):
    custom_config({"providers": [_provider(base_url=None)]})
    assert cpc.load_custom_providers() == []


def test_invalid_provider_key_dropped(custom_config):
    custom_config({"providers": [_provider(provider_key="Bad Key!")]})
    assert cpc.load_custom_providers() == []


def test_non_openai_api_type_dropped(custom_config):
    custom_config({"providers": [_provider(api_type="anthropic")]})
    assert cpc.load_custom_providers() == []


def test_missing_models_dropped(custom_config):
    raw = _provider()
    del raw["models"]
    custom_config({"providers": [raw]})
    assert cpc.load_custom_providers() == []


def test_duplicate_keys_first_wins(custom_config):
    p1 = _provider(display_name="First")
    p2 = _provider(display_name="Second")
    custom_config({"providers": [p1, p2]})
    providers = cpc.load_custom_providers()
    assert len(providers) == 1
    assert providers[0]["display_name"] == "First"


def test_provider_key_lowercased(custom_config):
    custom_config({"providers": [_provider(provider_key="XiaoMi")]})
    assert cpc.is_custom_openai_compatible_provider("xiaomi")
    assert cpc.is_custom_openai_compatible_provider("XIAOMI")


# ---- accessors ------------------------------------------------------------


def test_get_custom_api_key_env(custom_config):
    custom_config({"providers": [_provider()]})
    assert cpc.get_custom_api_key_env("xiaomi") == "XIAOMI_API_KEY"
    assert cpc.get_custom_api_key_env("nope") is None


def test_api_key_env_optional(custom_config):
    p = _provider()
    del p["api_key_env"]
    custom_config({"providers": [p]})
    assert cpc.get_custom_api_key_env("xiaomi") is None
    # Provider still loads (keyless gateway).
    assert cpc.is_custom_openai_compatible_provider("xiaomi")


def test_provider_choices_shape(custom_config):
    custom_config({"providers": [_provider()]})
    choices = cpc.get_custom_provider_choices()
    assert choices == [("Xiaomi", "xiaomi", "https://api.xiaomi.example/v1")]


# ---- malformed-file handling ---------------------------------------------


def test_malformed_json_returns_empty_and_warns(custom_config, caplog):
    custom_config("{ this is not valid json ")
    with caplog.at_level("WARNING"):
        assert cpc.load_custom_providers() == []
    assert any("custom_models.json" in r.message or "Could not parse" in r.message
               for r in caplog.records)


def test_providers_not_a_list_returns_empty(custom_config):
    custom_config({"providers": {"not": "a list"}})
    assert cpc.load_custom_providers() == []


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(cpc, "CUSTOM_MODELS_FILE", tmp_path / "absent.json")
    cpc.clear_cache()
    assert cpc.load_custom_providers() == []


def test_dropped_entry_warns(custom_config, caplog):
    custom_config({"providers": [_provider(provider_key="Bad Key!")]})
    with caplog.at_level("WARNING"):
        cpc.load_custom_providers()
    assert any("provider_key" in r.message for r in caplog.records)


# ---- caller mutation isolation -------------------------------------------


def test_returned_entries_are_copies(custom_config):
    custom_config({"providers": [_provider()]})
    first = cpc.load_custom_providers()
    first[0]["display_name"] = "mutated"
    second = cpc.load_custom_providers()
    assert second[0]["display_name"] == "Xiaomi"


# ---- model_catalog integration: built-in shadowing guard -----------------


def test_get_known_models_does_not_shadow_builtin(custom_config):
    """A custom provider keyed 'openai' must not replace the built-in list."""
    from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_known_models

    custom_config(
        {
            "providers": [
                _provider(
                    provider_key="openai",
                    display_name="Rogue OpenAI",
                    models={
                        "quick": [["rogue", "rogue-model"]],
                        "deep": [["rogue", "rogue-model"]],
                    },
                )
            ]
        }
    )
    known = get_known_models()
    builtin_values = {
        v for opts in MODEL_OPTIONS["openai"].values() for _, v in opts
    }
    assert builtin_values.issubset(set(known["openai"]))
    assert "rogue-model" not in known["openai"]


def test_get_known_models_includes_new_custom_provider(custom_config):
    from tradingagents.llm_clients.model_catalog import get_known_models

    custom_config({"providers": [_provider()]})
    known = get_known_models()
    assert "xiaomi" in known
    assert "mimo-quick" in known["xiaomi"]
    assert "mimo-deep" in known["xiaomi"]


# ---- end-to-end runtime key resolution (the Critical finding) ------------


def test_custom_provider_api_key_honored_end_to_end(custom_config, monkeypatch):
    """A custom provider's configured api_key_env must be read at runtime.

    Routes through create_llm_client -> OpenAIClient.get_llm() and asserts
    the resolved ChatOpenAI uses the custom key and base_url, not the
    OPENAI_API_KEY fallback.
    """
    from tradingagents.llm_clients import create_llm_client

    custom_config({"providers": [_provider()]})
    monkeypatch.setenv("XIAOMI_API_KEY", "sk-xiaomi-secret")
    # Ensure no accidental OpenAI fallback masks a regression.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-WRONG")

    client = create_llm_client(
        provider="xiaomi",
        model="mimo-deep",
        base_url="https://api.xiaomi.example/v1",
    )
    llm = client.get_llm()

    key = llm.openai_api_key
    key_value = key.get_secret_value() if hasattr(key, "get_secret_value") else key
    assert key_value == "sk-xiaomi-secret"
    assert "xiaomi" in str(llm.openai_api_base)


def test_custom_provider_missing_key_raises(custom_config, monkeypatch):
    """Configured api_key_env that is unset must raise a clear error."""
    from tradingagents.llm_clients import create_llm_client

    custom_config({"providers": [_provider()]})
    monkeypatch.delenv("XIAOMI_API_KEY", raising=False)

    client = create_llm_client(
        provider="xiaomi",
        model="mimo-deep",
        base_url="https://api.xiaomi.example/v1",
    )
    with pytest.raises(ValueError, match="XIAOMI_API_KEY"):
        client.get_llm()


def test_custom_provider_without_key_uses_placeholder(custom_config, monkeypatch):
    """A keyless custom gateway falls back to the placeholder key."""
    from tradingagents.llm_clients import create_llm_client

    p = _provider()
    del p["api_key_env"]
    custom_config({"providers": [p]})

    client = create_llm_client(
        provider="xiaomi",
        model="mimo-deep",
        base_url="https://api.xiaomi.example/v1",
    )
    llm = client.get_llm()
    key = llm.openai_api_key
    key_value = key.get_secret_value() if hasattr(key, "get_secret_value") else key
    assert key_value == "ollama"
