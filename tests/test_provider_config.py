"""Provider selection: Token Factory is the only provider in production, with no fallback; development may use another endpoint."""
import pytest

from glassbox.agent.config import TOKEN_FACTORY_BASE_URL, ConfigError, LLMConfig, from_env, make_client, resolve_endpoint

NONE = "/nonexistent/.env"
VARS = ("GLASSBOX_ENV", "NEBIUS_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_FAST_MODEL")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def test_production_defaults_to_token_factory_and_uses_the_nebius_key(monkeypatch):
    for k, v in {"GLASSBOX_ENV": "production", "NEBIUS_API_KEY": "nb-key", "LLM_MODEL": "big", "LLM_FAST_MODEL": "small"}.items():
        monkeypatch.setenv(k, v)
    cfg = from_env(NONE)
    assert cfg.base_url == TOKEN_FACTORY_BASE_URL and cfg.api_key == "nb-key" and cfg.provider_host == "api.tokenfactory.nebius.com"
    assert (cfg.model, cfg.fast_model) == ("big", "small")


@pytest.mark.parametrize("url", ["https://integrate.api.nvidia.com/v1", "http://api.tokenfactory.nebius.com/v1/", "https://api.tokenfactory.nebius.com.evil.test/v1/", "https://example.test/v1"])
def test_production_refuses_any_other_provider_and_does_not_fall_back(monkeypatch, url):
    for k, v in {"GLASSBOX_ENV": "production", "NEBIUS_API_KEY": "nb-key", "LLM_API_KEY": "other", "LLM_BASE_URL": url, "LLM_MODEL": "m", "LLM_FAST_MODEL": "f"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigError, match="only Nebius Token Factory"):
        from_env(NONE)


def test_production_requires_both_model_ids_and_a_key(monkeypatch):
    monkeypatch.setenv("GLASSBOX_ENV", "production"); monkeypatch.setenv("NEBIUS_API_KEY", "k"); monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(ConfigError, match="LLM_FAST_MODEL"):
        from_env(NONE)
    monkeypatch.delenv("NEBIUS_API_KEY"); monkeypatch.setenv("LLM_FAST_MODEL", "f")
    with pytest.raises(ConfigError, match="API key"):
        from_env(NONE)


def test_make_client_refuses_a_non_token_factory_host_in_production(monkeypatch):
    monkeypatch.setenv("GLASSBOX_ENV", "production")
    with pytest.raises(ConfigError):
        make_client(LLMConfig("https://integrate.api.nvidia.com/v1", "k", "m", "f"))


def test_development_may_use_another_endpoint_and_never_sends_the_nebius_key_there(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-key"); monkeypatch.setenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"); monkeypatch.setenv("LLM_API_KEY", "dev-key"); monkeypatch.setenv("LLM_MODEL", "m")
    base, key = resolve_endpoint(NONE)
    assert "nvidia.com" in base and key == "dev-key"
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(ConfigError, match="API key"):
        resolve_endpoint(NONE)  # NEBIUS_API_KEY must not leak to another host


def test_development_with_only_the_nebius_key_selects_token_factory(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-key"); monkeypatch.setenv("LLM_MODEL", "m")
    cfg = from_env(NONE)
    assert cfg.provider_host == "api.tokenfactory.nebius.com" and cfg.fast_model == "m"
    assert "nb-key" not in repr(cfg)


def test_build_app_stops_in_production_with_the_wrong_provider(monkeypatch, tmp_path):
    from glassbox.app.main import build_app
    for k, v in {"GLASSBOX_ENV": "production", "LLM_BASE_URL": "https://integrate.api.nvidia.com/v1", "LLM_API_KEY": "x", "GLASSBOX_DEMO_DIR": str(tmp_path), "GLASSBOX_WARM_REPORTS": "0"}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigError):
        build_app()
