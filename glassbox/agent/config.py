"""LLM provider settings, read only from the environment (optionally via a git-ignored .env file).
Nothing else in the agent names a provider: base URL, key and model IDs all come from here.

Production (GLASSBOX_ENV=production, set in the Dockerfile) accepts ONE provider, Nebius Token Factory, and refuses to start with anything else:
there is no fallback to another provider. Development (variable unset) may point LLM_BASE_URL at any OpenAI-compatible endpoint."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


TOKEN_FACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
TOKEN_FACTORY_HOST = "api.tokenfactory.nebius.com"


class ConfigError(RuntimeError):
    pass


def is_production() -> bool:
    return os.environ.get("GLASSBOX_ENV", "").strip().lower() == "production"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str       # reasoning model: writes the final report
    fast_model: str  # runs the tool-calling steps (defaults to `model`)

    @property
    def provider_host(self) -> str:
        """Recorded in every report so a run shows which provider it called (Token Factory for the submission)."""
        return urlparse(self.base_url).netloc

    def __repr__(self) -> str:  # never print the key
        return f"LLMConfig(host={self.provider_host!r}, model={self.model!r}, fast_model={self.fast_model!r}, api_key=<hidden>)"


def load_env(path: str = ".env") -> None:
    """Minimal KEY=VALUE parser. Existing environment variables win. Comments and blank lines are skipped."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def resolve_endpoint(path: str = ".env") -> tuple[str, str]:
    """(base_url, api_key). Token Factory is the default when NEBIUS_API_KEY is set or in production; NEBIUS_API_KEY is only ever sent to Token Factory.
    In production any other host is an error, so a stray LLM_BASE_URL cannot send requests elsewhere."""
    load_env(path)
    nebius_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    base = os.environ.get("LLM_BASE_URL", "").strip() or (TOKEN_FACTORY_BASE_URL if (nebius_key or is_production()) else "")
    if not base:
        raise ConfigError("missing LLM_BASE_URL (or NEBIUS_API_KEY for Nebius Token Factory); copy .env.example to .env and fill it in")
    u = urlparse(base)
    if is_production() and not (u.scheme == "https" and u.netloc == TOKEN_FACTORY_HOST):
        raise ConfigError(f"GLASSBOX_ENV=production allows only Nebius Token Factory (https://{TOKEN_FACTORY_HOST}/v1/); LLM_BASE_URL points at {u.netloc!r}. Unset it or fix it.")
    key = (nebius_key or os.environ.get("LLM_API_KEY", "").strip()) if u.netloc == TOKEN_FACTORY_HOST else os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        raise ConfigError("missing API key: set NEBIUS_API_KEY (Token Factory) or LLM_API_KEY")
    return base, key


def from_env(path: str = ".env") -> LLMConfig:
    base, key = resolve_endpoint(path)
    wanted = ("LLM_MODEL", "LLM_FAST_MODEL") if is_production() else ("LLM_MODEL",)
    missing = [k for k in wanted if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing {', '.join(missing)}; set the Token Factory model IDs (docs/TOKEN_FACTORY.md)")
    model = os.environ["LLM_MODEL"]
    return LLMConfig(base, key, model, os.environ.get("LLM_FAST_MODEL") or model)


def make_client(cfg: LLMConfig):
    from openai import OpenAI  # imported lazily so tests and the rest of the package do not need it

    if is_production() and cfg.provider_host != TOKEN_FACTORY_HOST:  # second check, at the single place a client is created
        raise ConfigError(f"production must call Token Factory, not {cfg.provider_host!r}")
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
