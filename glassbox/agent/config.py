"""LLM provider settings, read only from the environment (optionally via a git-ignored .env file).
Nothing in the agent names a provider: base URL, key and model IDs all come from here, so moving from NVIDIA's
development API to Nebius Token Factory is a change to .env and nothing else."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigError(RuntimeError):
    pass


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


def from_env(path: str = ".env") -> LLMConfig:
    load_env(path)
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing {', '.join(missing)}; copy .env.example to .env and fill it in")
    model = os.environ["LLM_MODEL"]
    return LLMConfig(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], model, os.environ.get("LLM_FAST_MODEL") or model)


def make_client(cfg: LLMConfig):
    from openai import OpenAI  # imported lazily so tests and the rest of the package do not need it

    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
