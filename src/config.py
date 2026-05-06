# src/config.py
"""
Centralized config loader.
Merges base.yaml with env-specific overrides (dev/prod).
Secrets loaded from .env via pydantic-settings.

Usage:
    from src.config import get_config, get_env
    cfg = get_config()
    env = get_env()
    print(cfg.generation.llm.model)   # gpt-4o
    print(env.openai_api_key)          # sk-...
"""

import os
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class EnvSettings(BaseSettings):
    """API keys and secrets — loaded from .env or environment variables."""

    openai_api_key:  str = Field("", env="OPENAI_API_KEY")
    cohere_api_key:  str = Field("", env="COHERE_API_KEY")
    environment:     str = Field("dev", env="ENVIRONMENT")
    api_key:         str = Field("", env="API_KEY")   # Bearer token for /chat auth

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_config() -> DictConfig:
    """Load and merge YAML configs.  base.yaml < dev/prod.yaml."""
    env  = os.getenv("ENVIRONMENT", "dev")
    base = OmegaConf.load(CONFIG_DIR / "base.yaml")
    env_path = CONFIG_DIR / f"{env}.yaml"
    if env_path.exists():
        return OmegaConf.merge(base, OmegaConf.load(env_path))
    return base


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    return EnvSettings()
