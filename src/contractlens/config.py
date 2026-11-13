from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("CL_DATA_DIR", "data")))
    index_dir: Path = field(default_factory=lambda: Path(_env("CL_INDEX_DIR", "data/index")))
    log_dir: Path = field(default_factory=lambda: Path(_env("CL_LOG_DIR", "logs")))

    model: str = field(default_factory=lambda: _env("CL_MODEL", "ollama_chat/gemma3:4b"))
    escalation_model: str = field(default_factory=lambda: _env("CL_ESCALATION_MODEL", ""))
    embedder: str = field(default_factory=lambda: _env("CL_EMBEDDER", "st:BAAI/bge-small-en-v1.5"))
    nli_model: str = field(default_factory=lambda: _env("CL_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall"))

    prompt_version: str = field(default_factory=lambda: _env("CL_PROMPT_VERSION", "v2"))
    top_k: int = field(default_factory=lambda: int(_env("CL_TOP_K", "4")))
    chunk_size: int = field(default_factory=lambda: int(_env("CL_CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CL_CHUNK_OVERLAP", "200")))
    use_verifier: bool = field(default_factory=lambda: _env("CL_USE_VERIFIER", "true").lower() == "true")
    confidence_threshold: float = field(default_factory=lambda: float(_env("CL_CONFIDENCE_THRESHOLD", "0.6")))

    def as_params(self) -> dict:
        params = asdict(self)
        return {k: str(v) for k, v in params.items()}


def get_settings(**overrides) -> Settings:
    settings = Settings()
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)
    return settings
