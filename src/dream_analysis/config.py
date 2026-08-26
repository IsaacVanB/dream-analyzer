"""Application configuration shared by library services and CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    """Connection and model defaults for the local Ollama service."""

    host: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class IndexSettings:
    """Persistent vector-index defaults."""

    path: Path = Path("data/chroma_db")
    collection_name: str = "dreams_nomic_embed_text"


@dataclass(frozen=True, slots=True)
class Settings:
    """Top-level application defaults.

    Paths intentionally remain relative to the current working directory for
    compatibility with the existing scripts during the migration.
    """

    dreams_path: Path = Path("data/dreams.jsonl")
    output_path: Path = Path("outputs")
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    index: IndexSettings = field(default_factory=IndexSettings)

