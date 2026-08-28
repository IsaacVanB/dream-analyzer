#!/usr/bin/env python3
"""Build a persistent ChromaDB index for parsed dream records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from dream_analysis.config import Settings
from dream_analysis.index import (
    DreamIndex,
    build_document as build_index_document,
    build_metadata as build_index_metadata,
)
from dream_analysis.models import Dream
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.repository import DreamRepository


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
CHROMA_PATH = str(DEFAULT_SETTINGS.index.path)
COLLECTION_NAME = DEFAULT_SETTINGS.index.collection_name
EMBED_MODEL = DEFAULT_SETTINGS.ollama.embedding_model


def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> list[float]:
    """Compatibility wrapper around the shared Ollama gateway."""
    return OllamaGateway().embed_one(text, model=model)


def load_dreams(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper returning decoded record dictionaries."""
    return [dream.to_record() for dream in DreamRepository(path).all()]


def dream_id_for(dream: dict[str, Any], index: int) -> str:
    dream_id = dream.get("dream_id")
    if dream_id:
        return str(dream_id)
    return f"dream-{index:06d}"


def dream_word_count(dream: dict[str, Any]) -> int:
    return int(dream.get("word_count", dream.get("word count", 0)))


def metadata_int(value: Any) -> int:
    return int(value) if value is not None else 0


def document_value(value: Any) -> str:
    return "" if value is None else str(value)


def build_document(dream: dict[str, Any], dream_id: str) -> str:
    record = dict(dream)
    record["dream_id"] = dream_id
    return build_index_document(Dream.from_record(record))


def build_metadata(dream: dict[str, Any]) -> dict[str, str | int]:
    return build_index_metadata(Dream.from_record(dream))


def recreate_collection(
    client: chromadb.PersistentClient,
    name: str,
    *,
    embed_model: str,
):
    try:
        client.delete_collection(name)
    except (NotFoundError, ValueError):
        pass
    return client.create_collection(
        name=name,
        metadata={
            "embedding_source": "dream_text",
            "embedding_model": embed_model,
        },
    )


def build_chroma_db(
    *,
    dreams_path: Path = DREAMS_PATH,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
    batch_size: int = 32,
) -> int:
    dreams = DreamRepository(dreams_path).all()
    index = DreamIndex(
        path=chroma_path,
        collection_name=collection_name,
        embedding_model=embed_model,
        ollama_gateway=OllamaGateway(),
    )
    return index.rebuild(
        dreams,
        batch_size=batch_size,
        progress=lambda current, total, dream: print(
            f"Embedding {current}/{total}: {dream.dream_id}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a persistent ChromaDB database from dream JSONL."
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_PATH,
        help="Directory where ChromaDB should persist the database.",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Name of the ChromaDB collection to recreate.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of dream texts to embed per Ollama request.",
    )
    args = parser.parse_args()

    indexed_count = build_chroma_db(
        dreams_path=args.dreams_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )
    print(f"Indexed {indexed_count} dreams in ChromaDB at {args.chroma_path}.")


if __name__ == "__main__":
    main()
