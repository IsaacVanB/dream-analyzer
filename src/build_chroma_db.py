#!/usr/bin/env python3
"""Build a persistent ChromaDB index for parsed dream records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
import requests


DREAMS_PATH = Path("data/dreams.jsonl")
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dreams"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"


def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": model,
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]
    return [float(value) for value in embedding]


def load_dreams(path: Path) -> list[dict[str, Any]]:
    dreams: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                try:
                    dreams.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return dreams


def dream_id_for(dream: dict[str, Any], index: int) -> str:
    dream_id = dream.get("dream_id")
    if dream_id:
        return str(dream_id)
    return f"dream-{index:06d}"


def dream_word_count(dream: dict[str, Any]) -> int:
    return int(dream.get("word_count", dream.get("word count", 0)))


def build_document(dream: dict[str, Any], dream_id: str) -> str:
    tags = dream.get("tags", [])
    tag_text = ", ".join(str(tag) for tag in tags)

    return (
        f"DREAM_ID: {dream_id}\n"
        f"DATE: {dream['date']}\n"
        f"YEAR: {dream.get('year', '')}\n"
        f"MONTH: {dream.get('month', '')}\n"
        f"TAGS: {tag_text}\n\n"
        f"--- DREAM TEXT ---\n\n"
        f"{dream['text']}"
    )


def build_metadata(dream: dict[str, Any]) -> dict[str, str | int]:
    tags = dream.get("tags", [])
    return {
        "date": str(dream["date"]),
        "year": int(dream["year"]),
        "month": int(dream["month"]),
        "tags": ", ".join(str(tag) for tag in tags),
        "word_count": dream_word_count(dream),
    }


def recreate_collection(client: chromadb.PersistentClient, name: str):
    try:
        client.delete_collection(name)
    except (NotFoundError, ValueError):
        pass
    return client.create_collection(name=name)


def build_chroma_db(
    *,
    dreams_path: Path = DREAMS_PATH,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> int:
    dreams = load_dreams(dreams_path)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = recreate_collection(client, collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    embeddings: list[list[float]] = []

    for index, dream in enumerate(dreams, start=1):
        dream_id = dream_id_for(dream, index)
        document = build_document(dream, dream_id)

        print(f"Embedding {index}/{len(dreams)}: {dream_id}")

        ids.append(dream_id)
        documents.append(document)
        metadatas.append(build_metadata(dream))
        embeddings.append(ollama_embed(document, model=embed_model))

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return len(dreams)


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
    args = parser.parse_args()

    indexed_count = build_chroma_db(
        dreams_path=args.dreams_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )
    print(f"Indexed {indexed_count} dreams in ChromaDB at {args.chroma_path}.")


if __name__ == "__main__":
    main()
