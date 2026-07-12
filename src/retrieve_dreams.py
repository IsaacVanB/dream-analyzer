#!/usr/bin/env python3
"""Retrieve dreams closest to a text query from the ChromaDB index."""

from __future__ import annotations

import argparse

import chromadb
import requests


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


def preview(text: str, max_chars: int = 300) -> str:
    compact_text = " ".join(text.split())
    suffix = "..." if len(compact_text) > max_chars else ""
    return compact_text[:max_chars] + suffix


def retrieve_dreams(
    *,
    query: str,
    top_k: int = 10,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> dict:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    query_embedding = ollama_embed(query, model=embed_model)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def print_results(results: dict, *, max_chars: int = 300) -> None:
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, dream_id in enumerate(ids, start=1):
        metadata = metadatas[rank - 1]
        document = documents[rank - 1]
        distance = distances[rank - 1]

        print(f"\n{rank}. {dream_id}")
        print(f"Date: {metadata['date']}")
        if metadata.get("tags"):
            print(f"Tags: {metadata['tags']}")
        if metadata.get("word_count") is not None:
            print(f"Word count: {metadata['word_count']}")
        print(f"Distance: {distance:.4f}")
        print(f"Preview: {preview(document, max_chars=max_chars)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search dream embeddings with a text query."
    )
    parser.add_argument("query", help="Text query to search for.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of closest dreams to return.",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_PATH,
        help="Path to the persistent ChromaDB database.",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Name of the ChromaDB collection to query.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="Maximum number of characters to show per result preview.",
    )
    args = parser.parse_args()

    results = retrieve_dreams(
        query=args.query,
        top_k=args.top_k,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )
    print_results(results, max_chars=args.preview_chars)


if __name__ == "__main__":
    main()
