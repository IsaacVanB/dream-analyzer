#!/usr/bin/env python3
"""Answer questions using retrieved dream entries as context."""

from __future__ import annotations

import argparse
from typing import Any

import chromadb
import ollama
import requests


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dreams"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b"
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


def validate_collection_embedding_model(
    collection: Any,
    *,
    collection_name: str,
    embed_model: str,
) -> None:
    """Ensure queries use the model that produced the stored embeddings."""
    metadata = collection.metadata or {}
    indexed_model = metadata.get("embedding_model")

    if indexed_model == embed_model:
        return

    if indexed_model is None:
        detail = "does not record an embedding model"
    else:
        detail = f"was built with embedding model {indexed_model!r}"

    raise ValueError(
        f"ChromaDB collection {collection_name!r} {detail}, but the requested "
        f"embedding model is {embed_model!r}. Rebuild a separate collection "
        "with src/build_chroma_db.py using matching --collection-name and "
        "--embed-model values."
    )


def retrieve_dreams(
    query: str,
    *,
    top_k: int = 8,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> list[dict[str, Any]]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    validate_collection_embedding_model(
        collection,
        collection_name=collection_name,
        embed_model=embed_model,
    )
    query_embedding = ollama_embed(query, model=embed_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    for dream_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "dream_id": dream_id,
                "date": metadata["date"],
                "distance": float(distance),
                "document": document,
            }
        )

    return retrieved


def clean_retrieval_query(query: str) -> str:
    return query.strip().strip('"').strip("'").strip()


def generate_retrieval_query(
    question: str,
    *,
    chat_model: str = CHAT_MODEL,
) -> str:
    system_prompt = (
        "You convert user questions into keyword-expanded semantic search "
        "queries for retrieving relevant dream journal entries. Return only "
        "the search query text. Do not answer the question."
    )
    user_prompt = f"""
/no_think

QUESTION:
{question}

TASK:
Write one concise keyword query for dream retrieval. Use 6 to 10 words total.
Do not write a sentence. Do not include filler words like "dreams about",
"patterns", "themes", "analyze", or "compare". Include the core image plus a
few distinct variants or adjacent dream-language terms. Avoid repeating the
same root idea more than twice.

Examples:
- Question: What patterns appear in dreams about hidden rooms?
- Query: hidden room hallway extra room concealed door behind wall

- Question: How do school anxiety dreams show up?
- Query: school class exam final late campus anxiety
"""

    response = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 80,
        },
    )

    retrieval_query = clean_retrieval_query(response["message"]["content"])
    return retrieval_query or question


def format_context(
    retrieved: list[dict[str, Any]],
    *,
    max_chars_per_dream: int = 2500,
) -> str:
    blocks: list[str] = []

    for item in retrieved:
        text = item["document"]
        if len(text) > max_chars_per_dream:
            text = text[:max_chars_per_dream] + "\n[TRUNCATED]"

        block = (
            f"### DREAM_ID: {item['dream_id']}\n"
            f"### DATE: {item['date']}\n"
            f"### RETRIEVAL_DISTANCE: {item['distance']:.4f}\n\n"
            f"{text}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def ask_chat_model(
    question: str,
    retrieved: list[dict[str, Any]],
    *,
    chat_model: str = CHAT_MODEL,
    max_chars_per_dream: int = 2500,
    num_ctx: int = 4096,
    num_predict: int = 700,
    temperature: float = 0.1,
) -> str:
    context = format_context(
        retrieved,
        max_chars_per_dream=max_chars_per_dream,
    )

    system_prompt = (
        "You are analyzing a private dream journal. "
        "Use only the supplied dream entries. "
        "Do not invent dates, dream IDs, people, events, or themes. "
        "If the supplied entries are insufficient, say so. "
        "Be concise and cite DREAM_ID and DATE for every claim."
    )

    user_prompt = f"""
/no_think

QUESTION:
{question}

RETRIEVED DREAM ENTRIES:
{context}

TASK:
Answer the question using only the retrieved dream entries.

Return:
1. A compact table with columns: dream_id | date | relevant evidence | conflict/theme
2. A short synthesis of recurring patterns
"""

    response = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )

    content = response["message"]["content"]
    if not content:
        return "[No answer returned by chat model.]"
    return content


def print_retrieved(retrieved: list[dict[str, Any]]) -> None:
    print("\nRetrieved dreams:")
    for item in retrieved:
        print(
            f"- {item['dream_id']} | {item['date']} | "
            f"distance={item['distance']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a question using retrieved dream journal entries."
    )
    parser.add_argument("question", help="Question or prompt to answer.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of dream entries to retrieve.",
    )
    parser.add_argument(
        "--retrieval-query",
        help=(
            "Optional focused query to embed for dream retrieval. If omitted, "
            "the chat model generates one from the question."
        ),
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
        "--chat-model",
        default=CHAT_MODEL,
        help="Ollama chat model name.",
    )
    parser.add_argument(
        "--max-chars-per-dream",
        type=int,
        default=2500,
        help="Maximum context characters to include per retrieved dream.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Ollama context window option.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=700,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the chat model.",
    )
    args = parser.parse_args()

    retrieval_query = args.retrieval_query
    if retrieval_query is None:
        retrieval_query = generate_retrieval_query(
            args.question,
            chat_model=args.chat_model,
        )
        print(f"\nGenerated retrieval query: {retrieval_query}")
    else:
        print(f"\nRetrieval query: {retrieval_query}")

    retrieved = retrieve_dreams(
        retrieval_query,
        top_k=args.top_k,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )

    print_retrieved(retrieved)
    print("\n--- ANSWER ---\n")
    answer = ask_chat_model(
        args.question,
        retrieved,
        chat_model=args.chat_model,
        max_chars_per_dream=args.max_chars_per_dream,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
