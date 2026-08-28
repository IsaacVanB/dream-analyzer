#!/usr/bin/env python3
"""Answer questions using retrieved dream entries as context."""

from __future__ import annotations

import argparse
from typing import Any

from dream_analysis.config import Settings
from dream_analysis.index import (
    DreamIndex,
    validate_collection_embedding_model as validate_index_embedding_model,
)
from dream_analysis.models import SearchResult
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.rag import (
    DirectRagService,
    clean_retrieval_query as clean_query,
    format_context as format_rag_context,
)


DEFAULT_SETTINGS = Settings()
CHROMA_PATH = str(DEFAULT_SETTINGS.index.path)
COLLECTION_NAME = DEFAULT_SETTINGS.index.collection_name
EMBED_MODEL = DEFAULT_SETTINGS.ollama.embedding_model
CHAT_MODEL = DEFAULT_SETTINGS.ollama.chat_model


def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> list[float]:
    """Compatibility wrapper around the shared Ollama gateway."""
    return OllamaGateway().embed_one(text, model=model)


def validate_collection_embedding_model(
    collection: Any,
    *,
    collection_name: str,
    embed_model: str,
) -> None:
    """Ensure queries use the model that produced the stored embeddings."""
    validate_index_embedding_model(
        collection,
        collection_name=collection_name,
        embedding_model=embed_model,
    )


def _make_service(
    *,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> DirectRagService:
    gateway = OllamaGateway()
    index = DreamIndex(
        path=chroma_path,
        collection_name=collection_name,
        embedding_model=embed_model,
        ollama_gateway=gateway,
    )
    return DirectRagService(ollama_gateway=gateway, index=index)


def _legacy_result(item: SearchResult) -> dict[str, Any]:
    return {
        "dream_id": item.dream_id,
        "date": item.date,
        "distance": item.distance,
        "document": item.document,
    }


def _search_result(item: dict[str, Any]) -> SearchResult:
    return SearchResult(
        dream_id=str(item["dream_id"]),
        document=str(item["document"]),
        metadata={"date": str(item.get("date", "unknown"))},
        distance=float(item["distance"]),
    )


def retrieve_dreams(
    query: str,
    *,
    top_k: int = 8,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> list[dict[str, Any]]:
    return [
        _legacy_result(item)
        for item in _make_service(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embed_model=embed_model,
        ).retrieve(query, top_k=top_k)
    ]


def clean_retrieval_query(query: str) -> str:
    return clean_query(query)


def generate_retrieval_query(
    question: str,
    *,
    chat_model: str = CHAT_MODEL,
) -> str:
    service = DirectRagService(ollama_gateway=OllamaGateway())
    return service.generate_retrieval_query(
        question,
        chat_model=chat_model,
    )


def format_context(
    retrieved: list[dict[str, Any]],
    *,
    max_chars_per_dream: int = 2500,
) -> str:
    return format_rag_context(
        [_search_result(item) for item in retrieved],
        max_chars_per_dream=max_chars_per_dream,
    )


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
    service = DirectRagService(ollama_gateway=OllamaGateway())
    return service.answer(
        question,
        [_search_result(item) for item in retrieved],
        chat_model=chat_model,
        max_chars_per_dream=max_chars_per_dream,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
    )


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

    service = _make_service(
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )

    retrieval_query = args.retrieval_query
    if retrieval_query is None:
        retrieval_query = service.generate_retrieval_query(
            args.question,
            chat_model=args.chat_model,
        )
        print(f"\nGenerated retrieval query: {retrieval_query}")
    else:
        print(f"\nRetrieval query: {retrieval_query}")

    matches = service.retrieve(retrieval_query, top_k=args.top_k)
    retrieved = [_legacy_result(item) for item in matches]

    print_retrieved(retrieved)
    print("\n--- ANSWER ---\n")
    answer = service.answer(
        args.question,
        matches,
        chat_model=args.chat_model,
        max_chars_per_dream=args.max_chars_per_dream,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
