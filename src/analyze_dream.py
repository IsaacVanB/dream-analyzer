#!/usr/bin/env python3
"""Analyze one dream with an Ollama chat model."""

from __future__ import annotations

import argparse
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any

from dream_analysis.analysis import (
    SingleDreamAnalysisService,
    format_related_context as format_analysis_context,
    format_saved_analysis as format_analysis_artifact,
    save_analysis as save_analysis_artifact,
    word_preview as make_word_preview,
)
from dream_analysis.config import Settings
from dream_analysis.index import (
    DREAM_TEXT_SEPARATOR,
    DreamIndex,
    cosine_similarity as index_cosine_similarity,
    extract_dream_text as extract_index_dream_text,
    parse_index_date as parse_stored_date,
    validate_collection_embedding_model as validate_index_embedding_model,
)
from dream_analysis.models import RelatedDream
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.repository import DreamNotFoundError, DreamRepository


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_DIR = DEFAULT_SETTINGS.output_path / "analysis"
CHROMA_PATH = str(DEFAULT_SETTINGS.index.path)
COLLECTION_NAME = DEFAULT_SETTINGS.index.collection_name
EMBED_MODEL = DEFAULT_SETTINGS.ollama.embedding_model
CHAT_MODEL = DEFAULT_SETTINGS.ollama.chat_model


def _legacy_related(item: RelatedDream) -> dict[str, Any]:
    return {
        "dream_id": item.dream_id,
        "date": item.date,
        "similarity": item.similarity,
        "text": item.text,
    }


def _related_dream(item: dict[str, Any]) -> RelatedDream:
    return RelatedDream(
        dream_id=str(item["dream_id"]),
        date=str(item.get("date", "unknown")),
        similarity=float(item["similarity"]),
        text=str(item["text"]),
    )


def _make_service(
    *,
    dreams_path: Path | None = None,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
    with_index: bool = True,
) -> SingleDreamAnalysisService:
    gateway = OllamaGateway()
    repository = DreamRepository(dreams_path) if dreams_path is not None else None
    index = (
        DreamIndex(
            path=chroma_path,
            collection_name=collection_name,
            embedding_model=embed_model,
            ollama_gateway=gateway,
        )
        if with_index
        else None
    )
    return SingleDreamAnalysisService(
        ollama_gateway=gateway,
        repository=repository,
        index=index,
    )


def load_dream_by_id(path: Path, dream_id: str) -> dict[str, Any]:
    """Compatibility wrapper returning the existing decoded-record shape."""
    try:
        return DreamRepository(path).get(dream_id).to_record()
    except DreamNotFoundError as exc:
        raise ValueError(str(exc)) from exc


def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> list[float]:
    """Compatibility wrapper around the shared Ollama gateway."""
    return OllamaGateway().embed_one(text, model=model)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    return index_cosine_similarity(first, second)


def extract_dream_text(document: str) -> str:
    return extract_index_dream_text(document)


def word_preview(text: str, *, max_words: int = 25) -> str:
    return make_word_preview(text, max_words=max_words)


def parse_index_date(value: Any) -> Date | None:
    return parse_stored_date(value)


def parse_cli_date(value: str) -> Date:
    parsed_date = parse_index_date(value)
    if parsed_date is None:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        )
    return parsed_date


def validate_collection_embedding_model(
    collection: Any,
    *,
    collection_name: str,
    embed_model: str,
) -> None:
    validate_index_embedding_model(
        collection,
        collection_name=collection_name,
        embedding_model=embed_model,
    )


def retrieve_related_dreams(
    text: str,
    *,
    n_results: int,
    similarity_threshold: float = 0.5,
    target_dream_id: str | None = None,
    start_date: Date | None = None,
    end_date: Date | None = None,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
) -> list[dict[str, Any]]:
    """Return the most similar indexed dreams above a cosine threshold."""
    if n_results < 0:
        raise ValueError("n_results cannot be negative.")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1.")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date.")
    if n_results == 0:
        return []
    service = _make_service(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embed_model=embed_model,
    )
    return [
        _legacy_related(item)
        for item in service.find_related(
            text,
            limit=n_results,
            similarity_threshold=similarity_threshold,
            target_dream_id=target_dream_id,
            start_date=start_date,
            end_date=end_date,
        )
    ]


def format_related_context(
    related_dreams: list[dict[str, Any]],
    *,
    max_chars_per_dream: int = 1500,
) -> str:
    return format_analysis_context(
        [_related_dream(item) for item in related_dreams],
        max_chars_per_dream=max_chars_per_dream,
    )


def analyze_dream(
    text: str,
    *,
    chat_model: str = CHAT_MODEL,
    dream_id: str | None = None,
    date: str | None = None,
    tags: list[str] | None = None,
    related_dreams: list[dict[str, Any]] | None = None,
    max_chars_per_related_dream: int = 1500,
    num_ctx: int = 8192,
    num_predict: int = 1500,
    temperature: float = 0.2,
) -> str:
    """Ask Ollama for a close, evidence-based analysis of one dream."""
    service = SingleDreamAnalysisService(ollama_gateway=OllamaGateway())
    return service.analyze(
        text,
        chat_model=chat_model,
        dream_id=dream_id,
        date=date,
        tags=tags,
        related_dreams=[_related_dream(item) for item in related_dreams or []],
        max_chars_per_related_dream=max_chars_per_related_dream,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
    )


def save_analysis(
    analysis: str,
    *,
    dream_id: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    timestamp: datetime | None = None,
) -> Path:
    return save_analysis_artifact(
        analysis,
        dream_id=dream_id,
        output_dir=output_dir,
        timestamp=timestamp,
    )


def format_saved_analysis(
    analysis: str,
    *,
    target_text: str,
    dream_id: str | None,
    date: str | None,
    related_dreams: list[dict[str, Any]],
) -> str:
    return format_analysis_artifact(
        analysis,
        target_text=target_text,
        dream_id=dream_id,
        date=date,
        related_dreams=[_related_dream(item) for item in related_dreams],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a dream selected by ID or supplied as text."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dream-id",
        help="ID of a dream to load from the JSONL file.",
    )
    source_group.add_argument(
        "--text",
        help="Dream text to analyze directly.",
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--chat-model",
        default=CHAT_MODEL,
        help="Ollama chat model name.",
    )
    parser.add_argument(
        "--related-dreams",
        type=int,
        default=0,
        metavar="N",
        help="Number of similar indexed dreams to use as context.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity for a related dream. Defaults to 0.5.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_cli_date,
        help="Earliest reference dream date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_cli_date,
        help="Latest reference dream date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--chroma-path",
        default=CHROMA_PATH,
        help="Path to the persistent ChromaDB database.",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Name of the ChromaDB collection to search.",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
        help="Ollama embedding model name.",
    )
    parser.add_argument(
        "--max-chars-per-related-dream",
        type=int,
        default=1500,
        help="Maximum context characters to include per related dream.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the analysis text file is saved.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Ollama context window option.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=2500,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the chat model.",
    )
    args = parser.parse_args()

    if args.related_dreams < 0:
        parser.error("--related-dreams cannot be negative")
    if not -1.0 <= args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be between -1 and 1")
    if args.max_chars_per_related_dream < 1:
        parser.error("--max-chars-per-related-dream must be positive")
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")
    if (
        args.start_date is not None
        and args.end_date is not None
        and args.start_date > args.end_date
    ):
        parser.error("--start-date must be before or equal to --end-date")

    service = _make_service(
        dreams_path=args.dreams_path if args.dream_id is not None else None,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        with_index=args.related_dreams > 0,
    )
    if args.dream_id is not None:
        dream = service.get_dream(args.dream_id)
        text = dream.text
        dream_id = dream.dream_id
        date = dream.date
        tags = dream.tags
        print(f"Analyzing {dream_id} | {date or 'date unknown'}\n")
    else:
        text = args.text
        dream_id = None
        date = None
        tags = None

    related = service.find_related(
        text,
        limit=args.related_dreams,
        similarity_threshold=args.similarity_threshold,
        target_dream_id=dream_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    related_dreams = [_legacy_related(item) for item in related]
    if args.related_dreams:
        if args.start_date is not None or args.end_date is not None:
            print(
                "Reference dream date range: "
                f"{args.start_date.isoformat() if args.start_date else 'earliest'} "
                f"through {args.end_date.isoformat() if args.end_date else 'latest'}"
            )
        print(
            f"Using {len(related)} related dream(s) with cosine "
            f"similarity >= {args.similarity_threshold:.2f}:"
        )
        for item in related:
            print(
                f"- {item.dream_id} | {item.date} | "
                f"similarity={item.similarity:.4f}"
            )
            print(f"  Preview: {word_preview(item.text)}")
        print()

    analysis = service.analyze(
        text,
        chat_model=args.chat_model,
        dream_id=dream_id,
        date=date,
        tags=tags,
        related_dreams=related,
        max_chars_per_related_dream=args.max_chars_per_related_dream,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
    )
    print(analysis)
    saved_content = format_saved_analysis(
        analysis,
        target_text=text,
        dream_id=dream_id,
        date=date,
        related_dreams=related_dreams,
    )
    output_path = save_analysis(
        saved_content,
        dream_id=dream_id,
        output_dir=args.output_dir,
    )
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
