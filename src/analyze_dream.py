#!/usr/bin/env python3
"""Analyze one dream with an Ollama chat model."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import ollama
import requests


DREAMS_PATH = Path("data/dreams.jsonl")
OUTPUT_DIR = Path("outputs/analysis")
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dreams"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DREAM_TEXT_SEPARATOR = "--- DREAM TEXT ---"


def load_dream_by_id(path: Path, dream_id: str) -> dict[str, Any]:
    """Return the JSONL record matching dream_id."""
    with path.open("r", encoding="utf-8") as dream_file:
        for line_number, line in enumerate(dream_file, start=1):
            if not line.strip():
                continue
            try:
                dream = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if dream.get("dream_id") == dream_id:
                return dream

    raise ValueError(f"Dream ID not found in {path}: {dream_id}")


def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": model, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return [float(value) for value in response.json()["embedding"]]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match.")

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(first, second))
    return dot_product / (first_norm * second_norm)


def extract_dream_text(document: str) -> str:
    """Extract raw dream text from a formatted Chroma document."""
    _, separator, dream_text = document.partition(DREAM_TEXT_SEPARATOR)
    return (dream_text if separator else document).strip()


def word_preview(text: str, *, max_words: int = 25) -> str:
    words = text.split()
    preview = " ".join(words[:max_words])
    return preview + ("..." if len(words) > max_words else "")


def parse_index_date(value: Any) -> Date | None:
    if not value:
        return None
    try:
        return Date.fromisoformat(str(value))
    except ValueError:
        return None


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

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    validate_collection_embedding_model(
        collection,
        collection_name=collection_name,
        embed_model=embed_model,
    )
    records = collection.get(include=["documents", "metadatas", "embeddings"])

    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    embeddings = records.get("embeddings")
    if embeddings is None:
        raise ValueError("The ChromaDB collection contains no embeddings.")

    target_embedding: list[float] | None = None
    if target_dream_id is not None:
        for indexed_id, embedding in zip(records["ids"], embeddings):
            if indexed_id == target_dream_id:
                target_embedding = [float(value) for value in embedding]
                break
    if target_embedding is None:
        target_embedding = ollama_embed(text, model=embed_model)

    related: list[dict[str, Any]] = []
    for dream_id, document, metadata, embedding in zip(
        records["ids"], documents, metadatas, embeddings
    ):
        if dream_id == target_dream_id:
            continue
        metadata = metadata or {}
        related_date = parse_index_date(metadata.get("date_sort"))
        if start_date is not None or end_date is not None:
            if related_date is None:
                continue
            if start_date is not None and related_date < start_date:
                continue
            if end_date is not None and related_date > end_date:
                continue

        similarity = cosine_similarity(
            target_embedding,
            [float(value) for value in embedding],
        )
        if similarity < similarity_threshold:
            continue

        related.append(
            {
                "dream_id": dream_id,
                "date": metadata.get("date", "unknown"),
                "similarity": similarity,
                "text": extract_dream_text(document or ""),
            }
        )

    related.sort(key=lambda item: item["similarity"], reverse=True)
    return related[:n_results]


def format_related_context(
    related_dreams: list[dict[str, Any]],
    *,
    max_chars_per_dream: int = 1500,
) -> str:
    if not related_dreams:
        return "No related dreams were supplied."

    blocks: list[str] = []
    for item in related_dreams:
        dream_text = item["text"]
        if len(dream_text) > max_chars_per_dream:
            dream_text = dream_text[:max_chars_per_dream] + "\n[TRUNCATED]"
        blocks.append(
            f"RELATED_DREAM_ID: {item['dream_id']}\n"
            f"DATE: {item['date']}\n"
            f"COSINE_SIMILARITY: {item['similarity']:.4f}\n\n"
            f"{dream_text}"
        )
    return "\n\n---\n\n".join(blocks)


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
    if not text.strip():
        raise ValueError("Dream text cannot be empty.")

    system_prompt = (
        "You analyze an individual dream as a narrative and subjective mental "
        "experience. Begin with what is concretely happening, then make careful "
        "interpretive hypotheses grounded in the supplied text. Discuss taboo, "
        "sexual, violent, shameful, disturbing, or contradictory material "
        "directly when it is present; do not sanitize it or avoid it. Do not try "
        "to validate, reassure, comfort, flatter, or morally judge the dreamer. "
        "Do not use universal dream dictionaries, fixed symbolic meanings, or "
        "claims such as 'X always symbolizes Y.' Treat interpretations as "
        "possibilities rather than facts, and distinguish evidence from "
        "inference. Do not diagnose mental illness or infer real-world events "
        "that the dream does not establish. Related dreams are comparison "
        "material only: use them to support or complicate interpretations of "
        "the target dream, but do not transfer their details into the target."
    )

    metadata_lines = []
    if dream_id is not None:
        metadata_lines.append(f"DREAM_ID: {dream_id}")
    if date is not None:
        metadata_lines.append(f"DATE: {date}")
    if tags:
        metadata_lines.append(f"JOURNAL_TAGS: {', '.join(tags)}")
    metadata = "\n".join(metadata_lines) or "SOURCE: text supplied directly"
    related_context = format_related_context(
        related_dreams or [],
        max_chars_per_dream=max_chars_per_related_dream,
    )

    user_prompt = f"""
/no_think

{metadata}

DREAM TEXT:
{text}

RELATED DREAMS FOR COMPARISON:
{related_context}

Analyze this dream using these sections:
1. What happens: a concise account of the events, shifts, characters, and setting.
2. Emotional and relational dynamics: tensions, desires, fears, power relations,
   contradictions, and changes in the dreamer's position.
3. Themes and motifs: the strongest recurring ideas or images, with evidence.
4. Interpretation: several plausible readings tied closely to details in the
   dream, including uncomfortable readings when supported. When related dreams
   are supplied, cite their IDs when they corroborate or contrast with a reading.
5. Uncertainties: details whose meaning depends on personal context, plus a few
   focused questions that would help distinguish between interpretations.
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
        return "[No analysis returned by chat model.]"

    done_reason = getattr(response, "done_reason", None)
    if done_reason is None and isinstance(response, dict):
        done_reason = response.get("done_reason")
    if done_reason == "length":
        content += (
            "\n\n[The model reached the generation limit. Rerun with a larger "
            "--num-predict value.]"
        )
    return content


def save_analysis(
    analysis: str,
    *,
    dream_id: str | None = None,
    output_dir: Path = OUTPUT_DIR,
    timestamp: datetime | None = None,
) -> Path:
    """Save an analysis and return its output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datetime_text = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")

    if dream_id is None:
        filename = f"{datetime_text}.txt"
    else:
        safe_dream_id = re.sub(r"[^A-Za-z0-9._-]", "_", dream_id)
        filename = f"{safe_dream_id}_{datetime_text}.txt"

    output_path = output_dir / filename
    output_path.write_text(analysis.rstrip() + "\n", encoding="utf-8")
    return output_path


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

    if args.dream_id is not None:
        dream = load_dream_by_id(args.dreams_path, args.dream_id)
        text = dream.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Dream {args.dream_id} has no valid text field.")
        dream_id = args.dream_id
        date = dream.get("date")
        tags = dream.get("tags")
        print(f"Analyzing {dream_id} | {date or 'date unknown'}\n")
    else:
        text = args.text
        dream_id = None
        date = None
        tags = None

    if args.related_dreams < 0:
        parser.error("--related-dreams cannot be negative")
    if not -1.0 <= args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be between -1 and 1")
    if args.max_chars_per_related_dream < 1:
        parser.error("--max-chars-per-related-dream must be positive")
    if (
        args.start_date is not None
        and args.end_date is not None
        and args.start_date > args.end_date
    ):
        parser.error("--start-date must be before or equal to --end-date")

    related_dreams = retrieve_related_dreams(
        text,
        n_results=args.related_dreams,
        similarity_threshold=args.similarity_threshold,
        target_dream_id=dream_id,
        start_date=args.start_date,
        end_date=args.end_date,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )
    if args.related_dreams:
        if args.start_date is not None or args.end_date is not None:
            print(
                "Reference dream date range: "
                f"{args.start_date.isoformat() if args.start_date else 'earliest'} "
                f"through {args.end_date.isoformat() if args.end_date else 'latest'}"
            )
        print(
            f"Using {len(related_dreams)} related dream(s) with cosine "
            f"similarity >= {args.similarity_threshold:.2f}:"
        )
        for item in related_dreams:
            print(
                f"- {item['dream_id']} | {item['date']} | "
                f"similarity={item['similarity']:.4f}"
            )
            print(f"  Preview: {word_preview(item['text'])}")
        print()

    analysis = analyze_dream(
        text,
        chat_model=args.chat_model,
        dream_id=dream_id,
        date=date,
        tags=tags,
        related_dreams=related_dreams,
        max_chars_per_related_dream=args.max_chars_per_related_dream,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
    )
    print(analysis)
    output_path = save_analysis(
        analysis,
        dream_id=dream_id,
        output_dir=args.output_dir,
    )
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
