#!/usr/bin/env python3
"""Compare a fixed prompt across chat and embedding model combinations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import analyze_dream
import basic_rag


CHAT_MODELS = ("qwen3:8b", "gemma3:12b")
EMBEDDING_INDEXES = (
    ("nomic-embed-text", "dreams_nomic_embed_text"),
    ("qwen3-embedding", "dreams_qwen3_embedding"),
)
OUTPUT_DIR = Path("outputs/model_comparisons")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def retrieval_summary(
    retrieved: list[dict[str, Any]],
    *,
    score_key: str,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "dream_id": item["dream_id"],
            "date": item["date"],
            score_key: item[score_key],
        }
        for rank, item in enumerate(retrieved, start=1)
    ]


def run_matrix(
    *,
    retrieve: Callable[[str, str], list[dict[str, Any]]],
    generate: Callable[[str, list[dict[str, Any]]], str],
    score_key: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for embed_model, collection_name in EMBEDDING_INDEXES:
        print(f"Retrieving with {embed_model} from {collection_name} ...")
        retrieval_started = perf_counter()
        try:
            retrieved = retrieve(embed_model, collection_name)
            retrieval_seconds = perf_counter() - retrieval_started
            retrieval_error: Exception | None = None
        except Exception as exc:  # Continue so the report captures all failures.
            retrieved = []
            retrieval_seconds = perf_counter() - retrieval_started
            retrieval_error = exc

        for chat_model in CHAT_MODELS:
            result: dict[str, Any] = {
                "chat_model": chat_model,
                "embed_model": embed_model,
                "collection_name": collection_name,
                "retrieval_seconds": round(retrieval_seconds, 3),
                "retrieved": retrieval_summary(retrieved, score_key=score_key),
            }

            if retrieval_error is not None:
                result.update(
                    {
                        "status": "error",
                        "generation_seconds": None,
                        "total_seconds": round(retrieval_seconds, 3),
                        "error_type": type(retrieval_error).__name__,
                        "error": str(retrieval_error),
                    }
                )
                results.append(result)
                print(f"    ERROR: {retrieval_error}", file=sys.stderr)
                continue

            print(f"  Generating with {chat_model} ...")
            generation_started = perf_counter()
            try:
                response = generate(chat_model, retrieved)
                generation_seconds = perf_counter() - generation_started
                result.update(
                    {
                        "status": "ok",
                        "generation_seconds": round(generation_seconds, 3),
                        "total_seconds": round(
                            retrieval_seconds + generation_seconds,
                            3,
                        ),
                        "response": response,
                    }
                )
            except Exception as exc:  # Continue through the remaining combinations.
                generation_seconds = perf_counter() - generation_started
                result.update(
                    {
                        "status": "error",
                        "generation_seconds": round(generation_seconds, 3),
                        "total_seconds": round(
                            retrieval_seconds + generation_seconds,
                            3,
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"    ERROR: {exc}", file=sys.stderr)

            results.append(result)

    return results


def run_analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.dream_id is not None:
        dream = analyze_dream.load_dream_by_id(args.dreams_path, args.dream_id)
        text = dream.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Dream {args.dream_id} has no valid text field.")
        dream_id = args.dream_id
        dream_date = dream.get("date")
        tags = dream.get("tags")
        source = {"dream_id": dream_id, "dreams_path": str(args.dreams_path)}
    else:
        text = args.text
        dream_id = None
        dream_date = None
        tags = None
        source = {"text": text}

    def retrieve(embed_model: str, collection_name: str) -> list[dict[str, Any]]:
        return analyze_dream.retrieve_related_dreams(
            text,
            n_results=args.related_dreams,
            similarity_threshold=args.similarity_threshold,
            target_dream_id=dream_id,
            start_date=args.start_date,
            end_date=args.end_date,
            chroma_path=args.chroma_path,
            collection_name=collection_name,
            embed_model=embed_model,
        )

    def generate(chat_model: str, retrieved: list[dict[str, Any]]) -> str:
        return analyze_dream.analyze_dream(
            text,
            chat_model=chat_model,
            dream_id=dream_id,
            date=dream_date,
            tags=tags,
            related_dreams=retrieved,
            max_chars_per_related_dream=args.max_chars_per_related_dream,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
        )

    results = run_matrix(
        retrieve=retrieve,
        generate=generate,
        score_key="similarity",
    )
    return {
        "mode": "analyze",
        "prompt": source,
        "settings": {
            "related_dreams": args.related_dreams,
            "similarity_threshold": args.similarity_threshold,
            "start_date": args.start_date.isoformat() if args.start_date else None,
            "end_date": args.end_date.isoformat() if args.end_date else None,
            "max_chars_per_related_dream": args.max_chars_per_related_dream,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
            "chroma_path": args.chroma_path,
        },
        "results": results,
    }


def run_rag(args: argparse.Namespace) -> dict[str, Any]:
    def retrieve(embed_model: str, collection_name: str) -> list[dict[str, Any]]:
        return basic_rag.retrieve_dreams(
            args.retrieval_query,
            top_k=args.top_k,
            chroma_path=args.chroma_path,
            collection_name=collection_name,
            embed_model=embed_model,
        )

    def generate(chat_model: str, retrieved: list[dict[str, Any]]) -> str:
        return basic_rag.ask_chat_model(
            args.question,
            retrieved,
            chat_model=chat_model,
            max_chars_per_dream=args.max_chars_per_dream,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
        )

    results = run_matrix(
        retrieve=retrieve,
        generate=generate,
        score_key="distance",
    )
    return {
        "mode": "rag",
        "prompt": {
            "question": args.question,
            "retrieval_query": args.retrieval_query,
        },
        "settings": {
            "top_k": args.top_k,
            "max_chars_per_dream": args.max_chars_per_dream,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
            "chroma_path": args.chroma_path,
        },
        "results": results,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Model comparison",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Mode: `{report['mode']}`",
        "- Chat models: `qwen3:8b`, `gemma3:12b`",
        "- Embedding models: `nomic-embed-text`, `qwen3-embedding`",
        "",
        "## Prompt",
        "",
        "```json",
        json.dumps(report["prompt"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(report["settings"], indent=2, ensure_ascii=False),
        "```",
    ]

    for index, result in enumerate(report["results"], start=1):
        lines.extend(
            [
                "",
                f"## {index}. {result['chat_model']} + {result['embed_model']}",
                "",
                f"- Collection: `{result['collection_name']}`",
                f"- Status: `{result['status']}`",
                f"- Retrieval time: `{result['retrieval_seconds']}` seconds",
                f"- Generation time: `{result['generation_seconds']}` seconds",
                f"- Total time: `{result['total_seconds']}` seconds",
            ]
        )

        if result["retrieved"]:
            score_key = (
                "similarity" if "similarity" in result["retrieved"][0] else "distance"
            )
            lines.extend(
                [
                    "",
                    f"| rank | dream_id | date | {score_key} |",
                    "|---:|---|---|---:|",
                ]
            )
            for item in result["retrieved"]:
                lines.append(
                    f"| {item['rank']} | {item['dream_id']} | {item['date']} | "
                    f"{item[score_key]:.4f} |"
                )

        lines.extend(["", "### Response", ""])
        if result["status"] == "ok":
            lines.append(result["response"].rstrip())
        else:
            lines.append(f"**{result['error_type']}:** {result['error']}")

    return "\n".join(lines).rstrip() + "\n"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chroma-path",
        default=analyze_dream.CHROMA_PATH,
        help="Path containing both ChromaDB collections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for JSON and Markdown comparison reports.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run qwen3:8b and gemma3:12b against nomic-embed-text and "
            "qwen3-embedding indexes."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compare analysis of one dream with related-dream retrieval.",
    )
    add_common_arguments(analyze_parser)
    source_group = analyze_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dream-id", help="Dream ID to load.")
    source_group.add_argument("--text", help="Dream text to analyze directly.")
    analyze_parser.add_argument(
        "--dreams-path",
        type=Path,
        default=analyze_dream.DREAMS_PATH,
    )
    analyze_parser.add_argument("--related-dreams", type=int, default=5)
    analyze_parser.add_argument("--similarity-threshold", type=float, default=0.5)
    analyze_parser.add_argument(
        "--start-date",
        type=analyze_dream.parse_cli_date,
    )
    analyze_parser.add_argument(
        "--end-date",
        type=analyze_dream.parse_cli_date,
    )
    analyze_parser.add_argument("--max-chars-per-related-dream", type=int, default=1500)
    analyze_parser.add_argument("--num-ctx", type=int, default=8192)
    analyze_parser.add_argument("--num-predict", type=int, default=2500)
    analyze_parser.add_argument("--temperature", type=float, default=0.0)
    analyze_parser.set_defaults(run=run_analyze)

    rag_parser = subparsers.add_parser(
        "rag",
        help="Compare answers to one question using a fixed retrieval query.",
    )
    add_common_arguments(rag_parser)
    rag_parser.add_argument("question", help="Question all combinations answer.")
    rag_parser.add_argument(
        "--retrieval-query",
        required=True,
        help="Fixed query embedded by both embedding models.",
    )
    rag_parser.add_argument("--top-k", type=int, default=8)
    rag_parser.add_argument("--max-chars-per-dream", type=int, default=2500)
    rag_parser.add_argument("--num-ctx", type=int, default=4096)
    rag_parser.add_argument("--num-predict", type=int, default=700)
    rag_parser.add_argument("--temperature", type=float, default=0.0)
    rag_parser.set_defaults(run=run_rag)

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")

    if args.mode == "rag":
        if args.top_k < 1:
            parser.error("--top-k must be positive")
        if args.max_chars_per_dream < 1:
            parser.error("--max-chars-per-dream must be positive")
        if not args.retrieval_query.strip():
            parser.error("--retrieval-query cannot be empty")
    else:
        if args.related_dreams < 1:
            parser.error("--related-dreams must be positive for an embedding comparison")
        if not -1.0 <= args.similarity_threshold <= 1.0:
            parser.error("--similarity-threshold must be between -1 and 1")
        if args.max_chars_per_related_dream < 1:
            parser.error("--max-chars-per-related-dream must be positive")
        if args.start_date and args.end_date and args.start_date > args.end_date:
            parser.error("--start-date must be before or equal to --end-date")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    created = datetime.now().astimezone()
    report = args.run(args)
    report["created_at"] = created.isoformat(timespec="seconds")
    report["chat_models"] = list(CHAT_MODELS)
    report["embedding_indexes"] = [
        {"embed_model": model, "collection_name": collection}
        for model, collection in EMBEDDING_INDEXES
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{created.strftime('%Y-%m-%d_%H-%M-%S-%f')}_{safe_name(args.mode)}"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")

    success_count = sum(item["status"] == "ok" for item in report["results"])
    print(f"\nCompleted {success_count}/{len(report['results'])} combinations.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    if success_count != len(report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
