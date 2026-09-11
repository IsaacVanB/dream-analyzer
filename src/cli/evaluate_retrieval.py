#!/usr/bin/env python3
"""Evaluate ranked dream retrieval against known relevance judgments."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

from cli import basic_rag
from dream_analysis.artifacts import write_json_atomic, write_text_atomic


QUERIES_PATH = Path("data/retrieval_eval_queries.json")
OUTPUT_DIR = Path("outputs/retrieval_evaluations")
CUTOFFS = (5, 10)


def load_evaluation_queries(path: Path) -> dict[str, Any]:
    """Load and validate the labeled retrieval queries."""
    with path.open(encoding="utf-8") as query_file:
        payload = json.load(query_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("Evaluation file must contain a 'queries' list.")
    if payload.get("num_queries") != len(payload["queries"]):
        raise ValueError("num_queries does not match the number of query records.")

    seen_queries: set[str] = set()
    for index, item in enumerate(payload["queries"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query record {index} must be an object.")
        query = item.get("query")
        category = item.get("category")
        relevant_ids = item.get("relevant_dream_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query record {index} has no valid query.")
        if query in seen_queries:
            raise ValueError(f"Duplicate evaluation query: {query!r}.")
        seen_queries.add(query)
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Query {query!r} has no valid category.")
        if not isinstance(relevant_ids, list) or not all(
            isinstance(dream_id, str) and dream_id for dream_id in relevant_ids
        ):
            raise ValueError(f"Query {query!r} has invalid relevant_dream_ids.")
        if len(relevant_ids) != len(set(relevant_ids)):
            raise ValueError(f"Query {query!r} contains duplicate relevant IDs.")
    return payload


def retrieval_metrics(
    retrieved_ids: Iterable[str],
    relevant_ids: Iterable[str],
) -> dict[str, int | float | None]:
    """Calculate binary relevance metrics for one ranked result list."""
    ranked = list(retrieved_ids)
    relevant = set(relevant_ids)
    relevant_count = len(relevant)
    metrics: dict[str, int | float | None] = {
        "relevant_count": relevant_count,
    }

    for cutoff in CUTOFFS:
        hits = len(set(ranked[:cutoff]) & relevant)
        metrics[f"relevant_at_{cutoff}"] = hits
        metrics[f"precision_at_{cutoff}"] = hits / cutoff
        metrics[f"max_precision_at_{cutoff}"] = min(relevant_count, cutoff) / cutoff
        metrics[f"recall_at_{cutoff}"] = (
            hits / relevant_count if relevant_count else None
        )

    r_hits = len(set(ranked[:relevant_count]) & relevant) if relevant_count else 0
    metrics["relevant_at_r"] = r_hits
    metrics["r_precision"] = r_hits / relevant_count if relevant_count else None
    return metrics


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return macro averages, excluding undefined metrics for zero-relevance queries."""
    return {
        "query_count": len(rows),
        "queries_with_relevant_dreams": sum(row["relevant_count"] > 0 for row in rows),
        **{
            metric: _mean(row[metric] for row in rows)
            for metric in (
                "precision_at_5",
                "max_precision_at_5",
                "recall_at_5",
                "precision_at_10",
                "max_precision_at_10",
                "recall_at_10",
                "r_precision",
            )
        },
    }


def evaluate_queries(
    queries: list[dict[str, Any]],
    *,
    chroma_path: str,
    collection_name: str,
    embed_model: str,
    retrieve: Callable[..., list[dict[str, Any]]] = basic_rag.retrieve_dreams,
) -> list[dict[str, Any]]:
    """Retrieve enough results for @10 and R-precision, then score every query."""
    rows: list[dict[str, Any]] = []
    for number, item in enumerate(queries, start=1):
        relevant_ids = item["relevant_dream_ids"]
        retrieval_depth = max(max(CUTOFFS), len(relevant_ids))
        print(f"[{number}/{len(queries)}] {item['query']}")
        started = perf_counter()
        retrieved = retrieve(
            item["query"],
            top_k=retrieval_depth,
            chroma_path=chroma_path,
            collection_name=collection_name,
            embed_model=embed_model,
        )
        retrieved_ids = [str(result["dream_id"]) for result in retrieved]
        rows.append(
            {
                "query": item["query"],
                "category": item["category"],
                **retrieval_metrics(retrieved_ids, relevant_ids),
                "retrieval_depth": retrieval_depth,
                "returned_count": len(retrieved_ids),
                "retrieval_seconds": round(perf_counter() - started, 3),
                "retrieved_dream_ids": retrieved_ids,
                "relevant_dream_ids": relevant_ids,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_evaluation_queries(args.queries_path)
    rows = evaluate_queries(
        payload["queries"],
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
    )
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "queries_path": str(args.queries_path),
        "journal_file": payload.get("journal_file"),
        "settings": {
            "chroma_path": args.chroma_path,
            "collection_name": args.collection_name,
            "embed_model": args.embed_model,
            "cutoffs": list(CUTOFFS),
        },
        "summary": summarize(rows),
        "category_summaries": {
            category: summarize(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
        "queries": rows,
    }


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _summary_table(rows: Iterable[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| group | queries | P@5 | max P@5 | R@5 | P@10 | max P@10 | R@10 | R-precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in rows:
        lines.append(
            f"| {label} | {summary['query_count']} | "
            f"{_display(summary['precision_at_5'])} | "
            f"{_display(summary['max_precision_at_5'])} | "
            f"{_display(summary['recall_at_5'])} | "
            f"{_display(summary['precision_at_10'])} | "
            f"{_display(summary['max_precision_at_10'])} | "
            f"{_display(summary['recall_at_10'])} | "
            f"{_display(summary['r_precision'])} |"
        )
    return lines


def markdown_report(report: dict[str, Any]) -> str:
    """Render aggregate and per-query retrieval metrics as Markdown."""
    settings = report["settings"]
    lines = [
        "# Retrieval benchmark",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Queries: `{report['queries_path']}`",
        f"- Chroma collection: `{settings['collection_name']}`",
        f"- Embedding model: `{settings['embed_model']}`",
        "- Undefined recall and R-precision for queries with no relevant dreams are shown as `n/a` and excluded from macro averages.",
        "- Maximum P@k is `min(number of relevant dreams, k) / k`.",
        "",
        "## Macro averages",
        "",
        *_summary_table([("all", report["summary"])]),
        "",
        "## Category macro averages",
        "",
        *_summary_table(report["category_summaries"].items()),
        "",
        "## Per-query results",
        "",
        "| query | category | relevant | hits@5 | P@5 | max P@5 | R@5 | hits@10 | P@10 | max P@10 | R@10 | R-precision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["queries"]:
        query = row["query"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {query} | {row['category']} | {row['relevant_count']} | "
            f"{row['relevant_at_5']} | {_display(row['precision_at_5'])} | "
            f"{_display(row['max_precision_at_5'])} | {_display(row['recall_at_5'])} | "
            f"{row['relevant_at_10']} | {_display(row['precision_at_10'])} | "
            f"{_display(row['max_precision_at_10'])} | {_display(row['recall_at_10'])} | "
            f"{_display(row['r_precision'])} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dream retrieval using labeled relevance judgments."
    )
    parser.add_argument("--queries-path", type=Path, default=QUERIES_PATH)
    parser.add_argument("--chroma-path", default=basic_rag.CHROMA_PATH)
    parser.add_argument("--collection-name", default=basic_rag.COLLECTION_NAME)
    parser.add_argument("--embed-model", default=basic_rag.EMBED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    created = datetime.now().astimezone()
    stem = created.strftime("benchmark_%Y-%m-%d_%H-%M-%S-%f")
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    write_json_atomic(json_path, report)
    write_text_atomic(markdown_path, markdown_report(report))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
