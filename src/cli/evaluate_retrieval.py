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
from typing import Any, Iterable

from cli import dream_agent
from dream_analysis.agent import DreamRagAgent, ToolExecution
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
    agent: DreamRagAgent,
    chat_model: str,
    max_tool_calls: int,
    num_ctx: int,
    num_predict: int,
    temperature: float,
) -> list[dict[str, Any]]:
    """Run the agent's tool planner and score its fused dream evidence."""
    rows: list[dict[str, Any]] = []
    for number, item in enumerate(queries, start=1):
        relevant_ids = item["relevant_dream_ids"]
        progress = f"[{number}/{len(queries)}]"
        print(f"{progress} Starting: {item['query']}", flush=True)
        started = perf_counter()
        response = agent.answer(
            item["query"],
            chat_model=chat_model,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            max_tool_calls=max_tool_calls,
            synthesize=False,
        )
        ranked = DreamRagAgent.rank_dream_evidence(response.tool_executions)
        retrieved_ids = [dream["dream_id"] for dream in ranked]
        retrieval_seconds = round(perf_counter() - started, 3)
        rows.append(
            {
                "query": item["query"],
                "category": item["category"],
                **retrieval_metrics(retrieved_ids, relevant_ids),
                "returned_count": len(retrieved_ids),
                "retrieval_seconds": retrieval_seconds,
                "retrieved_dream_ids": retrieved_ids,
                "relevant_dream_ids": relevant_ids,
                "tool_calls": [
                    tool_execution_report(execution)
                    for execution in response.tool_executions
                ],
                "unexecuted_tool_calls": [
                    {"name": call.name, "arguments": dict(call.arguments)}
                    for call in response.unexecuted_tool_calls
                ],
            }
        )
        print(
            f"{progress} Finished in {retrieval_seconds:.3f}s: "
            f"{len(response.tool_executions)} tool call(s), "
            f"{len(retrieved_ids)} unique dream(s)",
            flush=True,
        )
    return rows


def tool_execution_report(execution: ToolExecution) -> dict[str, Any]:
    """Keep the agent's generated queries and filters without journal text."""
    result = execution.report_result or execution.result
    return {
        "name": execution.name,
        "arguments": dict(execution.arguments),
        "ok": bool(execution.result.get("ok")),
        "cached": execution.cached,
        "result_count": result.get("result_count"),
        "dream_ids": [
            str(dream.get("dream_id")) for dream in result.get("dreams", []) or []
        ],
        "error": execution.result.get("error"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_evaluation_queries(args.queries_path)
    agent = dream_agent.build_agent(
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_chars_per_dream=args.max_chars_per_dream,
        dreams_path=args.dreams_path,
        structured_dreams_path=args.structured_dreams_path,
        characters_path=args.characters_path,
    )
    rows = evaluate_queries(
        payload["queries"],
        agent=agent,
        chat_model=args.chat_model,
        max_tool_calls=args.max_tool_calls,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
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
            "chat_model": args.chat_model,
            "dreams_path": str(args.dreams_path),
            "structured_dreams_path": str(args.structured_dreams_path),
            "characters_path": str(args.characters_path),
            "results_per_semantic_search": args.top_k,
            "max_tool_calls": args.max_tool_calls,
            "max_chars_per_dream": args.max_chars_per_dream,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
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
        f"- Agent chat model: `{settings['chat_model']}`",
        f"- Maximum tool calls per query: `{settings['max_tool_calls']}`",
        f"- Results per semantic search: `{settings['results_per_semantic_search']}`",
        "- Retrieval uses the dream agent's tool planner and reciprocal-rank fusion; final answer synthesis is skipped.",
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
    lines.extend(
        [
            "",
            "## Agent retrieval trace",
            "",
            "| query | generated tool calls |",
            "|---|---|",
        ]
    )
    for row in report["queries"]:
        query = row["query"].replace("|", "\\|").replace("\n", " ")
        calls = "; ".join(
            f"`{call['name']}({json.dumps(call['arguments'], ensure_ascii=False, sort_keys=True)})`"
            for call in row["tool_calls"]
        )
        escaped_calls = calls.replace("|", "\\|") or "none"
        lines.append(f"| {query} | {escaped_calls} |")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dream-agent tool retrieval using labeled relevance judgments."
        )
    )
    parser.add_argument("--queries-path", type=Path, default=QUERIES_PATH)
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=dream_agent.DEFAULT_SETTINGS.dreams_path,
    )
    parser.add_argument(
        "--structured-dreams-path",
        type=Path,
        default=dream_agent.STRUCTURED_DREAMS_PATH,
    )
    parser.add_argument(
        "--characters-path",
        type=Path,
        default=dream_agent.CHARACTERS_PATH,
    )
    parser.add_argument(
        "--chroma-path",
        default=str(dream_agent.DEFAULT_SETTINGS.index.path),
    )
    parser.add_argument(
        "--collection-name", default=dream_agent.DEFAULT_SETTINGS.index.collection_name
    )
    parser.add_argument(
        "--embed-model", default=dream_agent.DEFAULT_SETTINGS.ollama.embedding_model
    )
    parser.add_argument(
        "--chat-model",
        default=dream_agent.DEFAULT_SETTINGS.ollama.chat_model,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum dreams returned by each semantic search call.",
    )
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-chars-per-dream", type=int, default=2500)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 10 <= args.top_k <= 20:
        parser.error("--top-k must be between 10 and 20 for evaluation at 10")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be positive")
    if args.max_chars_per_dream < 1:
        parser.error("--max-chars-per-dream must be positive")
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
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
