"""Build a running leaderboard from immutable retrieval benchmark reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dream_analysis.artifacts import write_json_atomic, write_text_atomic


LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_MARKDOWN = "leaderboard.md"
METRICS = (
    ("r_precision", "R-precision", "max"),
    ("recall_at_5", "R@5", "max"),
    ("recall_at_10", "R@10", "max"),
    ("strategy_accuracy", "Routing accuracy", "max"),
    ("error_query_count", "Errors", "min"),
    ("mean_retrieval_seconds", "Mean seconds/query", "min"),
)


def file_sha256(path: Path | str) -> str | None:
    """Return a file-content fingerprint, or ``None`` when unavailable."""
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fallback_query_fingerprint(report: dict[str, Any]) -> str | None:
    queries = report.get("queries")
    if not isinstance(queries, list):
        return None
    normalized = [
        {
            "query": item.get("query"),
            "category": item.get("category"),
            "expected_strategy": item.get("expected_strategy"),
            "relevant_dream_ids": item.get("relevant_dream_ids"),
        }
        for item in queries
        if isinstance(item, dict)
    ]
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compatibility(report: dict[str, Any]) -> tuple[str, str | None, str | None]:
    fingerprints = report.get("fingerprints") or {}
    query_fingerprint = fingerprints.get("queries_sha256")
    dream_fingerprint = fingerprints.get("dreams_sha256")
    if not isinstance(query_fingerprint, str):
        query_fingerprint = _fallback_query_fingerprint(report)
    if not isinstance(dream_fingerprint, str):
        dream_fingerprint = None
    if query_fingerprint and dream_fingerprint:
        key = f"{query_fingerprint[:12]}-{dream_fingerprint[:12]}"
    elif query_fingerprint:
        key = f"legacy-{query_fingerprint[:12]}-unknown-corpus"
    else:
        key = "legacy-unknown-suite-and-corpus"
    return key, query_fingerprint, dream_fingerprint


def _compatibility_key(
    query_fingerprint: str | None,
    dream_fingerprint: str | None,
) -> str | None:
    if query_fingerprint and dream_fingerprint:
        return f"{query_fingerprint[:12]}-{dream_fingerprint[:12]}"
    return None


def _mean_latency(report: dict[str, Any]) -> float | None:
    summary_value = (report.get("summary") or {}).get("mean_retrieval_seconds")
    if isinstance(summary_value, (int, float)) and not isinstance(
        summary_value, bool
    ):
        return float(summary_value)
    values = [
        row.get("retrieval_seconds")
        for row in report.get("queries", [])
        if isinstance(row, dict)
        and row.get("status", "ok") == "ok"
        and isinstance(row.get("retrieval_seconds"), (int, float))
        and not isinstance(row.get("retrieval_seconds"), bool)
    ]
    return sum(values) / len(values) if values else None


def _entry(report: dict[str, Any], source: Path) -> dict[str, Any] | None:
    summary = report.get("summary")
    settings = report.get("settings")
    if not isinstance(summary, dict) or not isinstance(settings, dict):
        return None
    mode = settings.get("retrieval_mode")
    if not isinstance(mode, str):
        return None
    compatibility_key, query_fingerprint, dream_fingerprint = _compatibility(
        report
    )
    experiment = report.get("experiment") or {}
    name = experiment.get("name")
    if not isinstance(name, str) or not name.strip():
        name = source.stem
    note = experiment.get("note")
    if not isinstance(note, str):
        note = ""
    metrics = {
        "r_precision": summary.get("r_precision"),
        "recall_at_5": summary.get("recall_at_5"),
        "recall_at_10": summary.get("recall_at_10"),
        "strategy_accuracy": summary.get("strategy_accuracy"),
        "error_query_count": summary.get("error_query_count"),
        "mean_retrieval_seconds": _mean_latency(report),
    }
    markdown_source = source.with_suffix(".md")
    return {
        "name": name.strip(),
        "note": note.strip(),
        "mode": mode,
        "created_at": report.get("created_at", "unknown"),
        "source_json": source.name,
        "source_markdown": (
            markdown_source.name if markdown_source.is_file() else None
        ),
        "query_count": summary.get("query_count"),
        "evaluated_query_count": summary.get("evaluated_query_count"),
        "compatibility_key": compatibility_key,
        "query_fingerprint": query_fingerprint,
        "dream_fingerprint": dream_fingerprint,
        "metrics": metrics,
        "settings": settings,
        "source_control": report.get("source_control"),
        "category_summaries": report.get("category_summaries", {}),
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _best(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for key, label, direction in METRICS:
        candidates = [
            (entry, _numeric(entry["metrics"].get(key))) for entry in entries
        ]
        candidates = [(entry, value) for entry, value in candidates if value is not None]
        if not candidates:
            continue
        target = (
            max(value for _, value in candidates)
            if direction == "max"
            else min(value for _, value in candidates)
        )
        winner_entries = [entry for entry, value in candidates if value == target]
        best[key] = {
            "label": label,
            "value": target,
            "winners": [entry["source_json"] for entry in winner_entries],
            "winner_names": [entry["name"] for entry in winner_entries],
        }
    return best


def _category_best(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted(
        {
            category
            for entry in entries
            for category in entry.get("category_summaries", {})
        }
    )
    winners: list[dict[str, Any]] = []
    for category in categories:
        candidates = []
        for entry in entries:
            summary = entry.get("category_summaries", {}).get(category, {})
            value = _numeric(summary.get("r_precision"))
            if value is not None:
                candidates.append((entry["name"], value))
        if not candidates:
            continue
        target = max(value for _, value in candidates)
        winners.append(
            {
                "category": category,
                "r_precision": target,
                "winners": [name for name, value in candidates if value == target],
            }
        )
    return winners


def build_leaderboard(
    output_dir: Path | str,
    *,
    current_query_fingerprint: str | None = None,
    current_dream_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Read benchmark JSON files and return grouped leaderboard data."""
    directory = Path(output_dir)
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for source in sorted(directory.glob("benchmark_*.json")):
        try:
            report = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("top-level JSON value is not an object")
            entry = _entry(report, source)
            if entry is None:
                raise ValueError("not a retrieval benchmark report")
        except Exception as exc:
            skipped.append({"file": source.name, "error": str(exc)})
            continue
        entries.append(entry)

    entries.sort(key=lambda item: (str(item["created_at"]), item["source_json"]), reverse=True)
    configured_current_key = _compatibility_key(
        current_query_fingerprint,
        current_dream_fingerprint,
    )
    current_key = configured_current_key or (
        entries[0]["compatibility_key"] if entries else None
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry["compatibility_key"], []).append(entry)
    ordered_keys = sorted(
        grouped,
        key=lambda key: str(grouped[key][0]["created_at"]),
        reverse=True,
    )
    if current_key in ordered_keys:
        ordered_keys.remove(current_key)
        ordered_keys.insert(0, current_key)
    groups = []
    for key in ordered_keys:
        group_entries = grouped[key]
        groups.append(
            {
                "compatibility_key": key,
                "current": key == current_key,
                "query_fingerprint": group_entries[0]["query_fingerprint"],
                "dream_fingerprint": group_entries[0]["dream_fingerprint"],
                "experiments": group_entries,
                "best": _best(group_entries),
                "category_best": _category_best(group_entries),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_compatibility_key": current_key,
        "experiment_count": len(entries),
        "groups": groups,
        "skipped_reports": skipped,
    }


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _display(value: Any) -> str:
    number = _numeric(value)
    if number is None:
        return "n/a"
    if float(number).is_integer() and abs(number) >= 1:
        return str(int(number))
    return f"{number:.3f}"


def _display_metric(metric: str, value: Any) -> str:
    number = _numeric(value)
    if metric == "error_query_count" and number is not None:
        return str(int(number))
    return _display(value)


def _is_winner(group: dict[str, Any], metric: str, entry: dict[str, Any]) -> bool:
    winner = group.get("best", {}).get(metric)
    return bool(winner and entry["source_json"] in winner["winners"])


def _metric_cell(group: dict[str, Any], metric: str, entry: dict[str, Any]) -> str:
    rendered = _display_metric(metric, entry["metrics"].get(metric))
    return f"**{rendered}**" if rendered != "n/a" and _is_winner(group, metric, entry) else rendered


def _experiment_link(entry: dict[str, Any]) -> str:
    label = _escape(entry["name"])
    target = entry.get("source_markdown") or entry["source_json"]
    return f"[{label}](<{target}>)"


def _group_markdown(group: dict[str, Any]) -> list[str]:
    lines = [
        f"Compatibility key: `{group['compatibility_key']}`",
        "",
        f"- Query fingerprint: `{group['query_fingerprint'] or 'unknown'}`",
        f"- Dream corpus fingerprint: `{group['dream_fingerprint'] or 'unknown'}`",
        "",
        "| Experiment | Mode | Created | Queries | R-precision | R@5 | R@10 | Routing accuracy | Errors | Mean seconds/query | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in group["experiments"]:
        lines.append(
            f"| {_experiment_link(entry)} | {_escape(entry['mode'])} | "
            f"{_escape(entry['created_at'])} | {_display(entry['query_count'])} | "
            f"{_metric_cell(group, 'r_precision', entry)} | "
            f"{_metric_cell(group, 'recall_at_5', entry)} | "
            f"{_metric_cell(group, 'recall_at_10', entry)} | "
            f"{_metric_cell(group, 'strategy_accuracy', entry)} | "
            f"{_metric_cell(group, 'error_query_count', entry)} | "
            f"{_metric_cell(group, 'mean_retrieval_seconds', entry)} | "
            f"{_escape(entry['note'])} |"
        )
    return lines


def markdown_leaderboard(leaderboard: dict[str, Any]) -> str:
    """Render a human-readable leaderboard without precision-at-k columns."""
    lines = [
        "# Retrieval evaluation leaderboard",
        "",
        f"- Generated: `{leaderboard['generated_at']}`",
        f"- Experiments: `{leaderboard['experiment_count']}`",
        "- Bold values are best within a compatible query-suite and dream-corpus group.",
        "- Routing accuracy applies only to agent runs.",
        "",
    ]
    groups = leaderboard.get("groups", [])
    if not groups:
        lines.extend(["No benchmark reports found.", ""])
        return "\n".join(lines).rstrip() + "\n"

    current = next((group for group in groups if group.get("current")), None)
    lines.extend(["## Current best results", ""])
    if current is None:
        lines.extend(
            [
                "No experiments match the currently configured query suite and dream corpus.",
                "",
            ]
        )
    else:
        for metric, _, _ in METRICS:
            winner = current.get("best", {}).get(metric)
            if not winner:
                continue
            names = ", ".join(_escape(name) for name in winner["winner_names"])
            lines.append(
                f"- {winner['label']}: "
                f"**{_display_metric(metric, winner['value'])}** — {names}"
            )
        lines.extend(["", "## Current compatible experiments", ""])
        lines.extend(_group_markdown(current))

    if current is not None and current.get("category_best"):
        lines.extend(
            [
                "",
                "### Current R-precision winners by category",
                "",
                "| Category | Best R-precision | Experiment |",
                "|---|---:|---|",
            ]
        )
        for item in current["category_best"]:
            lines.append(
                f"| {_escape(item['category'])} | {_display(item['r_precision'])} | "
                f"{_escape(', '.join(item['winners']))} |"
            )

    older = [group for group in groups if group is not current]
    if older:
        lines.extend(["", "## Older or incompatible experiment groups", ""])
        for group in older:
            lines.extend(
                [
                    f"### `{group['compatibility_key']}`",
                    "",
                    *_group_markdown(group),
                    "",
                ]
            )
    skipped = leaderboard.get("skipped_reports", [])
    if skipped:
        lines.extend(["", "## Skipped reports", ""])
        for item in skipped:
            lines.append(f"- `{item['file']}`: {_escape(item['error'])}")
    return "\n".join(lines).rstrip() + "\n"


def write_leaderboard(
    output_dir: Path | str,
    *,
    current_query_fingerprint: str | None = None,
    current_dream_fingerprint: str | None = None,
) -> tuple[Path, Path]:
    """Rebuild and atomically write JSON and Markdown leaderboard artifacts."""
    directory = Path(output_dir)
    leaderboard = build_leaderboard(
        directory,
        current_query_fingerprint=current_query_fingerprint,
        current_dream_fingerprint=current_dream_fingerprint,
    )
    json_path = directory / LEADERBOARD_JSON
    markdown_path = directory / LEADERBOARD_MARKDOWN
    write_json_atomic(json_path, leaderboard)
    write_text_atomic(markdown_path, markdown_leaderboard(leaderboard))
    return json_path, markdown_path
