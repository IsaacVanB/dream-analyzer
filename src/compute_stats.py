#!/usr/bin/env python3
"""Compute summary statistics for parsed dream records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DREAMS_PATH = Path("data/dreams.jsonl")
OUTPUT_PATH = Path("outputs/stats/dream_stats.json")


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


def parse_date_filter(date_text: str | None, *, argument_name: str) -> pd.Timestamp | None:
    if date_text is None:
        return None
    try:
        return pd.to_datetime(date_text)
    except ValueError as exc:
        raise ValueError(f"Invalid {argument_name}: {date_text!r}") from exc


def dream_stats_date(dream: dict[str, Any]) -> pd.Timestamp | None:
    date_text = dream.get("date_sort", dream.get("date"))
    if not date_text:
        return None
    try:
        return pd.to_datetime(date_text)
    except ValueError:
        return None


def format_period_label(period: pd.Timestamp, *, freq: str) -> str:
    if freq == "Y":
        return f"{period.year}"
    if freq == "Q":
        quarter = ((period.month - 1) // 3) + 1
        return f"Q{quarter}-{period.year}"
    return f"{period.month}-{period.year}"


def filter_dreams_by_date(
    dreams: list[dict[str, Any]],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    start = parse_date_filter(start_date, argument_name="start_date")
    end = parse_date_filter(end_date, argument_name="end_date")

    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be before or equal to end_date.")

    filtered: list[dict[str, Any]] = []
    for dream in dreams:
        date = dream_stats_date(dream)
        if date is None:
            continue
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        filtered.append(dream)

    return filtered


def word_count_for(dream: dict[str, Any]) -> int:
    return int(dream.get("word_count", dream.get("word count", 0)))


def dream_summary(dream: dict[str, Any]) -> dict[str, Any]:
    return {
        "dream_id": dream.get("dream_id"),
        "date": dream.get("date"),
        "date_precision": dream.get("date_precision", "day"),
        "word_count": word_count_for(dream),
    }


def entries_per_period(
    dreams: list[dict[str, Any]],
    *,
    freq: str,
) -> list[dict[str, Any]]:
    periods: list[pd.Timestamp] = []
    for dream in dreams:
        date = dream_stats_date(dream)
        if date is not None:
            periods.append(date.to_period(freq).to_timestamp())

    counts = Counter(periods)
    return [
        {
            "period": format_period_label(period, freq=freq),
            "entries": counts[period],
        }
        for period in sorted(counts)
    ]


def tag_stats(dreams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dream_count = len(dreams)
    counts: Counter[str] = Counter()

    for dream in dreams:
        counts.update({str(tag) for tag in dream.get("tags", [])})

    return [
        {
            "tag": tag,
            "count": count,
            "percentage": round((count / dream_count) * 100, 2) if dream_count else 0,
        }
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def length_stats(dreams: list[dict[str, Any]]) -> dict[str, Any]:
    if not dreams:
        return {
            "longest": None,
            "shortest": None,
            "average_word_count": 0,
            "median_word_count": 0,
        }

    lengths = [word_count_for(dream) for dream in dreams]
    longest = max(dreams, key=word_count_for)
    shortest = min(dreams, key=word_count_for)

    return {
        "longest": dream_summary(longest),
        "shortest": dream_summary(shortest),
        "average_word_count": round(statistics.mean(lengths), 2),
        "median_word_count": round(statistics.median(lengths), 2),
    }


def compute_dream_stats(
    *,
    dreams_path: Path = DREAMS_PATH,
    output_path: Path | None = OUTPUT_PATH,
    freq: str = "M",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    all_dreams = load_dreams(dreams_path)
    excluded_unknown_date_count = sum(
        1 for dream in all_dreams if dream_stats_date(dream) is None
    )
    dreams = filter_dreams_by_date(
        all_dreams,
        start_date=start_date,
        end_date=end_date,
    )
    dates = [dream_stats_date(dream) for dream in dreams]

    stats = {
        "dreams_path": str(dreams_path),
        "output_path": str(output_path) if output_path else None,
        "freq": freq,
        "start_date": start_date,
        "end_date": end_date,
        "dream_count": len(dreams),
        "excluded_unknown_date_count": excluded_unknown_date_count,
        "date_min": min(dates).date().isoformat() if dates else None,
        "date_max": max(dates).date().isoformat() if dates else None,
        "entries_per_period": entries_per_period(dreams, freq=freq),
        "tag_stats": tag_stats(dreams),
        "length_stats": length_stats(dreams),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for parsed dream records."
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path where JSON stats should be saved.",
    )
    parser.add_argument(
        "--freq",
        choices=["M", "Q", "Y"],
        default="M",
        help="Time grouping frequency: M=month, Q=quarter, Y=year.",
    )
    parser.add_argument(
        "--start-date",
        help="Only include dreams on or after this date, e.g. 2023-01-01.",
    )
    parser.add_argument(
        "--end-date",
        help="Only include dreams on or before this date, e.g. 2023-12-31.",
    )
    args = parser.parse_args()

    stats = compute_dream_stats(
        dreams_path=args.dreams_path,
        output_path=args.output,
        freq=args.freq,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
