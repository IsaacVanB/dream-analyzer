#!/usr/bin/env python3
"""Compute summary statistics for parsed dream records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dream_analysis.artifacts import write_json_atomic
from dream_analysis.config import Settings
from dream_analysis.dates import (
    filter_records_by_date,
    format_period_label as shared_period_label,
    parse_date_bound,
    record_date,
)
from dream_analysis.repository import DreamRepository
from dream_analysis.models import Dream
from dream_analysis.statistics import (
    DEFAULT_STOPWORDS as SERVICE_STOPWORDS,
    WORD_RE,
    DreamStatisticsService,
    common_word_statistics,
    dream_summary as summarize_dream,
    entries_per_period as service_entries_per_period,
    length_statistics,
    tag_statistics,
    tokenize_words,
)


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_PATH = DEFAULT_SETTINGS.output_path / "stats/dream_stats.json"
DEFAULT_STOPWORDS = set(SERVICE_STOPWORDS)


def load_dreams(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper returning validated record dictionaries."""
    return DreamRepository(path).records()


def parse_date_filter(date_text: str | None, *, argument_name: str) -> pd.Timestamp | None:
    parsed = parse_date_bound(date_text, argument_name=argument_name)
    return pd.Timestamp(parsed) if parsed is not None else None


def dream_stats_date(dream: dict[str, Any]) -> pd.Timestamp | None:
    parsed = record_date(dream)
    return pd.Timestamp(parsed) if parsed is not None else None


def format_period_label(period: pd.Timestamp, *, freq: str) -> str:
    return shared_period_label(period, frequency=freq)


def filter_dreams_by_date(
    dreams: list[dict[str, Any]],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    return filter_records_by_date(
        dreams,
        start_date=start_date,
        end_date=end_date,
    )


def word_count_for(dream: dict[str, Any]) -> int:
    return int(dream.get("word_count", dream.get("word count", 0)))


def load_stopwords(path: Path | None = None) -> set[str]:
    stopwords = set(DEFAULT_STOPWORDS)
    if path is None:
        return stopwords

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                stopwords.add(word)

    return stopwords


def dream_summary(dream: dict[str, Any]) -> dict[str, Any]:
    return summarize_dream(Dream.from_record(dream))


def entries_per_period(
    dreams: list[dict[str, Any]],
    *,
    freq: str,
) -> list[dict[str, Any]]:
    return service_entries_per_period(_as_dreams(dreams), frequency=freq)


def tag_stats(dreams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return tag_statistics(_as_dreams(dreams))


def length_stats(dreams: list[dict[str, Any]]) -> dict[str, Any]:
    return length_statistics(_as_dreams(dreams))


def common_word_stats(
    dreams: list[dict[str, Any]],
    *,
    top_n: int = 20,
    stopwords: set[str] | None = None,
    min_word_length: int = 3,
) -> list[dict[str, Any]]:
    return common_word_statistics(
        _as_dreams(dreams),
        top_n=top_n,
        stopwords=stopwords or DEFAULT_STOPWORDS,
        min_word_length=min_word_length,
    )


def _as_dreams(records: list[dict[str, Any]]) -> list[Dream]:
    return [Dream.from_record(record) for record in records]


def compute_dream_stats(
    *,
    dreams_path: Path = DREAMS_PATH,
    output_path: Path | None = OUTPUT_PATH,
    freq: str = "M",
    start_date: str | None = None,
    end_date: str | None = None,
    common_words: int = 20,
    stopwords_path: Path | None = None,
    min_word_length: int = 3,
) -> dict[str, Any]:
    dreams = DreamRepository(dreams_path).all()
    stopwords = load_stopwords(stopwords_path)
    service_stats = DreamStatisticsService(dreams).summarize(
        frequency=freq,
        start_date=start_date,
        end_date=end_date,
        common_words=common_words,
        stopwords=stopwords,
        min_word_length=min_word_length,
    )

    stats = {
        "dreams_path": str(dreams_path),
        "output_path": str(output_path) if output_path else None,
        "freq": service_stats.pop("frequency"),
        **service_stats,
    }

    if output_path is not None:
        write_json_atomic(output_path, stats, ensure_ascii=True)

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
    parser.add_argument(
        "--common-words",
        type=int,
        default=20,
        help="Number of most common non-trivial words to include.",
    )
    parser.add_argument(
        "--stopwords-path",
        type=Path,
        help="Optional text file of additional stopwords, one per line.",
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Minimum word length to include in common-word stats.",
    )
    args = parser.parse_args()

    stats = compute_dream_stats(
        dreams_path=args.dreams_path,
        output_path=args.output,
        freq=args.freq,
        start_date=args.start_date,
        end_date=args.end_date,
        common_words=args.common_words,
        stopwords_path=args.stopwords_path,
        min_word_length=args.min_word_length,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
