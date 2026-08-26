#!/usr/bin/env python3
"""Compute summary statistics for parsed dream records."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from dream_analysis.config import Settings
from dream_analysis.dates import (
    filter_records_by_date,
    format_period_label as shared_period_label,
    parse_date_bound,
    record_date,
)
from dream_analysis.repository import DreamRepository


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_PATH = DEFAULT_SETTINGS.output_path / "stats/dream_stats.json"
WORD_RE = re.compile(r"[A-Za-z']+")
DEFAULT_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "didn't",
    "do",
    "don't",
    "dream",
    "dreaming",
    "dreams",
    "every",
    "everyone",
    "for",
    "from",
    "get",
    "go",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "i",
    "i'd",
    "i'm",
    "in",
    "into",
    "is",
    "it",
    "it's",
    "just",
    "kept",
    "know",
    "like",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "only",
    "or",
    "out",
    "really",
    "remember",
    "said",
    "see",
    "she",
    "so",
    "some",
    "someone",
    "something",
    "that",
    "the",
    "them",
    "then",
    "there",
    "they",
    "this",
    "through",
    "to",
    "too",
    "tried",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "with",
    "woke",
    "would",
    "you",
}


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


def tokenize_words(text: str) -> list[str]:
    return [
        match.group(0).lower().strip("'")
        for match in WORD_RE.finditer(text)
        if match.group(0).strip("'")
    ]


def common_word_stats(
    dreams: list[dict[str, Any]],
    *,
    top_n: int = 20,
    stopwords: set[str] | None = None,
    min_word_length: int = 3,
) -> list[dict[str, Any]]:
    stopwords = stopwords or DEFAULT_STOPWORDS
    counts: Counter[str] = Counter()
    total_kept_words = 0

    for dream in dreams:
        for word in tokenize_words(str(dream.get("text", ""))):
            if len(word) < min_word_length or word in stopwords:
                continue
            counts[word] += 1
            total_kept_words += 1

    return [
        {
            "word": word,
            "count": count,
            "percentage": round((count / total_kept_words) * 100, 2)
            if total_kept_words
            else 0,
        }
        for word, count in counts.most_common(top_n)
    ]


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
    stopwords = load_stopwords(stopwords_path)

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
        "common_words": common_word_stats(
            dreams,
            top_n=common_words,
            stopwords=stopwords,
            min_word_length=min_word_length,
        ),
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
