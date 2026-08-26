"""Deterministic summary statistics for validated dreams."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from dream_analysis.dates import (
    filter_dreams_by_date,
    format_period_label,
    period_start,
    validate_period_frequency,
)
from dream_analysis.models import Dream


WORD_RE = re.compile(r"[A-Za-z']+")
DEFAULT_STOPWORDS = frozenset(
    {
        "a", "about", "after", "again", "all", "also", "am", "an", "and",
        "any", "are", "as", "at", "back", "be", "because", "been", "before",
        "being", "but", "by", "can", "could", "did", "didn't", "do", "don't",
        "dream", "dreaming", "dreams", "every", "everyone", "for", "from",
        "get", "go", "got", "had", "has", "have", "he", "her", "him", "his",
        "how", "i", "i'd", "i'm", "in", "into", "is", "it", "it's", "just",
        "kept", "know", "like", "me", "my", "no", "not", "of", "on", "one",
        "only", "or", "out", "really", "remember", "said", "see", "she", "so",
        "some", "someone", "something", "that", "the", "them", "then", "there",
        "they", "this", "through", "to", "too", "tried", "up", "was", "we",
        "were", "what", "when", "where", "which", "while", "who", "with",
        "woke", "would", "you",
    }
)


def tokenize_words(text: str) -> list[str]:
    return [
        match.group(0).lower().strip("'")
        for match in WORD_RE.finditer(text)
        if match.group(0).strip("'")
    ]


def dream_summary(dream: Dream) -> dict[str, Any]:
    return {
        "dream_id": dream.dream_id,
        "date": dream.date,
        "date_precision": dream.date_precision,
        "word_count": dream.word_count,
    }


def entries_per_period(
    dreams: Sequence[Dream],
    *,
    frequency: str,
) -> list[dict[str, Any]]:
    validate_period_frequency(frequency)
    counts = Counter(
        period_start(dream.date_sort, frequency=frequency)
        for dream in dreams
        if dream.date_sort is not None
    )
    return [
        {
            "period": format_period_label(period, frequency=frequency),
            "entries": counts[period],
        }
        for period in sorted(counts)
    ]


def tag_statistics(dreams: Sequence[Dream]) -> list[dict[str, Any]]:
    dream_count = len(dreams)
    counts: Counter[str] = Counter()
    for dream in dreams:
        counts.update(set(dream.tags))
    return [
        {
            "tag": tag,
            "count": count,
            "percentage": round((count / dream_count) * 100, 2)
            if dream_count
            else 0,
        }
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def length_statistics(dreams: Sequence[Dream]) -> dict[str, Any]:
    if not dreams:
        return {
            "longest": None,
            "shortest": None,
            "average_word_count": 0,
            "median_word_count": 0,
        }

    lengths = [dream.word_count for dream in dreams]
    return {
        "longest": dream_summary(max(dreams, key=lambda dream: dream.word_count)),
        "shortest": dream_summary(min(dreams, key=lambda dream: dream.word_count)),
        "average_word_count": round(statistics.mean(lengths), 2),
        "median_word_count": round(statistics.median(lengths), 2),
    }


def common_word_statistics(
    dreams: Sequence[Dream],
    *,
    top_n: int = 20,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    min_word_length: int = 3,
) -> list[dict[str, Any]]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if min_word_length < 1:
        raise ValueError("min_word_length must be at least 1")

    excluded_words = {word.casefold() for word in stopwords}
    counts: Counter[str] = Counter()
    total_kept_words = 0
    for dream in dreams:
        for word in tokenize_words(dream.text):
            if len(word) < min_word_length or word in excluded_words:
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


class DreamStatisticsService:
    """Calculate JSON-compatible statistics without file or model access."""

    def __init__(self, dreams: Sequence[Dream]) -> None:
        self._dreams = tuple(dreams)

    def filter_by_date(
        self,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> list[Dream]:
        return filter_dreams_by_date(
            self._dreams,
            start_date=start_date,
            end_date=end_date,
        )

    def summarize(
        self,
        *,
        frequency: str = "M",
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        common_words: int = 20,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        min_word_length: int = 3,
    ) -> dict[str, Any]:
        validate_period_frequency(frequency)
        dreams = self.filter_by_date(start_date=start_date, end_date=end_date)
        known_dates = [dream.date_sort for dream in dreams if dream.date_sort is not None]

        return {
            "frequency": frequency,
            "start_date": _date_argument(start_date),
            "end_date": _date_argument(end_date),
            "dream_count": len(dreams),
            "excluded_unknown_date_count": sum(
                dream.date_sort is None for dream in self._dreams
            ),
            "date_min": min(known_dates).isoformat() if known_dates else None,
            "date_max": max(known_dates).isoformat() if known_dates else None,
            "entries_per_period": entries_per_period(dreams, frequency=frequency),
            "tag_stats": tag_statistics(dreams),
            "length_stats": length_statistics(dreams),
            "common_words": common_word_statistics(
                dreams,
                top_n=common_words,
                stopwords=stopwords,
                min_word_length=min_word_length,
            ),
        }


def _date_argument(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
