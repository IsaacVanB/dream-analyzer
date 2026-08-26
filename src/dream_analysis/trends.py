"""Deterministic tag-frequency trends for validated dreams."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from dream_analysis.dates import (
    filter_dreams_by_date,
    format_period_label,
    period_start,
    validate_period_frequency,
)
from dream_analysis.models import Dream


def rank_tags(dreams: Sequence[Dream], *, top_n: int) -> list[str]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    counts = Counter(tag for dream in dreams for tag in dream.tags)
    return [
        tag
        for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :top_n
        ]
    ]


def dream_counts_by_period(
    dreams: Sequence[Dream],
    *,
    frequency: str,
) -> dict[date, int]:
    validate_period_frequency(frequency)
    return dict(
        sorted(
            Counter(
                period_start(dream.date_sort, frequency=frequency)
                for dream in dreams
                if dream.date_sort is not None
            ).items()
        )
    )


class TagTrendService:
    """Build JSON-compatible tag trends without plotting or model access."""

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

    def analyze(
        self,
        *,
        frequency: str = "M",
        tags: Sequence[str] | None = None,
        top_n: int = 10,
        normalize: bool = False,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> dict[str, Any]:
        validate_period_frequency(frequency)
        dreams = self.filter_by_date(start_date=start_date, end_date=end_date)
        if not dreams:
            raise ValueError("No dreams found.")

        selected_tags = list(tags) if tags is not None else rank_tags(dreams, top_n=top_n)
        if not selected_tags:
            raise ValueError("No tags found in dreams.")
        if any(not isinstance(tag, str) or not tag for tag in selected_tags):
            raise ValueError("tags must contain non-empty strings")

        available_tags = {tag for dream in dreams for tag in dream.tags}
        period_totals = dream_counts_by_period(dreams, frequency=frequency)
        period_tag_counts: dict[date, Counter[str]] = {
            period: Counter() for period in period_totals
        }
        selected = set(selected_tags)
        for dream in dreams:
            if dream.date_sort is None:
                continue
            period = period_start(dream.date_sort, frequency=frequency)
            period_tag_counts[period].update(set(dream.tags) & selected)

        periods: list[dict[str, Any]] = []
        for period, dream_count in period_totals.items():
            counts = period_tag_counts[period]
            if normalize:
                values: dict[str, int | float] = {
                    tag: (counts[tag] / dream_count) * 100 for tag in selected_tags
                }
            else:
                values = {tag: counts[tag] for tag in selected_tags}
            periods.append(
                {
                    "period_start": period.isoformat(),
                    "period": format_period_label(period, frequency=frequency),
                    "dream_count": dream_count,
                    "values": values,
                }
            )

        known_dates = [dream.date_sort for dream in dreams if dream.date_sort is not None]
        return {
            "frequency": frequency,
            "normalized": normalize,
            "value_unit": "percent" if normalize else "count",
            "tags": selected_tags,
            "missing_tags": [tag for tag in selected_tags if tag not in available_tags],
            "dream_count": len(dreams),
            "excluded_unknown_date_count": sum(
                dream.date_sort is None for dream in self._dreams
            ),
            "start_date": _date_argument(start_date),
            "end_date": _date_argument(end_date),
            "date_min": min(known_dates).isoformat() if known_dates else None,
            "date_max": max(known_dates).isoformat() if known_dates else None,
            "periods": periods,
        }


def _date_argument(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
