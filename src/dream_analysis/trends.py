"""Deterministic tag-frequency trends for validated dreams."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from dream_analysis.dates import (
    filter_dreams_by_date,
    format_period_label,
    parse_date_bound,
    period_start,
    validate_period_frequency,
)
from dream_analysis.models import Dream


def rank_tags(dreams: Sequence[Dream], *, top_n: int) -> list[str]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    counts: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    for dream in dreams:
        identities: set[str] = set()
        for tag in dream.tags:
            identity = tag.casefold()
            display_names.setdefault(identity, tag)
            identities.add(identity)
        counts.update(identities)
    return [
        display_names[identity]
        for identity, _ in sorted(
            counts.items(), key=lambda item: (-item[1], display_names[item[0]].casefold())
        )[:top_n]
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
        include_empty_periods: bool = False,
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
        selected_identities = [tag.casefold() for tag in selected_tags]
        if len(set(selected_identities)) != len(selected_identities):
            raise ValueError("tags must not contain case-insensitive duplicates")

        available_tags = {tag.casefold() for dream in dreams for tag in dream.tags}
        period_totals = dream_counts_by_period(dreams, frequency=frequency)
        if include_empty_periods and period_totals:
            period_totals = _fill_empty_periods(
                period_totals,
                frequency=frequency,
                start_date=parse_date_bound(
                    start_date, argument_name="start_date"
                ),
                end_date=parse_date_bound(end_date, argument_name="end_date"),
            )
        period_tag_counts: dict[date, Counter[str]] = {
            period: Counter() for period in period_totals
        }
        selected = set(selected_identities)
        for dream in dreams:
            if dream.date_sort is None:
                continue
            period = period_start(dream.date_sort, frequency=frequency)
            dream_tags = {tag.casefold() for tag in dream.tags}
            period_tag_counts[period].update(dream_tags & selected)

        periods: list[dict[str, Any]] = []
        for period, dream_count in period_totals.items():
            counts = period_tag_counts[period]
            if normalize:
                values: dict[str, int | float] = {
                    tag: (counts[identity] / dream_count) * 100
                    if dream_count
                    else 0.0
                    for tag, identity in zip(selected_tags, selected_identities)
                }
            else:
                values = {
                    tag: counts[identity]
                    for tag, identity in zip(selected_tags, selected_identities)
                }
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
            "missing_tags": [
                tag
                for tag, identity in zip(selected_tags, selected_identities)
                if identity not in available_tags
            ],
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


def _fill_empty_periods(
    counts: dict[date, int],
    *,
    frequency: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[date, int]:
    """Fill gaps across the observed or explicitly requested period range."""
    first = (
        period_start(start_date, frequency=frequency)
        if start_date is not None
        else min(counts)
    )
    last = (
        period_start(end_date, frequency=frequency)
        if end_date is not None
        else max(counts)
    )
    filled: dict[date, int] = {}
    current = first
    while current <= last:
        filled[current] = counts.get(current, 0)
        current = _next_period(current, frequency=frequency)
    return filled


def _next_period(value: date, *, frequency: str) -> date:
    validate_period_frequency(frequency)
    if frequency == "Y":
        return date(value.year + 1, 1, 1)
    month_increment = 3 if frequency == "Q" else 1
    month_index = value.year * 12 + value.month - 1 + month_increment
    return date(month_index // 12, month_index % 12 + 1, 1)


def _date_argument(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
