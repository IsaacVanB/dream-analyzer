"""Shared parsing and filtering helpers for dream record dates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from dream_analysis.models import Dream


RecordT = TypeVar("RecordT", bound=Mapping[str, Any])
SUPPORTED_PERIOD_FREQUENCIES = frozenset({"M", "Q", "Y"})


def parse_iso_date_value(value: str | None) -> date | None:
    """Parse an optional strict ISO calendar date without raising."""
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_date_value(value: Any) -> date | None:
    """Parse supported stored or CLI date values without raising."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    for format_string in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue
    return None


def format_period_label(period: Any, *, frequency: str) -> str:
    """Format a pandas-like monthly, quarterly, or yearly period value."""
    validate_period_frequency(frequency)
    if frequency == "Y":
        return f"{period.year}"
    if frequency == "Q":
        quarter = ((period.month - 1) // 3) + 1
        return f"Q{quarter}-{period.year}"
    return f"{period.month}-{period.year}"


def validate_period_frequency(frequency: str) -> None:
    if frequency not in SUPPORTED_PERIOD_FREQUENCIES:
        choices = ", ".join(sorted(SUPPORTED_PERIOD_FREQUENCIES))
        raise ValueError(f"frequency must be one of: {choices}")


def period_start(value: date, *, frequency: str) -> date:
    """Return the first calendar day of the requested reporting period."""
    validate_period_frequency(frequency)
    if frequency == "Y":
        return date(value.year, 1, 1)
    if frequency == "Q":
        quarter_month = ((value.month - 1) // 3) * 3 + 1
        return date(value.year, quarter_month, 1)
    return date(value.year, value.month, 1)


def parse_date_bound(
    value: str | date | datetime | None,
    *,
    argument_name: str,
) -> date | None:
    """Parse an optional user-supplied bound or raise a contextual error."""
    if value is None:
        return None
    parsed = parse_date_value(value)
    if parsed is None:
        raise ValueError(f"Invalid {argument_name}: {value!r}")
    return parsed


def record_date(record: Mapping[str, Any]) -> date | None:
    """Return a record's sortable date using established field precedence.

    When ``date_sort`` is present, even as an invalid or empty value, it is
    authoritative. The display ``date`` is only a fallback for older records
    that do not contain ``date_sort``.
    """
    if "date_sort" in record:
        return parse_date_value(record.get("date_sort"))
    return parse_date_value(record.get("date"))


def validate_date_range(start: date | None, end: date | None) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be before or equal to end_date.")


def filter_records_by_date(
    records: Sequence[RecordT],
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> list[RecordT]:
    """Return records with known dates inside an inclusive range."""
    start = parse_date_bound(start_date, argument_name="start_date")
    end = parse_date_bound(end_date, argument_name="end_date")
    validate_date_range(start, end)

    selected: list[RecordT] = []
    for record in records:
        value = record_date(record)
        if value is None:
            continue
        if start is not None and value < start:
            continue
        if end is not None and value > end:
            continue
        selected.append(record)
    return selected


def filter_dreams_by_date(
    dreams: Sequence["Dream"],
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    include_unknown_dates: bool = False,
) -> list["Dream"]:
    """Return typed dreams within an inclusive date range.

    Unknown dates are excluded by default because statistics and trend reports
    cannot place them on a timeline. Callers that are not producing a temporal
    report may opt in to retaining them.
    """
    start = parse_date_bound(start_date, argument_name="start_date")
    end = parse_date_bound(end_date, argument_name="end_date")
    validate_date_range(start, end)

    selected: list["Dream"] = []
    for dream in dreams:
        value = dream.date_sort
        if value is None:
            if include_unknown_dates and start is None and end is None:
                selected.append(dream)
            continue
        if start is not None and value < start:
            continue
        if end is not None and value > end:
            continue
        selected.append(dream)
    return selected
