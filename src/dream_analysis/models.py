"""Typed domain objects used across dream-analysis services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping


class DreamValidationError(ValueError):
    """Raised when a parsed dream record does not satisfy the domain model."""


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DreamValidationError(f"{field} must be a non-empty string")
    return value


def _optional_int(record: Mapping[str, Any], field: str) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool):
        raise DreamValidationError(f"{field} must be an integer or null")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DreamValidationError(f"{field} must be an integer or null") from exc


def _date_sort(record: Mapping[str, Any]) -> date | None:
    value = record.get("date_sort")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise DreamValidationError("date_sort must be an ISO date string or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DreamValidationError(
            "date_sort must use YYYY-MM-DD format or be null"
        ) from exc


@dataclass(frozen=True, slots=True)
class Dream:
    """One parsed dream-journal entry."""

    dream_id: str
    date: str
    text: str
    tags: tuple[str, ...] = ()
    year: int | None = None
    month: int | None = None
    day: int | None = None
    date_precision: str = "day"
    date_sort: date | None = None
    word_count: int = 0

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Dream":
        """Validate and convert a decoded JSON object into a dream."""
        if not isinstance(record, Mapping):
            raise DreamValidationError("dream record must be an object")

        raw_tags = record.get("tags", [])
        if not isinstance(raw_tags, list) or any(
            not isinstance(tag, str) for tag in raw_tags
        ):
            raise DreamValidationError("tags must be an array of strings")

        text = _required_text(record, "text")
        raw_word_count = record.get("word_count")
        if raw_word_count is None:
            word_count = len(text.split())
        elif isinstance(raw_word_count, bool):
            raise DreamValidationError("word_count must be a non-negative integer")
        else:
            try:
                word_count = int(raw_word_count)
            except (TypeError, ValueError) as exc:
                raise DreamValidationError(
                    "word_count must be a non-negative integer"
                ) from exc
        if word_count < 0:
            raise DreamValidationError("word_count must be a non-negative integer")

        date_precision = record.get("date_precision", "day")
        if not isinstance(date_precision, str) or not date_precision:
            raise DreamValidationError("date_precision must be a non-empty string")

        return cls(
            dream_id=_required_text(record, "dream_id"),
            date=_required_text(record, "date"),
            text=text,
            tags=tuple(raw_tags),
            year=_optional_int(record, "year"),
            month=_optional_int(record, "month"),
            day=_optional_int(record, "day"),
            date_precision=date_precision,
            date_sort=_date_sort(record),
            word_count=word_count,
        )

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used by current files."""
        return {
            "dream_id": self.dream_id,
            "date": self.date,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "date_precision": self.date_precision,
            "date_sort": self.date_sort.isoformat() if self.date_sort else None,
            "tags": list(self.tags),
            "text": self.text,
            "word_count": self.word_count,
        }


ChromaMetadata = Mapping[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One normalized result returned by the dream vector index."""

    dream_id: str
    document: str
    metadata: ChromaMetadata
    distance: float

    @property
    def date(self) -> str:
        return str(self.metadata.get("date", "unknown"))


@dataclass(frozen=True, slots=True)
class RelatedDream:
    """A dream selected as cosine-similar comparison material."""

    dream_id: str
    date: str
    text: str
    similarity: float
