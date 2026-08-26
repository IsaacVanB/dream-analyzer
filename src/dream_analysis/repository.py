"""Read-only access to parsed dream JSONL records."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from dream_analysis.dates import validate_date_range
from dream_analysis.models import Dream, DreamValidationError


class DreamNotFoundError(LookupError):
    """Raised when a requested dream ID does not exist."""


def load_jsonl_objects(path: Path | str) -> list[tuple[int, dict[str, Any]]]:
    """Decode JSONL objects while preserving their original line numbers."""
    source_path = Path(path)
    records: list[tuple[int, dict[str, Any]]] = []
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DreamValidationError(
                    f"Invalid JSON on line {line_number} of {source_path}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise DreamValidationError(
                    f"Dream on line {line_number} of {source_path} is not an object"
                )
            records.append((line_number, record))
    return records


class DreamRepository:
    """Load and query dreams stored as one JSON object per line."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def all(self) -> list[Dream]:
        dreams: list[Dream] = []
        seen_ids: set[str] = set()

        for line_number, record in load_jsonl_objects(self.path):
            try:
                dream = Dream.from_record(record)
            except DreamValidationError as exc:
                raise DreamValidationError(
                    f"Invalid dream on line {line_number} of {self.path}: {exc}"
                ) from exc
            if dream.dream_id in seen_ids:
                raise DreamValidationError(
                    f"Duplicate dream_id {dream.dream_id!r} on line "
                    f"{line_number} of {self.path}"
                )
            seen_ids.add(dream.dream_id)
            dreams.append(dream)

        return dreams

    def records(self) -> list[dict[str, Any]]:
        """Return validated dreams in the existing JSON-compatible shape."""
        return [dream.to_record() for dream in self.all()]

    def get(self, dream_id: str) -> Dream:
        if not dream_id:
            raise ValueError("dream_id cannot be empty")
        for dream in self.all():
            if dream.dream_id == dream_id:
                return dream
        raise DreamNotFoundError(f"Dream ID not found in {self.path}: {dream_id}")

    def between(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Dream]:
        """Return dreams within an inclusive date range.

        Records without a sortable date are omitted when either bound is used.
        """
        validate_date_range(start, end)
        if start is None and end is None:
            return self.all()

        selected: list[Dream] = []
        for dream in self.all():
            if dream.date_sort is None:
                continue
            if start is not None and dream.date_sort < start:
                continue
            if end is not None and dream.date_sort > end:
                continue
            selected.append(dream)
        return selected
