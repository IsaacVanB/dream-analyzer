"""Read-only access to parsed dream JSONL records."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from dream_analysis.dates import parse_iso_date_value, validate_date_range
from dream_analysis.models import Dream, DreamValidationError


class DreamNotFoundError(LookupError):
    """Raised when a requested dream ID does not exist."""


class CharacterDictionaryRepository:
    """Load a manually enrichable character dictionary from JSON."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def all(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as source:
                document = json.load(source)
        except json.JSONDecodeError as exc:
            raise DreamValidationError(
                f"Invalid JSON in character dictionary {self.path}: {exc.msg}"
            ) from exc

        if isinstance(document, list):
            characters = document
        elif isinstance(document, dict) and isinstance(
            document.get("characters"), list
        ):
            characters = document["characters"]
        else:
            raise DreamValidationError(
                f"Character dictionary {self.path} must be an array or contain "
                "a characters array"
            )

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, character in enumerate(characters, start=1):
            if not isinstance(character, dict):
                raise DreamValidationError(
                    f"Character {index} in {self.path} is not an object"
                )
            character_id = character.get("id")
            name = character.get("name")
            aliases = character.get("aliases", [])
            if not isinstance(character_id, str) or not character_id.strip():
                raise DreamValidationError(
                    f"Character {index} in {self.path} has no valid id"
                )
            if character_id in seen_ids:
                raise DreamValidationError(
                    f"Duplicate character id {character_id!r} in {self.path}"
                )
            if not isinstance(name, str) or not name.strip():
                raise DreamValidationError(
                    f"Character {character_id!r} in {self.path} has no valid name"
                )
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) or not alias.strip() for alias in aliases
            ):
                raise DreamValidationError(
                    f"Character {character_id!r} in {self.path} has no valid "
                    "aliases array"
                )
            for field in ("relationship", "context"):
                value = character.get(field, "")
                if not isinstance(value, str):
                    raise DreamValidationError(
                        f"Character {character_id!r} in {self.path} has a "
                        f"non-string {field}"
                    )
            history = character.get("relationship_history")
            if history is not None:
                self._validate_history(history, character_id=character_id)
            seen_ids.add(character_id)
            records.append(dict(character))
        return records

    def _validate_history(self, value: Any, *, character_id: str) -> None:
        if not isinstance(value, list):
            raise DreamValidationError(
                f"Character {character_id!r} in {self.path} has no valid "
                "relationship_history array"
            )
        for index, entry in enumerate(value, start=1):
            if not isinstance(entry, dict):
                raise DreamValidationError(
                    f"Relationship history entry {index} for character "
                    f"{character_id!r} in {self.path} is not an object"
                )
            for field in ("relationship", "context"):
                if not isinstance(entry.get(field, ""), str):
                    raise DreamValidationError(
                        f"Relationship history entry {index} for character "
                        f"{character_id!r} in {self.path} has a non-string {field}"
                    )
            for field in ("start_date", "end_date"):
                date_value = entry.get(field)
                if date_value is not None and (
                    not isinstance(date_value, str)
                    or parse_iso_date_value(date_value) is None
                ):
                    raise DreamValidationError(
                        f"Relationship history entry {index} for character "
                        f"{character_id!r} in {self.path} has an invalid {field}"
                    )
            start = parse_iso_date_value(entry.get("start_date"))
            end = parse_iso_date_value(entry.get("end_date"))
            validate_date_range(start, end)


class StructuredDreamRepository:
    """Load structured dream-feature records with character-field validation."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def all(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for line_number, record in load_jsonl_objects(self.path):
            dream_id = record.get("dream_id")
            if not isinstance(dream_id, str) or not dream_id:
                raise DreamValidationError(
                    f"Structured dream on line {line_number} of {self.path} "
                    "has no dream_id"
                )
            if dream_id in seen_ids:
                raise DreamValidationError(
                    f"Duplicate structured dream_id {dream_id!r} on line "
                    f"{line_number} of {self.path}"
                )
            names = record.get("named_characters")
            if not isinstance(names, list) or any(
                not isinstance(name, str) for name in names
            ):
                raise DreamValidationError(
                    f"Structured dream {dream_id!r} on line {line_number} of "
                    f"{self.path} has no valid named_characters array"
                )
            seen_ids.add(dream_id)
            records.append(dict(record))
        return records


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

    def tagged(
        self,
        tags: list[str] | tuple[str, ...],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Dream]:
        """Return dreams containing every requested tag.

        Matching is case-insensitive but otherwise exact, so punctuation remains
        significant (for example, ``lucid?`` is distinct from ``lucid``).
        """
        if not tags:
            raise ValueError("tags must contain at least one tag")
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("tags must contain non-empty strings")

        requested = {tag.strip().casefold() for tag in tags}
        return [
            dream
            for dream in self.between(start, end)
            if requested.issubset({tag.casefold() for tag in dream.tags})
        ]
