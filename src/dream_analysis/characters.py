"""Deterministic character aggregation for structured dream records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from dream_analysis.dates import parse_iso_date_value


LOOKUP_SCHEMA_VERSION = 1


def character_id(name: str, used_ids: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "character"
    if base not in used_ids:
        used_ids.add(base)
        return base
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}_{suffix}"
    used_ids.add(candidate)
    return candidate


def valid_date(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    return value if parse_iso_date_value(value) is not None else None


class CharacterLookupService:
    """Aggregate character mentions without file or model access."""

    def aggregate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        temporal_context: bool = False,
    ) -> list[dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}
        for record in records:
            dream_id = _dream_id(record)
            names = _named_characters(record, dream_id=dream_id)
            date_sort = valid_date(record.get("date_sort"))
            seen_in_dream: set[str] = set()
            for raw_name in names:
                name = re.sub(r"\s+", " ", raw_name.strip())
                identity = name.casefold()
                if not name or identity in seen_in_dream:
                    continue
                seen_in_dream.add(identity)
                entry = aggregated.setdefault(
                    identity,
                    {"name": name, "dream_ids": [], "dates": []},
                )
                entry["dream_ids"].append(dream_id)
                if date_sort is not None:
                    entry["dates"].append(date_sort)

        used_ids: set[str] = set()
        characters: list[dict[str, Any]] = []
        for identity in sorted(aggregated):
            entry = aggregated[identity]
            dates = sorted(set(entry["dates"]))
            character: dict[str, Any] = {
                "id": character_id(entry["name"], used_ids),
                "name": entry["name"],
                "aliases": [],
                "mentions": {
                    "count": len(entry["dream_ids"]),
                    "first_date": dates[0] if dates else None,
                    "last_date": dates[-1] if dates else None,
                    "dream_ids": entry["dream_ids"],
                },
            }
            if temporal_context:
                character["relationship_history"] = [
                    {
                        "start_date": None,
                        "end_date": None,
                        "relationship": "",
                        "context": "",
                    }
                ]
            else:
                character["relationship"] = ""
                character["context"] = ""
            characters.append(character)
        return characters

    def build_lookup(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source: str,
        temporal_context: bool,
        generated_at: datetime,
    ) -> dict[str, Any]:
        """Build the registry envelope with an explicitly supplied timestamp."""
        timestamp = generated_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.astimezone()
        return {
            "schema_version": LOOKUP_SCHEMA_VERSION,
            "generated_at": timestamp.isoformat(timespec="seconds"),
            "source": source,
            "temporal_context": temporal_context,
            "date_format": "YYYY-MM-DD",
            "characters": self.aggregate(
                records,
                temporal_context=temporal_context,
            ),
        }


def _dream_id(record: Mapping[str, Any]) -> str:
    dream_id = record.get("dream_id")
    if not isinstance(dream_id, str) or not dream_id:
        raise ValueError("Structured dream record has no dream_id.")
    return dream_id


def _named_characters(record: Mapping[str, Any], *, dream_id: str) -> list[str]:
    names = record.get("named_characters")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Record {dream_id!r} has no valid named_characters array.")
    return names
