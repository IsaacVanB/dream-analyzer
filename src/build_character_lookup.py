#!/usr/bin/env python3
"""Build a fillable character registry from structured dream records."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dream_analysis.characters import (
    LOOKUP_SCHEMA_VERSION,
    CharacterLookupService,
    character_id as service_character_id,
    valid_date as service_valid_date,
)


INPUT_PATH = Path("outputs/structured_dreams/dream_features.jsonl")
OUTPUT_PATH = Path("data/characters.json")


def load_structured_dreams(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record on line {line_number} of {path} is not an object.")
            dream_id = record.get("dream_id")
            if not isinstance(dream_id, str) or not dream_id:
                raise ValueError(f"Record on line {line_number} of {path} has no dream_id.")
            names = record.get("named_characters")
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                raise ValueError(
                    f"Record {dream_id!r} has no valid named_characters array. "
                    "Rerun src/structure_dreams.py with the current schema."
                )
            records.append(record)
    return records


def character_id(name: str, used_ids: set[str]) -> str:
    return service_character_id(name, used_ids)


def valid_date(value: Any) -> str | None:
    return service_valid_date(value)


def aggregate_characters(
    records: list[dict[str, Any]],
    *,
    temporal_context: bool,
) -> list[dict[str, Any]]:
    return CharacterLookupService().aggregate(
        records,
        temporal_context=temporal_context,
    )


def build_lookup(
    records: list[dict[str, Any]],
    *,
    source_path: Path,
    temporal_context: bool,
) -> dict[str, Any]:
    return CharacterLookupService().build_lookup(
        records,
        source=str(source_path),
        temporal_context=temporal_context,
        generated_at=datetime.now().astimezone(),
    )


def save_lookup(path: Path, lookup: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {path}; it may contain user edits. "
            "Use --overwrite to replace it explicitly."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(lookup, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a fillable character lookup without additional LLM calls."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--temporal-context",
        action="store_true",
        help="Create date-bounded relationship_history templates.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing lookup, including any manual edits.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_structured_dreams(args.input)
    lookup = build_lookup(
        records,
        source_path=args.input,
        temporal_context=args.temporal_context,
    )
    save_lookup(args.output, lookup, overwrite=args.overwrite)
    print(f"Saved {len(lookup['characters'])} character(s) to {args.output}.")


if __name__ == "__main__":
    main()
