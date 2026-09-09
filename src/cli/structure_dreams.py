#!/usr/bin/env python3
"""Extract grounded, structured features from dream journal entries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from dream_analysis.artifacts import write_text_atomic
from dream_analysis.config import Settings
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.structuring import (
    ARRAY_FIELDS,
    BOOLEAN_FIELDS,
    DEFAULT_STRUCTURING_MODEL,
    DREAM_FEATURE_SCHEMA,
    LEVELS,
    SCHEMA_VERSION,
    DreamStructuringService,
    build_extraction_messages,
    build_record,
    extract_features,
    load_dreams,
    load_existing_records,
    select_dreams,
    select_pending_dreams,
    serialize_records,
    validate_features,
)


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_PATH = DEFAULT_SETTINGS.output_path / "structured_dreams/dream_features.jsonl"
MODEL = DEFAULT_STRUCTURING_MODEL

# Preserve imports used by existing callers while implementations live in the
# reusable service module.
__all__ = [
    "ARRAY_FIELDS",
    "BOOLEAN_FIELDS",
    "DREAM_FEATURE_SCHEMA",
    "LEVELS",
    "MODEL",
    "SCHEMA_VERSION",
    "DreamStructuringService",
    "build_extraction_messages",
    "build_record",
    "extract_features",
    "load_dreams",
    "load_existing_records",
    "save_records",
    "select_dreams",
    "select_pending_dreams",
    "serialize_records",
    "validate_features",
]


def save_records(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically save structured records as JSONL."""
    write_text_atomic(path, serialize_records(records))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured JSONL features from dream journal entries."
    )
    parser.add_argument("--dream-id", help="Process only the specified dream.")
    parser.add_argument("--dreams-path", type=Path, default=DREAMS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess selected dreams already present in the output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")

    dreams = select_dreams(load_dreams(args.dreams_path), args.dream_id)
    records = load_existing_records(args.output)
    pending = select_pending_dreams(
        dreams,
        records,
        overwrite=args.overwrite,
    )
    skipped = len(dreams) - len(pending)
    if not pending:
        print(f"Nothing to process; skipped {skipped} existing dream(s).")
        print(f"Output: {args.output}")
        return

    failures = 0
    started = perf_counter()
    service = DreamStructuringService(
        ollama_gateway=OllamaGateway(),
        model=args.model,
        num_ctx=args.num_ctx,
    )
    for index, dream in enumerate(pending, start=1):
        dream_id = str(dream.get("dream_id", "<unknown>"))
        print(f"Structuring {index}/{len(pending)}: {dream_id}")
        try:
            records[dream_id] = service.structure(dream)
            save_records(args.output, records)
        except Exception as exc:
            failures += 1
            print(f"ERROR {dream_id}: {exc}", file=sys.stderr)

    elapsed = perf_counter() - started
    completed = len(pending) - failures
    print(
        f"\nCompleted {completed}/{len(pending)} dream(s) in {elapsed:.1f} seconds; "
        f"skipped {skipped}."
    )
    print(f"Output: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
