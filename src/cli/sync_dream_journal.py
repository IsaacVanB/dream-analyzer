#!/usr/bin/env python3
"""Import an append-only dream journal and record its new dream IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dream_analysis.imports import (
    DEFAULT_IMPORT_STATE_PATH,
    AppendOnlyImportError,
    sync_journal_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize an append-only journal with parsed dream records."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--dreams-path", type=Path, default=Path("data/dreams.jsonl"))
    parser.add_argument("--state", type=Path, default=DEFAULT_IMPORT_STATE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minimum-similarity", type=float, default=0.9)
    parser.add_argument("--dream-separator-blank-lines", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = sync_journal_files(
            args.input,
            args.dreams_path,
            args.state,
            dry_run=args.dry_run,
            minimum_similarity=args.minimum_similarity,
            dream_separator_blank_lines=args.dream_separator_blank_lines,
        )
    except (ValueError, AppendOnlyImportError) as exc:
        parser.error(str(exc))
    counts = result.manifest["counts"]
    qualifier = (
        "Previously imported"
        if result.repeated
        else ("Would import" if args.dry_run else "Imported")
    )
    print(
        f"{qualifier} {result.manifest['import_id']}: "
        f"{counts['unchanged']} unchanged, {counts['edited']} edited, "
        f"{counts['new']} new."
    )
    print(f"Dreams: {args.dreams_path}")
    print(f"Import state: {args.state}")


if __name__ == "__main__":
    main()
