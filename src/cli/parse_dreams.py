#!/usr/bin/env python3
"""Parse a dream journal into one JSON object per dream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dream_analysis.artifacts import write_text_atomic
from dream_analysis.parser import (
    DATE_RE,
    TAG_LINE_RE,
    WORD_RE,
    JournalParser,
    build_dream,
    detect_dream_separator_blank_lines,
    normalize_date_parts,
    normalize_year,
    parse_journal,
    parse_tag_line,
    validate_date_parts,
    word_count,
)


__all__ = [
    "DATE_RE",
    "TAG_LINE_RE",
    "WORD_RE",
    "JournalParser",
    "build_dream",
    "detect_dream_separator_blank_lines",
    "normalize_date_parts",
    "normalize_year",
    "parse_journal",
    "parse_tag_line",
    "validate_date_parts",
    "word_count",
    "write_jsonl",
]


def write_jsonl(dreams: list[dict[str, Any]], output_path: Path) -> None:
    content = "".join(
        json.dumps(dream, ensure_ascii=False) + "\n" for dream in dreams
    )
    write_text_atomic(output_path, content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse a dream journal text file into JSON Lines."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/mock_dream_journal.txt"),
        help="Path to the dream journal text file.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("data/dreams.jsonl"),
        help="Path where the JSON Lines output should be written.",
    )
    parser.add_argument(
        "--dream-separator-blank-lines",
        type=int,
        help=(
            "Number of consecutive blank lines that separates dreams. "
            "Defaults to auto-detection."
        ),
    )
    return parser


def main() -> None:
    argument_parser = build_parser()
    args = argument_parser.parse_args()
    journal_text = args.input.read_text(encoding="utf-8")
    try:
        dreams = JournalParser(
            dream_separator_blank_lines=args.dream_separator_blank_lines
        ).parse(journal_text)
    except ValueError as exc:
        argument_parser.error(str(exc))
    write_jsonl(dreams, args.output)
    print(f"Wrote {len(dreams)} dreams to {args.output}")


if __name__ == "__main__":
    main()
