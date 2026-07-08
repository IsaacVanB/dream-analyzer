#!/usr/bin/env python3
"""Parse a dream journal into one JSON object per dream."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\s*$")
TAG_LINE_RE = re.compile(r"^\s*(#[A-Za-z0-9_?+-]+(?:\s+#[A-Za-z0-9_?+-]+)*)\s*$")
WORD_RE = re.compile(r"\b[\w']+\b")


def normalize_year(year_text: str) -> int:
    """Convert two-digit journal years to four-digit years."""
    year = int(year_text)
    if len(year_text) == 2:
        return 1900 + year if year >= 70 else 2000 + year
    return year


def parse_tag_line(line: str) -> list[str] | None:
    """Return tags from a tag-only line, or None when the line is dream text."""
    if not TAG_LINE_RE.match(line):
        return None
    return [tag.removeprefix("#") for tag in line.split()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def build_dream(
    *,
    month: int,
    day: int,
    year: int,
    dream_lines: list[str],
) -> dict[str, Any]:
    tags: list[str] = []
    text_start = 0

    for index, line in enumerate(dream_lines):
        line_tags = parse_tag_line(line)
        if line_tags is None:
            text_start = index
            break
        tags.extend(line_tags)
    else:
        text_start = len(dream_lines)

    text = "\n".join(dream_lines[text_start:]).strip()

    return {
        "date": f"{month}/{day}/{year}",
        "year": year,
        "month": month,
        "tags": tags,
        "text": text,
        "word count": word_count(text),
    }


def parse_journal(journal_text: str) -> list[dict[str, Any]]:
    dreams: list[dict[str, Any]] = []
    current_date: tuple[int, int, int] | None = None
    current_dream_lines: list[str] = []

    def flush_dream() -> None:
        nonlocal current_dream_lines
        if current_date is None or not current_dream_lines:
            current_dream_lines = []
            return

        month, day, year = current_date
        dreams.append(
            build_dream(
                month=month,
                day=day,
                year=year,
                dream_lines=current_dream_lines,
            )
        )
        current_dream_lines = []

    for raw_line in journal_text.splitlines():
        line = raw_line.rstrip()
        date_match = DATE_RE.match(line)

        if date_match:
            flush_dream()
            month_text, day_text, year_text = date_match.groups()
            current_date = (
                int(month_text),
                int(day_text),
                normalize_year(year_text),
            )
            continue

        if current_date is None:
            if line.strip():
                raise ValueError(f"Found dream text before first date: {line!r}")
            continue

        if not line.strip():
            flush_dream()
            continue

        current_dream_lines.append(line)

    flush_dream()
    return dreams


def write_jsonl(dreams: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for dream in dreams:
            output_file.write(json.dumps(dream, ensure_ascii=False) + "\n")


def main() -> None:
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
    args = parser.parse_args()

    journal_text = args.input.read_text(encoding="utf-8")
    dreams = parse_journal(journal_text)
    write_jsonl(dreams, args.output)
    print(f"Wrote {len(dreams)} dreams to {args.output}")


if __name__ == "__main__":
    main()
