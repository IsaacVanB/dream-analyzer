#!/usr/bin/env python3
"""Parse a dream journal into one JSON object per dream."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dream_analysis.artifacts import write_text_atomic


DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\s*$")
TAG_LINE_RE = re.compile(r"^\s*(#[A-Za-z0-9_?+-]+(?:\s+#[A-Za-z0-9_?+-]+)*)\s*$")
WORD_RE = re.compile(r"\b[\w']+\b")


def normalize_year(year_text: str) -> int:
    """Convert two-digit journal years to four-digit years."""
    year = int(year_text)
    if len(year_text) == 2:
        return 1900 + year if year >= 70 else 2000 + year
    return year


def normalize_date_parts(
    *,
    month: int,
    day: int,
    year: int,
    year_text: str,
) -> dict[str, Any]:
    parsed_year: int | None = year
    parsed_month: int | None = month
    parsed_day: int | None = day
    date_sort: str | None = None

    if month == 0 and day == 0 and year_text in {"0", "00", "0000"}:
        parsed_year = None
        parsed_month = None
        parsed_day = None
        date_precision = "unknown"
    elif month == 0 and day == 0:
        parsed_month = None
        parsed_day = None
        date_precision = "year"
        date_sort = f"{year:04d}-01-01"
    elif day == 0:
        parsed_day = None
        date_precision = "month"
        date_sort = f"{year:04d}-{month:02d}-01"
    else:
        date_precision = "day"
        date_sort = f"{year:04d}-{month:02d}-{day:02d}"

    return {
        "year": parsed_year,
        "month": parsed_month,
        "day": parsed_day,
        "date_precision": date_precision,
        "date_sort": date_sort,
    }


def parse_tag_line(line: str) -> list[str] | None:
    """Return tags from a tag-only line, or None when the line is dream text."""
    if not TAG_LINE_RE.match(line):
        return None
    return [tag.removeprefix("#") for tag in line.split()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def detect_dream_separator_blank_lines(journal_text: str) -> int:
    """Infer whether dreams are separated by one or two blank lines."""
    max_blank_run = 0
    current_blank_run = 0

    for raw_line in journal_text.splitlines():
        line = raw_line.rstrip().removeprefix("\ufeff")
        if not line.strip():
            current_blank_run += 1
            max_blank_run = max(max_blank_run, current_blank_run)
        else:
            current_blank_run = 0

    return 2 if max_blank_run >= 2 else 1


def build_dream(
    *,
    month: int,
    day: int,
    year: int,
    year_text: str,
    dream_index: int,
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
    date_parts = normalize_date_parts(
        month=month,
        day=day,
        year=year,
        year_text=year_text,
    )

    return {
        "dream_id": f"dream-{year}-{month}-{day}-{dream_index}",
        "date": f"{month}/{day}/{year}",
        **date_parts,
        "tags": tags,
        "text": text,
        "word_count": word_count(text),
    }


def parse_journal(
    journal_text: str,
    *,
    dream_separator_blank_lines: int | None = None,
) -> list[dict[str, Any]]:
    dreams: list[dict[str, Any]] = []
    current_date: tuple[int, int, int, str] | None = None
    current_dream_lines: list[str] = []
    current_date_dream_index = 0
    pending_blank_lines = 0

    if dream_separator_blank_lines is None:
        dream_separator_blank_lines = detect_dream_separator_blank_lines(journal_text)
    if dream_separator_blank_lines < 1:
        raise ValueError("dream_separator_blank_lines must be at least 1.")

    def flush_dream() -> None:
        nonlocal current_date_dream_index, current_dream_lines
        if current_date is None or not current_dream_lines:
            current_dream_lines = []
            return

        month, day, year, year_text = current_date
        dreams.append(
            build_dream(
                month=month,
                day=day,
                year=year,
                year_text=year_text,
                dream_index=current_date_dream_index,
                dream_lines=current_dream_lines,
            )
        )
        current_date_dream_index += 1
        current_dream_lines = []

    for raw_line in journal_text.splitlines():
        line = raw_line.rstrip().removeprefix("\ufeff")
        date_match = DATE_RE.match(line)

        if date_match:
            pending_blank_lines = 0
            flush_dream()
            month_text, day_text, year_text = date_match.groups()
            current_date = (
                int(month_text),
                int(day_text),
                normalize_year(year_text),
                year_text,
            )
            current_date_dream_index = 0
            continue

        if current_date is None:
            if line.strip():
                raise ValueError(f"Found dream text before first date: {line!r}")
            continue

        if not line.strip():
            pending_blank_lines += 1
            continue

        if pending_blank_lines >= dream_separator_blank_lines:
            flush_dream()
        elif pending_blank_lines:
            current_dream_lines.extend([""] * pending_blank_lines)
        pending_blank_lines = 0

        current_dream_lines.append(line)

    flush_dream()
    return dreams


def write_jsonl(dreams: list[dict[str, Any]], output_path: Path) -> None:
    content = "".join(
        json.dumps(dream, ensure_ascii=False) + "\n" for dream in dreams
    )
    write_text_atomic(output_path, content)


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
    parser.add_argument(
        "--dream-separator-blank-lines",
        type=int,
        help=(
            "Number of consecutive blank lines that separates dreams. "
            "Defaults to auto-detection."
        ),
    )
    args = parser.parse_args()

    journal_text = args.input.read_text(encoding="utf-8")
    dreams = parse_journal(
        journal_text,
        dream_separator_blank_lines=args.dream_separator_blank_lines,
    )
    write_jsonl(dreams, args.output)
    print(f"Wrote {len(dreams)} dreams to {args.output}")


if __name__ == "__main__":
    main()
