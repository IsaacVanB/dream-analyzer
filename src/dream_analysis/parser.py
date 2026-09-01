"""Reusable parser for date-grouped plain-text dream journals."""

from __future__ import annotations

import re
from datetime import date
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


def validate_date_parts(
    *,
    month: int,
    day: int,
    year: int,
    year_text: str,
) -> None:
    """Reject impossible dates while allowing the journal's zero placeholders."""
    if month == 0 and day == 0 and year_text in {"0", "00", "0000"}:
        return
    if month == 0:
        if day != 0:
            raise ValueError("a day cannot be specified when the month is 0")
        date(year, 1, 1)
        return
    if day == 0:
        date(year, month, 1)
        return
    date(year, month, day)


def normalize_date_parts(
    *,
    month: int,
    day: int,
    year: int,
    year_text: str,
) -> dict[str, Any]:
    """Represent exact, month-only, year-only, and unknown journal dates."""
    validate_date_parts(
        month=month,
        day=day,
        year=year,
        year_text=year_text,
    )
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
        line = _clean_line(raw_line)
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
    """Build the established JSON-compatible representation of one dream."""
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


class JournalParser:
    """Parse a journal string into JSON-compatible dream records."""

    def __init__(self, *, dream_separator_blank_lines: int | None = None) -> None:
        if (
            dream_separator_blank_lines is not None
            and dream_separator_blank_lines < 1
        ):
            raise ValueError("dream_separator_blank_lines must be at least 1.")
        self.dream_separator_blank_lines = dream_separator_blank_lines

    def parse(self, journal_text: str) -> list[dict[str, Any]]:
        if not isinstance(journal_text, str):
            raise TypeError("journal_text must be a string")

        separator = self.dream_separator_blank_lines
        if separator is None:
            separator = detect_dream_separator_blank_lines(journal_text)

        dreams: list[dict[str, Any]] = []
        current_date: tuple[int, int, int, str] | None = None
        current_dream_lines: list[str] = []
        current_date_dream_index = 0
        pending_blank_lines = 0

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

        for line_number, raw_line in enumerate(journal_text.splitlines(), start=1):
            line = _clean_line(raw_line)
            date_match = DATE_RE.match(line)

            if date_match:
                month_text, day_text, year_text = date_match.groups()
                month = int(month_text)
                day = int(day_text)
                year = normalize_year(year_text)
                try:
                    validate_date_parts(
                        month=month,
                        day=day,
                        year=year,
                        year_text=year_text,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid journal date {line.strip()!r} on line "
                        f"{line_number}: {exc}"
                    ) from exc
                pending_blank_lines = 0
                flush_dream()
                current_date = (
                    month,
                    day,
                    year,
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

            if pending_blank_lines >= separator:
                flush_dream()
            elif pending_blank_lines:
                current_dream_lines.extend([""] * pending_blank_lines)
            pending_blank_lines = 0

            current_dream_lines.append(line)

        flush_dream()
        return dreams


def parse_journal(
    journal_text: str,
    *,
    dream_separator_blank_lines: int | None = None,
) -> list[dict[str, Any]]:
    """Compatibility-friendly functional entry point to ``JournalParser``."""
    return JournalParser(
        dream_separator_blank_lines=dream_separator_blank_lines
    ).parse(journal_text)


def _clean_line(raw_line: str) -> str:
    return raw_line.rstrip().removeprefix("\ufeff")
