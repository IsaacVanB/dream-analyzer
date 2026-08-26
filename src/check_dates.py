#!/usr/bin/env python3
"""Find suspicious year values in an ordered dream JSONL file. Helpful for finding human errors in dream journal dates."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from dream_analysis.config import Settings
from dream_analysis.dates import parse_date_value
from dream_analysis.repository import load_jsonl_objects


DREAMS_PATH = Settings().dreams_path
DATE_PARTS_RE = re.compile(r"^\s*(\d{1,4})/(\d{1,2})/(\d{1,4})\s*$")


@dataclass
class DateBlock:
    """One consecutive journal date, which may contain multiple dreams."""

    date_text: str
    year: int
    sort_date: date | None
    dream_ids: list[str] = field(default_factory=list)
    line_numbers: list[int] = field(default_factory=list)


def load_dreams(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load raw records with line numbers for human-error diagnostics."""
    return load_jsonl_objects(path)


def parse_sort_date(dream: dict[str, Any]) -> date | None:
    date_sort = dream.get("date_sort")
    if date_sort:
        return parse_date_value(date_sort)
    return parse_date_value(dream.get("date"))


def has_unknown_date_placeholder(dream: dict[str, Any]) -> bool:
    """Return True when zero represents an unknown date component."""
    date_precision = dream.get("date_precision")
    if date_precision in {"month", "year", "unknown"}:
        return True

    date_match = DATE_PARTS_RE.match(str(dream.get("date") or ""))
    if date_match is None:
        return False
    return any(int(part) == 0 for part in date_match.groups())


def find_duplicate_dream_ids(
    dreams: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Find repeated dream IDs before any date records are filtered."""
    occurrences: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for line_number, dream in dreams:
        raw_dream_id = dream.get("dream_id")
        if raw_dream_id is None:
            continue
        dream_id = str(raw_dream_id).strip()
        if dream_id:
            occurrences.setdefault(dream_id, []).append((line_number, dream))

    duplicates: list[dict[str, Any]] = []
    for dream_id, matching_dreams in occurrences.items():
        if len(matching_dreams) < 2:
            continue
        duplicates.append(
            {
                "dream_id": dream_id,
                "occurrence_count": len(matching_dreams),
                "line_numbers": [item[0] for item in matching_dreams],
                "dates": [str(item[1].get("date") or "unknown") for item in matching_dreams],
                "contains_placeholder_date": any(
                    has_unknown_date_placeholder(item[1])
                    for item in matching_dreams
                ),
            }
        )
    return duplicates


def build_date_blocks(
    dreams: list[tuple[int, dict[str, Any]]],
) -> tuple[list[DateBlock], int, int]:
    """Collapse consecutive dreams with the same date into date blocks."""
    blocks: list[DateBlock] = []
    unknown_year_count = 0
    placeholder_date_count = 0

    for line_number, dream in dreams:
        if has_unknown_date_placeholder(dream):
            placeholder_date_count += 1
            continue

        raw_year = dream.get("year")
        if raw_year is None:
            unknown_year_count += 1
            continue
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            unknown_year_count += 1
            continue

        date_text = str(dream.get("date") or "date unknown")
        sort_date = parse_sort_date(dream)
        dream_id = str(dream.get("dream_id") or f"line-{line_number}")
        same_as_previous = (
            blocks
            and blocks[-1].date_text == date_text
            and blocks[-1].year == year
            and blocks[-1].sort_date == sort_date
        )

        if same_as_previous:
            blocks[-1].dream_ids.append(dream_id)
            blocks[-1].line_numbers.append(line_number)
        else:
            blocks.append(
                DateBlock(
                    date_text=date_text,
                    year=year,
                    sort_date=sort_date,
                    dream_ids=[dream_id],
                    line_numbers=[line_number],
                )
            )

    return blocks, unknown_year_count, placeholder_date_count


def year_runs(blocks: list[DateBlock]) -> list[tuple[int, int, int]]:
    """Return (year, start index, end index) for consecutive year runs."""
    if not blocks:
        return []

    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(blocks)):
        if blocks[index].year != blocks[start].year:
            runs.append((blocks[start].year, start, index - 1))
            start = index
    runs.append((blocks[start].year, start, len(blocks) - 1))
    return runs


def find_suspicious_dates(
    blocks: list[DateBlock],
    *,
    max_isolated_dates: int = 3,
    max_year_jump: int = 1,
) -> list[dict[str, Any]]:
    """Return suspicious date blocks with reasons and nearby context."""
    reasons_by_index: dict[int, list[str]] = {}
    isolated_indices: set[int] = set()

    def add_reason(index: int, reason: str) -> None:
        reasons_by_index.setdefault(index, []).append(reason)

    first_index_by_date: dict[date | tuple[str, int], int] = {}
    for index, block in enumerate(blocks):
        date_key: date | tuple[str, int]
        if block.sort_date is not None:
            date_key = block.sort_date
        else:
            date_key = (block.date_text, block.year)

        first_index = first_index_by_date.get(date_key)
        if first_index is None:
            first_index_by_date[date_key] = index
        else:
            add_reason(
                index,
                f"date duplicates the earlier block at "
                f"{blocks[first_index].date_text}",
            )

    runs = year_runs(blocks)
    for run_index in range(1, len(runs) - 1):
        year, start, end = runs[run_index]
        previous_year = runs[run_index - 1][0]
        next_year = runs[run_index + 1][0]
        run_length = end - start + 1
        if (
            run_length <= max_isolated_dates
            and previous_year == next_year
            and year != previous_year
        ):
            previous_date = blocks[start - 1].sort_date
            next_date = blocks[end + 1].sort_date
            corrected_dates: list[date] = []
            for block in blocks[start : end + 1]:
                if block.sort_date is None:
                    corrected_dates = []
                    break
                try:
                    corrected_dates.append(
                        block.sort_date.replace(year=previous_year)
                    )
                except ValueError:
                    corrected_dates = []
                    break

            correction_restores_order = (
                previous_date is not None
                and next_date is not None
                and bool(corrected_dates)
                and previous_date <= corrected_dates[0]
                and corrected_dates == sorted(corrected_dates)
                and corrected_dates[-1] <= next_date
            )
            if not correction_restores_order:
                continue

            reason = (
                f"short run of {run_length} date(s) in {year} is surrounded "
                f"by dates in {previous_year}; changing the year to "
                f"{previous_year} restores chronological order"
            )
            for index in range(start, end + 1):
                isolated_indices.add(index)
                add_reason(index, reason)

    for index in range(1, len(blocks)):
        previous = blocks[index - 1]
        current = blocks[index]
        if (
            previous.sort_date is not None
            and current.sort_date is not None
            and current.sort_date < previous.sort_date
        ):
            finding_index = index - 1 if index - 1 in isolated_indices else index
            if finding_index == index - 1:
                reason = f"next date {current.date_text} goes backward"
            else:
                reason = f"date goes backward after {previous.date_text}"
            add_reason(
                finding_index,
                reason,
            )

        year_jump = abs(current.year - previous.year)
        if year_jump > max_year_jump:
            finding_index = index - 1 if index - 1 in isolated_indices else index
            add_reason(
                finding_index,
                f"year changes by {year_jump} after {previous.date_text}",
            )

    findings: list[dict[str, Any]] = []
    for index, reasons in sorted(reasons_by_index.items()):
        block = blocks[index]
        findings.append(
            {
                "date": block.date_text,
                "year": block.year,
                "dream_ids": block.dream_ids,
                "line_numbers": block.line_numbers,
                "previous_date": blocks[index - 1].date_text if index else None,
                "next_date": (
                    blocks[index + 1].date_text
                    if index + 1 < len(blocks)
                    else None
                ),
                "reasons": reasons,
            }
        )
    return findings


def check_years(
    dreams_path: Path = DREAMS_PATH,
    *,
    max_isolated_dates: int = 3,
    max_year_jump: int = 1,
) -> dict[str, Any]:
    dreams = load_dreams(dreams_path)
    duplicate_dream_ids = find_duplicate_dream_ids(dreams)
    blocks, unknown_year_count, placeholder_date_count = build_date_blocks(dreams)
    findings = find_suspicious_dates(
        blocks,
        max_isolated_dates=max_isolated_dates,
        max_year_jump=max_year_jump,
    )
    return {
        "dreams_path": str(dreams_path),
        "dream_count": len(dreams),
        "date_count": len(blocks),
        "unknown_year_dream_count": unknown_year_count,
        "placeholder_date_dream_count": placeholder_date_count,
        "duplicate_dream_id_count": len(duplicate_dream_ids),
        "duplicate_dream_ids": duplicate_dream_ids,
        "suspicious_date_count": len(findings),
        "findings": findings,
    }


def print_report(report: dict[str, Any]) -> None:
    findings = report["findings"]
    duplicate_dream_ids = report["duplicate_dream_ids"]
    print(
        f"Checked {report['dream_count']} dreams across "
        f"{report['date_count']} dated blocks."
    )
    if report["unknown_year_dream_count"]:
        print(
            f"Skipped {report['unknown_year_dream_count']} dream(s) with "
            "unknown or invalid years."
        )
    if report["placeholder_date_dream_count"]:
        print(
            f"Ignored {report['placeholder_date_dream_count']} dream(s) with "
            "zero placeholders in their dates."
        )

    if duplicate_dream_ids:
        print(f"Found {len(duplicate_dream_ids)} duplicate dream ID(s):")
        for duplicate in duplicate_dream_ids:
            placeholder_note = (
                " (includes placeholder date)"
                if duplicate["contains_placeholder_date"]
                else ""
            )
            print(f"\n- {duplicate['dream_id']}{placeholder_note}")
            print(
                "  JSONL lines: "
                + ", ".join(
                    str(line) for line in duplicate["line_numbers"]
                )
            )
            print("  Dates: " + ", ".join(duplicate["dates"]))

    if not findings:
        print("No suspicious dates found.")
        return

    print(f"Found {len(findings)} suspicious date(s):")
    for number, finding in enumerate(findings, start=1):
        print(f"\n{number}. {finding['date']} (year {finding['year']})")
        print(f"   Dream IDs: {', '.join(finding['dream_ids'])}")
        print(
            "   JSONL lines: "
            + ", ".join(str(line) for line in finding["line_numbers"])
        )
        for reason in finding["reasons"]:
            print(f"   Reason: {reason}")
        print(
            f"   Context: {finding['previous_date'] or '[start]'} -> "
            f"[{finding['date']}] -> {finding['next_date'] or '[end]'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find suspicious year values in ordered dream JSONL records."
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--max-isolated-dates",
        type=int,
        default=3,
        help="Largest surrounded year run to flag. Defaults to 3 dates.",
    )
    parser.add_argument(
        "--max-year-jump",
        type=int,
        default=1,
        help="Largest adjacent year change not flagged. Defaults to 1.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text report.",
    )
    args = parser.parse_args()

    if args.max_isolated_dates < 1:
        parser.error("--max-isolated-dates must be positive")
    if args.max_year_jump < 0:
        parser.error("--max-year-jump cannot be negative")

    report = check_years(
        args.dreams_path,
        max_isolated_dates=args.max_isolated_dates,
        max_year_jump=args.max_year_jump,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
