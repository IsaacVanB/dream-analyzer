#!/usr/bin/env python3
"""Extract grounded, structured features from dream journal entries."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import ollama


DREAMS_PATH = Path("data/dreams.jsonl")
OUTPUT_PATH = Path("outputs/structured_dreams/dream_features.jsonl")
MODEL = "gemma3:12b"
SCHEMA_VERSION = 1

LEVELS = ["none", "low", "moderate", "high"]

DREAM_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "relationships": {"type": "array", "items": {"type": "string"}},
        "emotions": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "sensory_details": {"type": "array", "items": {"type": "string"}},
        "dream_mechanics": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "lucidity": {"type": "boolean"},
        "lucidity_level": {
            "type": "string",
            "enum": ["none", "questioning", "lucid"],
        },
        "violence": {"type": "string", "enum": LEVELS},
        "sexual_content": {"type": "string", "enum": LEVELS},
        "social_conflict": {"type": "boolean"},
        "threat_level": {"type": "string", "enum": LEVELS},
        "agency": {
            "type": "string",
            "enum": ["low", "moderate", "high", "unclear"],
        },
        "bizarreness": {
            "type": "string",
            "enum": ["ordinary", "low", "moderate", "high"],
        },
        "perspective": {
            "type": "string",
            "enum": ["first_person", "third_person", "mixed", "unclear"],
        },
        "ending": {
            "type": "string",
            "enum": ["resolved", "unresolved", "interrupted", "unclear"],
        },
        "memory_quality": {
            "type": "string",
            "enum": ["fragmentary", "partial", "detailed"],
        },
        "summary": {"type": "string"},
    },
    "required": [
        "setting",
        "characters",
        "relationships",
        "emotions",
        "themes",
        "objects",
        "actions",
        "sensory_details",
        "dream_mechanics",
        "tone",
        "lucidity",
        "lucidity_level",
        "violence",
        "sexual_content",
        "social_conflict",
        "threat_level",
        "agency",
        "bizarreness",
        "perspective",
        "ending",
        "memory_quality",
        "summary",
    ],
    "additionalProperties": False,
}

ARRAY_FIELDS = {
    name
    for name, definition in DREAM_FEATURE_SCHEMA["properties"].items()
    if definition["type"] == "array"
}
BOOLEAN_FIELDS = {"lucidity", "social_conflict"}


def load_dreams(path: Path) -> list[dict[str, Any]]:
    dreams: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                dream = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(dream, dict):
                raise ValueError(f"Dream on line {line_number} of {path} is not an object.")
            dream_id = dream.get("dream_id")
            if not isinstance(dream_id, str) or not dream_id:
                raise ValueError(f"Dream on line {line_number} of {path} has no dream_id.")
            if dream_id in seen_ids:
                raise ValueError(f"Duplicate dream_id {dream_id!r} in {path}.")
            seen_ids.add(dream_id)
            dreams.append(dream)
    return dreams


def select_dreams(
    dreams: list[dict[str, Any]],
    dream_id: str | None,
) -> list[dict[str, Any]]:
    if dream_id is None:
        return dreams
    selected = [dream for dream in dreams if dream.get("dream_id") == dream_id]
    if not selected:
        raise ValueError(f"Dream ID not found: {dream_id}")
    return selected


def extract_features(
    dream: dict[str, Any],
    *,
    model: str = MODEL,
    num_ctx: int = 8192,
) -> dict[str, Any]:
    text = dream.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Dream {dream.get('dream_id', '<unknown>')} has no valid text.")

    system_prompt = (
        "Extract structured, descriptive features from a dream report. Use only "
        "information supported by the supplied text. Do not apply dream dictionaries, "
        "diagnose the dreamer, infer real-world events, or assign symbolic meanings. "
        "Use concise lowercase phrases in arrays, remove duplicates, and use empty "
        "arrays when a category has no evidence. Characters should use roles or "
        "relationships when possible rather than inventing identities. Themes should "
        "describe observable narrative patterns such as being chased, failing a task, "
        "or discovering a hidden space—not speculative psychological interpretations."
    )
    user_prompt = f"""
DREAM_ID: {dream.get('dream_id', 'unknown')}
DATE: {dream.get('date', 'unknown')}
JOURNAL_TAGS: {', '.join(str(tag) for tag in dream.get('tags', [])) or 'none'}

DREAM TEXT:
{text}

Extraction guidance:
- `setting`: distinct physical or social locations.
- `characters`: people, animals, creatures, or personified entities.
- `relationships`: explicit social roles or interactions.
- `emotions`: stated or strongly evidenced feelings only.
- `themes`: concrete recurring situations, goals, conflicts, or transformations.
- `objects`: salient physical objects, not every incidental noun.
- `actions`: major actions that move the dream forward.
- `sensory_details`: notable colors, sounds, textures, bodily sensations, or weather.
- `dream_mechanics`: impossible transformations, false awakenings, unstable spaces,
  time discontinuity, altered physics, or other explicitly dreamlike mechanics.
- `tone`: one concise dominant tone, or `unclear`.
- `lucidity`: true only when the dreamer knows they are dreaming.
- `violence`, `sexual_content`, and `threat_level`: none, low, moderate, or high.
- `social_conflict`: true when there is interpersonal opposition, rejection,
  coercion, betrayal, humiliation, or argument.
- `agency`: how effectively the dreamer makes consequential choices.
- `bizarreness`: degree of departure from ordinary reality.
- `perspective`: first_person, third_person, mixed, or unclear.
- `ending`: resolved, unresolved, interrupted, or unclear.
- `memory_quality`: fragmentary, partial, or detailed based on the report itself.
- `summary`: one or two factual sentences covering the central events.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=DREAM_FEATURE_SCHEMA,
        think=False,
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 1200,
        },
    )
    content = response["message"]["content"]
    if not content:
        raise ValueError("The model returned an empty structured response.")
    try:
        features = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON: {exc}") from exc
    return validate_features(features)


def validate_features(features: Any) -> dict[str, Any]:
    if not isinstance(features, dict):
        raise ValueError("Structured features must be a JSON object.")

    required = set(DREAM_FEATURE_SCHEMA["required"])
    actual = set(features)
    if missing := sorted(required - actual):
        raise ValueError(f"Structured response is missing fields: {missing}")
    if unexpected := sorted(actual - required):
        raise ValueError(f"Structured response has unexpected fields: {unexpected}")

    for field in ARRAY_FIELDS:
        value = features[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be an array of strings.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = re.sub(r"\s+", " ", item.strip().lower())
            if normalized and normalized not in seen:
                seen.add(normalized)
                cleaned.append(normalized)
        features[field] = cleaned

    for field in BOOLEAN_FIELDS:
        if type(features[field]) is not bool:
            raise ValueError(f"{field} must be a boolean.")

    for field, definition in DREAM_FEATURE_SCHEMA["properties"].items():
        if field in ARRAY_FIELDS or field in BOOLEAN_FIELDS:
            continue
        value = features[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        value = re.sub(r"\s+", " ", value.strip())
        if "enum" in definition and value not in definition["enum"]:
            raise ValueError(f"{field} must be one of {definition['enum']}; got {value!r}.")
        features[field] = value

    if features["lucidity"] != (features["lucidity_level"] == "lucid"):
        raise ValueError("lucidity must be true exactly when lucidity_level is 'lucid'.")
    return features


def build_record(
    dream: dict[str, Any],
    features: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "dream_id": dream.get("dream_id"),
        "date": dream.get("date"),
        "date_sort": dream.get("date_sort"),
        "journal_tags": dream.get("tags", []),
        "source_word_count": dream.get("word_count"),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **features,
    }


def load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            dream_id = record.get("dream_id") if isinstance(record, dict) else None
            if not isinstance(dream_id, str) or not dream_id:
                raise ValueError(f"Record on line {line_number} of {path} has no dream_id.")
            records[dream_id] = record
    return records


def save_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for record in records.values():
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


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
    pending = [
        dream
        for dream in dreams
        if args.overwrite or dream.get("dream_id") not in records
    ]
    skipped = len(dreams) - len(pending)
    if not pending:
        print(f"Nothing to process; skipped {skipped} existing dream(s).")
        print(f"Output: {args.output}")
        return

    failures = 0
    started = perf_counter()
    for index, dream in enumerate(pending, start=1):
        dream_id = str(dream.get("dream_id", "<unknown>"))
        print(f"Structuring {index}/{len(pending)}: {dream_id}")
        try:
            features = extract_features(
                dream,
                model=args.model,
                num_ctx=args.num_ctx,
            )
            records[dream_id] = build_record(dream, features, model=args.model)
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
