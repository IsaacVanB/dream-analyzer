#!/usr/bin/env python3
"""Extract grounded, structured features from dream journal entries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from dream_analysis.config import Settings
from dream_analysis.artifacts import write_text_atomic
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.repository import DreamRepository


DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_PATH = DEFAULT_SETTINGS.output_path / "structured_dreams/dream_features.jsonl"
MODEL = "gemma3:12b"
SCHEMA_VERSION = 4

LEVELS = ["none", "low", "moderate", "high"]

DREAM_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "array", "items": {"type": "string"}},
        "characters": {"type": "array", "items": {"type": "string"}},
        "named_characters": {"type": "array", "items": {"type": "string"}},
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
        "social_conflict": {"type": "string", "enum": LEVELS},
        "threat_level": {"type": "string", "enum": LEVELS},
        "agency": {
            "type": "string",
            "enum": ["low", "moderate", "high", "unclear"],
        },
        "bizarreness": {"type": "string", "enum": LEVELS},
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
        "named_characters",
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
BOOLEAN_FIELDS = {"lucidity"}


def load_dreams(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper returning validated record dictionaries."""
    return DreamRepository(path).records()


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
    gateway: OllamaGateway | None = None,
) -> dict[str, Any]:
    text = dream.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Dream {dream.get('dream_id', '<unknown>')} has no valid text.")

    system_prompt = (
        "Extract structured, descriptive features from a dream report. Use only "
        "information supported by the supplied text. Do not apply dream dictionaries, "
        "diagnose the dreamer, infer real-world events, or assign symbolic meanings. "
        "Use concise lowercase phrases in arrays except named_characters, preserve "
        "the capitalization of names, remove duplicates, and use empty arrays when a "
        "category has no evidence. Do not invent identities. Themes should "
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
- `characters`: unnamed characters expressed as roles, such as mother, unknown
  man, teacher, dog, or former classmate. Do not duplicate named characters here.
- `named_characters`: only characters explicitly called by a proper name in the
  report, preserving how the name is capitalized. Include named real people,
  public figures, fictional characters, animals, or other personified entities.
  Do not infer a name from a role or description.
- `emotions`: stated or strongly evidenced feelings only.
- `themes`: concrete recurring situations, goals, conflicts, or transformations.
- `objects`: salient physical objects, not every incidental noun.
- `actions`: major actions that move the dream forward.
- `sensory_details`: notable colors, sounds, textures, bodily sensations, or weather.
- `dream_mechanics`: impossible transformations, false awakenings, unstable spaces,
  time discontinuity, altered physics, or other explicitly dreamlike mechanics.
- `tone`: one concise dominant tone, or `unclear`.
- `lucidity`: true only when the dreamer knows they are dreaming.
- `violence`, `sexual_content`, `threat_level`, `social_conflict`, and
  `bizarreness` use none, low, moderate, or high.
- `social_conflict`: none for no interpersonal friction; low for mild tension,
  awkwardness, or disagreement; moderate for sustained hostility, rejection,
  coercion, humiliation, or betrayal; high for severe domination, interpersonal
  danger, or violent conflict.
- `agency`: how effectively the dreamer makes consequential choices.
- `bizarreness`: none for ordinary and physically plausible events; low for a
  small number of odd but coherent details; moderate for clear impossibilities,
  transformations, or unstable space/time; high when radical impossibility or
  incoherence pervades the dream.
- `perspective`: first_person, third_person, mixed, or unclear.
- `ending`: resolved, unresolved, interrupted, or unclear.
- `memory_quality`: fragmentary, partial, or detailed based on the report itself.
- `summary`: one or two factual sentences covering the central events.
"""

    features = (gateway or OllamaGateway()).chat_json(
        schema=DREAM_FEATURE_SCHEMA,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 1200,
        },
    )
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
            normalized = re.sub(r"\s+", " ", item.strip())
            if field != "named_characters":
                normalized = normalized.lower()
            identity = normalized.casefold()
            if normalized and identity not in seen:
                seen.add(identity)
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
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records.values()
    )
    write_text_atomic(path, content)


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
        if args.overwrite
        or dream.get("dream_id") not in records
        or records[str(dream.get("dream_id"))].get("schema_version")
        != SCHEMA_VERSION
    ]
    skipped = len(dreams) - len(pending)
    if not pending:
        print(f"Nothing to process; skipped {skipped} existing dream(s).")
        print(f"Output: {args.output}")
        return

    failures = 0
    started = perf_counter()
    gateway = OllamaGateway()
    for index, dream in enumerate(pending, start=1):
        dream_id = str(dream.get("dream_id", "<unknown>"))
        print(f"Structuring {index}/{len(pending)}: {dream_id}")
        try:
            features = extract_features(
                dream,
                model=args.model,
                num_ctx=args.num_ctx,
                gateway=gateway,
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
