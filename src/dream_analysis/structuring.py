"""Reusable structured-feature extraction for dream journal entries."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.imports import dream_text_hash
from dream_analysis.prompts import STRUCTURING_SYSTEM_PROMPT, structuring_user_prompt
from dream_analysis.repository import DreamRepository


DEFAULT_STRUCTURING_MODEL = "gemma3:12b"
SCHEMA_VERSION = 10
LEVELS = ["none", "low", "moderate", "high"]
ARRAY_SENTINELS = frozenset(
    {"none", "n/a", "not applicable", "unknown", "unclear"}
)
GENERIC_CHARACTER_ROLES = frozenset(
    {
        "aunt",
        "boss",
        "boyfriend",
        "brother",
        "cashier",
        "classmate",
        "colleague",
        "cop",
        "cops",
        "co-worker",
        "coworker",
        "dad",
        "doctor",
        "father",
        "friend",
        "girlfriend",
        "grandfather",
        "grandma",
        "grandmother",
        "grandpa",
        "husband",
        "manager",
        "mom",
        "mother",
        "neighbor",
        "nurse",
        "partner",
        "police",
        "police officer",
        "professor",
        "roommate",
        "sister",
        "stranger",
        "teacher",
        "therapist",
        "uncle",
        "waiter",
        "waitress",
        "wife",
    }
)

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
        "retrieval_quality": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Query-independent estimate of how much specific, coherent, "
                "and distinctive evidence the report offers for retrieval."
            ),
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
        "retrieval_quality",
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
NUMBER_FIELDS = {"retrieval_quality"}

SYSTEM_PROMPT = STRUCTURING_SYSTEM_PROMPT


def _is_evident_generic_role(value: str) -> bool:
    normalized = value.casefold()
    normalized = re.sub(r"^(?:a|an|the|my)\s+", "", normalized)
    if normalized in GENERIC_CHARACTER_ROLES:
        return True
    if re.fullmatch(
        r"(?:unknown|unidentified)\s+"
        r"(?:boy|child|girl|guy|man|person|woman)",
        normalized,
    ):
        return True
    return bool(
        re.fullmatch(
            r"(?:boy|girl|guy|man|person|woman)\s+(?:at|from|in)\s+.+",
            normalized,
        )
    )


def _source_contains_phrase(source_text: str, phrase: str) -> bool:
    compact_source = re.sub(r"\s+", " ", source_text).casefold()
    compact_phrase = re.sub(r"\s+", " ", phrase.strip()).casefold()
    return bool(
        compact_phrase
        and re.search(
            rf"(?<!\w){re.escape(compact_phrase)}(?!\w)",
            compact_source,
        )
    )


def load_dreams(path: Path | str) -> list[dict[str, Any]]:
    """Load validated dream records for feature extraction."""
    return DreamRepository(path).records()


def select_dreams(
    dreams: Sequence[dict[str, Any]],
    dream_id: str | None,
) -> list[dict[str, Any]]:
    """Select all dreams or one exact ID, retaining input order."""
    if dream_id is None:
        return list(dreams)
    selected = [dream for dream in dreams if dream.get("dream_id") == dream_id]
    if not selected:
        raise ValueError(f"Dream ID not found: {dream_id}")
    return selected


def select_pending_dreams(
    dreams: Sequence[dict[str, Any]],
    existing_records: Mapping[str, Mapping[str, Any]],
    *,
    overwrite: bool = False,
    schema_version: int = SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Return dreams requiring extraction under the requested resume policy."""
    return [
        dream
        for dream in dreams
        if overwrite
        or dream.get("dream_id") not in existing_records
        or existing_records[str(dream.get("dream_id"))].get("schema_version")
        != schema_version
    ]


def build_extraction_messages(dream: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the grounded system and user messages for one dream."""
    text = dream.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Dream {dream.get('dream_id', '<unknown>')} has no valid text."
        )

    user_prompt = structuring_user_prompt(
        dream_id=dream.get("dream_id", "unknown"),
        dream_date=dream.get("date", "unknown"),
        tags=", ".join(str(tag) for tag in dream.get("tags", [])) or "none",
        dream_text=text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_features(
    features: Any,
    *,
    dream_text: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a structured Ollama response."""
    if not isinstance(features, dict):
        raise ValueError("Structured features must be a JSON object.")

    normalized_features = dict(features)
    required = set(DREAM_FEATURE_SCHEMA["required"])
    actual = set(normalized_features)
    if missing := sorted(required - actual):
        raise ValueError(f"Structured response is missing fields: {missing}")
    if unexpected := sorted(actual - required):
        raise ValueError(f"Structured response has unexpected fields: {unexpected}")

    for field in ARRAY_FIELDS:
        value = normalized_features[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field} must be an array of strings.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = re.sub(r"\s+", " ", item.strip())
            if field != "named_characters":
                normalized = normalized.lower()
            identity = normalized.casefold()
            if (
                normalized
                and identity not in ARRAY_SENTINELS
                and identity not in seen
            ):
                seen.add(identity)
                cleaned.append(normalized)
        normalized_features[field] = cleaned

    if dream_text is not None:
        characters = normalized_features["characters"]
        character_identities = {character.casefold() for character in characters}
        named_characters: list[str] = []
        for character in normalized_features["named_characters"]:
            if _is_evident_generic_role(character) and _source_contains_phrase(
                dream_text,
                character,
            ):
                role = character.lower()
                if role.casefold() not in character_identities:
                    character_identities.add(role.casefold())
                    characters.append(role)
            else:
                named_characters.append(character)
        normalized_features["named_characters"] = named_characters

    for field in BOOLEAN_FIELDS:
        if type(normalized_features[field]) is not bool:
            raise ValueError(f"{field} must be a boolean.")

    for field in NUMBER_FIELDS:
        value = normalized_features[field]
        definition = DREAM_FEATURE_SCHEMA["properties"][field]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not definition["minimum"] <= value <= definition["maximum"]
        ):
            raise ValueError(
                f"{field} must be a finite number between "
                f"{definition['minimum']} and {definition['maximum']}."
            )
        normalized_features[field] = float(value)

    for field, definition in DREAM_FEATURE_SCHEMA["properties"].items():
        if field in ARRAY_FIELDS or field in BOOLEAN_FIELDS or field in NUMBER_FIELDS:
            continue
        value = normalized_features[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        value = re.sub(r"\s+", " ", value.strip())
        if "enum" in definition and value not in definition["enum"]:
            raise ValueError(
                f"{field} must be one of {definition['enum']}; got {value!r}."
            )
        normalized_features[field] = value

    if normalized_features["lucidity"] != (
        normalized_features["lucidity_level"] == "lucid"
    ):
        raise ValueError(
            "lucidity must be true exactly when lucidity_level is 'lucid'."
        )
    return normalized_features


def extract_features(
    dream: Mapping[str, Any],
    *,
    model: str = DEFAULT_STRUCTURING_MODEL,
    num_ctx: int = 8192,
    gateway: OllamaGateway | None = None,
) -> dict[str, Any]:
    """Extract and validate structured features for one dream."""
    if num_ctx < 1:
        raise ValueError("num_ctx must be positive")
    messages = build_extraction_messages(dream)
    features = (gateway or OllamaGateway()).chat_json(
        schema=DREAM_FEATURE_SCHEMA,
        model=model,
        messages=messages,
        think=False,
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 1200,
        },
    )
    dream_text = dream.get("text")
    return validate_features(
        features,
        dream_text=dream_text if isinstance(dream_text, str) else None,
    )


def build_record(
    dream: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    model: str,
    extracted_at: datetime | None = None,
) -> dict[str, Any]:
    """Combine source metadata and extracted features into a versioned record."""
    timestamp = extracted_at or datetime.now().astimezone()
    return {
        "dream_id": dream.get("dream_id"),
        "date": dream.get("date"),
        "date_sort": dream.get("date_sort"),
        "journal_tags": dream.get("tags", []),
        "source_word_count": dream.get("word_count"),
        "structured_text_hash": dream_text_hash(str(dream.get("text", ""))),
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "extracted_at": timestamp.isoformat(timespec="seconds"),
        **dict(features),
    }


def load_existing_records(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load resumable structured JSONL records keyed by dream ID."""
    source_path = Path(path)
    if not source_path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source_path}: {exc}"
                ) from exc
            dream_id = record.get("dream_id") if isinstance(record, dict) else None
            if not isinstance(dream_id, str) or not dream_id:
                raise ValueError(
                    f"Record on line {line_number} of {source_path} has no dream_id."
                )
            records[dream_id] = record
    return records


def serialize_records(records: Mapping[str, Mapping[str, Any]]) -> str:
    """Serialize structured records as JSONL while preserving mapping order."""
    return "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records.values()
    )


class DreamStructuringService:
    """Extract validated, versioned feature records through Ollama."""

    def __init__(
        self,
        *,
        ollama_gateway: OllamaGateway,
        model: str = DEFAULT_STRUCTURING_MODEL,
        num_ctx: int = 8192,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if num_ctx < 1:
            raise ValueError("num_ctx must be positive")
        self.ollama = ollama_gateway
        self.model = model.strip()
        self.num_ctx = num_ctx

    def extract_features(self, dream: Mapping[str, Any]) -> dict[str, Any]:
        return extract_features(
            dream,
            model=self.model,
            num_ctx=self.num_ctx,
            gateway=self.ollama,
        )

    def structure(self, dream: Mapping[str, Any]) -> dict[str, Any]:
        features = self.extract_features(dream)
        return build_record(dream, features, model=self.model)
