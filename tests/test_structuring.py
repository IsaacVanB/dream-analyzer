from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from dream_analysis.structuring import (
    DREAM_FEATURE_SCHEMA,
    SCHEMA_VERSION,
    DreamStructuringService,
    build_record,
    load_existing_records,
    select_pending_dreams,
    serialize_records,
    validate_features,
)


def valid_features() -> dict:
    features = {}
    for name, definition in DREAM_FEATURE_SCHEMA["properties"].items():
        if definition["type"] == "array":
            features[name] = []
        elif definition["type"] == "boolean":
            features[name] = False
        elif "enum" in definition:
            features[name] = definition["enum"][0]
        else:
            features[name] = "unclear"
    features["lucidity_level"] = "none"
    return features


class FakeGateway:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.response


class DreamStructuringTests(unittest.TestCase):
    def test_service_extracts_and_builds_a_versioned_record(self) -> None:
        gateway = FakeGateway(valid_features())
        service = DreamStructuringService(
            ollama_gateway=gateway,
            model="feature-model",
            num_ctx=4096,
        )
        dream = {
            "dream_id": "dream-1",
            "date": "1/2/2024",
            "date_sort": "2024-01-02",
            "tags": ["house"],
            "word_count": 5,
            "text": "I found another hidden room.",
        }

        record = service.structure(dream)

        self.assertEqual(record["dream_id"], "dream-1")
        self.assertEqual(record["schema_version"], SCHEMA_VERSION)
        self.assertEqual(record["model"], "feature-model")
        self.assertEqual(gateway.calls[0]["schema"], DREAM_FEATURE_SCHEMA)
        self.assertEqual(gateway.calls[0]["options"]["num_ctx"], 4096)

    def test_validation_normalizes_without_mutating_the_response(self) -> None:
        features = valid_features()
        features["themes"] = [" Hidden   Space ", "hidden space"]

        normalized = validate_features(features)

        self.assertEqual(normalized["themes"], ["hidden space"])
        self.assertEqual(features["themes"], [" Hidden   Space ", "hidden space"])

    def test_pending_selection_respects_version_and_overwrite(self) -> None:
        dreams = [{"dream_id": "current"}, {"dream_id": "stale"}]
        existing = {
            "current": {"schema_version": SCHEMA_VERSION},
            "stale": {"schema_version": SCHEMA_VERSION - 1},
        }

        self.assertEqual(
            select_pending_dreams(dreams, existing),
            [{"dream_id": "stale"}],
        )
        self.assertEqual(
            select_pending_dreams(dreams, existing, overwrite=True),
            dreams,
        )

    def test_existing_jsonl_loading_and_serialization_preserve_order(self) -> None:
        records = {
            "one": {"dream_id": "one", "schema_version": SCHEMA_VERSION},
            "two": {"dream_id": "two", "schema_version": SCHEMA_VERSION},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features.jsonl"
            path.write_text(serialize_records(records), encoding="utf-8")

            loaded = load_existing_records(path)

        self.assertEqual(list(loaded), ["one", "two"])
        self.assertEqual(loaded, records)

    def test_build_record_accepts_a_deterministic_timestamp(self) -> None:
        timestamp = datetime(2024, 2, 3, 4, 5, tzinfo=timezone.utc)

        record = build_record(
            {"dream_id": "one", "text": "Dream"},
            valid_features(),
            model="model",
            extracted_at=timestamp,
        )

        self.assertEqual(record["extracted_at"], "2024-02-03T04:05:00+00:00")
        json.dumps(record)


if __name__ == "__main__":
    unittest.main()
