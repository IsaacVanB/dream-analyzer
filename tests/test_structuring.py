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
    build_extraction_messages,
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
        elif definition["type"] == "number":
            features[name] = 0.5
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


class SequenceGateway:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.responses.pop(0)


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
        self.assertRegex(record["structured_text_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(gateway.calls[0]["schema"], DREAM_FEATURE_SCHEMA)
        self.assertEqual(gateway.calls[0]["options"]["num_ctx"], 4096)
        self.assertEqual(len(gateway.calls), 1)

    def test_service_retries_a_validation_error_with_focused_feedback(self) -> None:
        invalid = valid_features()
        invalid["lucidity"] = True
        invalid["lucidity_level"] = "none"
        corrected = valid_features()
        corrected["lucidity"] = True
        corrected["lucidity_level"] = "lucid"
        gateway = SequenceGateway([invalid, corrected])
        service = DreamStructuringService(
            ollama_gateway=gateway,
            model="feature-model",
        )

        record = service.structure(
            {
                "dream_id": "lucid",
                "text": "I realized that I was dreaming.",
            }
        )

        self.assertTrue(record["lucidity"])
        self.assertEqual(record["lucidity_level"], "lucid")
        self.assertEqual(len(gateway.calls), 2)
        retry_messages = gateway.calls[1]["messages"]
        self.assertEqual(retry_messages[-2]["role"], "assistant")
        self.assertIn('"lucidity": true', retry_messages[-2]["content"])
        self.assertIn(
            "lucidity must be true exactly when lucidity_level is 'lucid'",
            retry_messages[-1]["content"],
        )
        self.assertIn("complete corrected JSON object", retry_messages[-1]["content"])

    def test_service_fails_after_one_invalid_retry(self) -> None:
        invalid = valid_features()
        invalid["lucidity"] = True
        invalid["lucidity_level"] = "none"
        gateway = SequenceGateway([invalid, invalid])
        service = DreamStructuringService(
            ollama_gateway=gateway,
            model="feature-model",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Structured response failed validation after 2 attempts",
        ):
            service.structure(
                {
                    "dream_id": "lucid",
                    "text": "I realized that I was dreaming.",
                }
            )

        self.assertEqual(len(gateway.calls), 2)

    def test_lucid_tag_triggers_retry_until_lucidity_is_true(self) -> None:
        unmarked = valid_features()
        corrected = valid_features()
        corrected["lucidity"] = True
        corrected["lucidity_level"] = "lucid"
        gateway = SequenceGateway([unmarked, corrected])
        service = DreamStructuringService(
            ollama_gateway=gateway,
            model="feature-model",
        )

        record = service.structure(
            {
                "dream_id": "tagged-lucid",
                "tags": ["dream", "#LuCiD"],
                "text": "I flew over the town.",
            }
        )

        self.assertTrue(record["lucidity"])
        self.assertEqual(record["lucidity_level"], "lucid")
        self.assertEqual(len(gateway.calls), 2)
        self.assertIn(
            "dreams tagged 'lucid' must have lucidity true",
            gateway.calls[1]["messages"][-1]["content"],
        )

    def test_absent_lucid_tag_does_not_constrain_lucidity(self) -> None:
        non_lucid = validate_features(
            valid_features(),
            dream_tags=["flying"],
        )
        lucid = valid_features()
        lucid["lucidity"] = True
        lucid["lucidity_level"] = "lucid"

        validated_lucid = validate_features(lucid, dream_tags=[])

        self.assertFalse(non_lucid["lucidity"])
        self.assertTrue(validated_lucid["lucidity"])

    def test_validation_normalizes_without_mutating_the_response(self) -> None:
        features = valid_features()
        features["themes"] = [" Hidden   Space ", "hidden space"]

        normalized = validate_features(features)

        self.assertEqual(normalized["themes"], ["hidden space"])
        self.assertEqual(features["themes"], [" Hidden   Space ", "hidden space"])

    def test_validation_removes_array_sentinels(self) -> None:
        features = valid_features()
        features["themes"] = ["none", " Hidden room ", "N/A", "hidden room"]
        features["named_characters"] = ["Unknown", "Maya", "UNCLEAR"]

        normalized = validate_features(features)

        self.assertEqual(normalized["themes"], ["hidden room"])
        self.assertEqual(normalized["named_characters"], ["Maya"])

    def test_validation_moves_source_backed_generic_roles_to_characters(self) -> None:
        features = valid_features()
        features["characters"] = ["teacher"]
        features["named_characters"] = [
            "Cop",
            "Maya",
            "Guy From Work",
            "Coworker",
        ]

        normalized = validate_features(
            features,
            dream_text="Maya spoke to the cop and the guy from work.",
        )

        self.assertEqual(
            normalized["characters"],
            ["teacher", "cop", "guy from work"],
        )
        self.assertEqual(normalized["named_characters"], ["Maya", "Coworker"])
        self.assertEqual(
            features["named_characters"],
            ["Cop", "Maya", "Guy From Work", "Coworker"],
        )

    def test_service_uses_dream_text_when_reclassifying_generic_roles(self) -> None:
        features = valid_features()
        features["named_characters"] = ["Cop"]
        service = DreamStructuringService(
            ollama_gateway=FakeGateway(features),
            model="feature-model",
        )

        record = service.structure(
            {
                "dream_id": "role",
                "text": "A cop waved to me.",
            }
        )

        self.assertEqual(record["characters"], ["cop"])
        self.assertEqual(record["named_characters"], [])

    def test_extraction_prompt_requires_conservative_grounding(self) -> None:
        messages = build_extraction_messages(
            {
                "dream_id": "names-only",
                "date": "unknown",
                "tags": [],
                "text": "Ksenia\nGabby Rachelle",
            }
        )
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("never `named_characters`", prompt)
        self.assertIn("contains names but", prompt)
        self.assertIn("do not infer any action, setting, emotion", prompt)
        self.assertIn("Do not omit, euphemize, sanitize", prompt)
        self.assertIn("Never invent details", prompt)

    def test_extraction_prompt_defines_content_intensity_levels(self) -> None:
        messages = build_extraction_messages(
            {
                "dream_id": "levels",
                "date": "unknown",
                "tags": [],
                "text": "Someone followed me, but never attacked me.",
            }
        )
        prompt = "\n".join(message["content"] for message in messages)

        self.assertIn("`violence`: none for no physical aggression", prompt)
        self.assertIn("A threat by itself does not count as violence", prompt)
        self.assertIn("`sexual_content`: none for no sexual behavior", prompt)
        self.assertIn("`threat_level`: none for no credible danger", prompt)
        self.assertIn("Threat can be high even when no", prompt)
        self.assertIn("absence of that tag does not imply", prompt)

    def test_validation_normalizes_retrieval_quality_to_float(self) -> None:
        features = valid_features()
        features["retrieval_quality"] = 1

        normalized = validate_features(features)

        self.assertEqual(normalized["retrieval_quality"], 1.0)
        self.assertIs(type(normalized["retrieval_quality"]), float)

    def test_validation_rejects_invalid_retrieval_quality_values(self) -> None:
        for invalid_score in (-0.1, 1.1, True, "0.5", float("nan")):
            with self.subTest(invalid_score=invalid_score):
                features = valid_features()
                features["retrieval_quality"] = invalid_score

                with self.assertRaisesRegex(
                    ValueError,
                    "retrieval_quality must be a finite number between 0.0 and 1.0",
                ):
                    validate_features(features)

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
