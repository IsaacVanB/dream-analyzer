from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from dream_analysis.models import DreamValidationError
from dream_analysis.repository import (
    CharacterDictionaryRepository,
    DreamNotFoundError,
    DreamRepository,
    StructuredDreamRepository,
    load_jsonl_objects,
)


class DreamRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "dreams.jsonl"

    def write_records(self, records: list[object]) -> None:
        self.path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_load_get_and_filter(self) -> None:
        self.write_records(
            [
                {
                    "dream_id": "early",
                    "date": "1/1/2024",
                    "date_sort": "2024-01-01",
                    "text": "First dream",
                },
                {
                    "dream_id": "late",
                    "date": "2/1/2024",
                    "date_sort": "2024-02-01",
                    "text": "Second dream",
                },
                {
                    "dream_id": "unknown",
                    "date": "0/0/2024",
                    "date_sort": None,
                    "text": "Unknown date",
                },
            ]
        )
        repository = DreamRepository(self.path)

        self.assertEqual(len(repository.all()), 3)
        self.assertEqual(repository.get("late").text, "Second dream")
        self.assertEqual(
            [dream.dream_id for dream in repository.between(date(2024, 1, 15))],
            ["late"],
        )

    def test_missing_id_has_a_specific_error(self) -> None:
        self.write_records([])
        with self.assertRaises(DreamNotFoundError):
            DreamRepository(self.path).get("missing")

    def test_duplicate_ids_are_rejected_with_line_context(self) -> None:
        record = {"dream_id": "same", "date": "1/1/2024", "text": "Dream"}
        self.write_records([record, record])

        with self.assertRaisesRegex(DreamValidationError, "line 2"):
            DreamRepository(self.path).all()

    def test_invalid_json_has_line_context(self) -> None:
        self.path.write_text('{"dream_id":\n', encoding="utf-8")

        with self.assertRaisesRegex(DreamValidationError, "line 1"):
            DreamRepository(self.path).all()

    def test_records_return_json_compatible_validated_shape(self) -> None:
        self.write_records(
            [
                {
                    "dream_id": "dream-1",
                    "date": "1/2/2024",
                    "date_sort": "2024-01-02",
                    "tags": ["house"],
                    "text": "Dream text",
                }
            ]
        )

        record = DreamRepository(self.path).records()[0]

        self.assertEqual(record["date_sort"], "2024-01-02")
        self.assertEqual(record["tags"], ["house"])
        self.assertEqual(record["word_count"], 2)

    def test_tagged_requires_every_exact_tag_case_insensitively(self) -> None:
        self.write_records(
            [
                {
                    "dream_id": "school-only",
                    "date": "1/1/2024",
                    "tags": ["school"],
                    "text": "School",
                },
                {
                    "dream_id": "both",
                    "date": "1/2/2024",
                    "tags": ["School", "lucid?"],
                    "text": "Both",
                },
                {
                    "dream_id": "different-punctuation",
                    "date": "1/3/2024",
                    "tags": ["school", "lucid"],
                    "text": "Not an exact match",
                },
            ]
        )

        repository = DreamRepository(self.path)
        school_matches = repository.tagged(["SCHOOL"])
        matches = repository.tagged(["SCHOOL", "lucid?"])

        self.assertEqual(
            [dream.dream_id for dream in school_matches],
            ["school-only", "both", "different-punctuation"],
        )
        self.assertEqual([dream.dream_id for dream in matches], ["both"])

    def test_line_aware_loader_preserves_blank_line_offsets_and_duplicates(self) -> None:
        record = {"dream_id": "same", "date": "1/1/2024", "text": "Dream"}
        self.path.write_text(
            json.dumps(record) + "\n\n" + json.dumps(record) + "\n",
            encoding="utf-8",
        )

        loaded = load_jsonl_objects(self.path)

        self.assertEqual([line_number for line_number, _ in loaded], [1, 3])
        self.assertEqual([item["dream_id"] for _, item in loaded], ["same", "same"])


class StructuredDreamRepositoryTests(unittest.TestCase):
    def test_loads_validated_character_feature_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dream_id": "dream-1",
                        "date_sort": "2025-01-02",
                        "named_characters": ["Maya"],
                        "summary": "A dream.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = StructuredDreamRepository(path).all()

        self.assertEqual(records[0]["named_characters"], ["Maya"])
        self.assertEqual(records[0]["summary"], "A dream.")

    def test_rejects_missing_character_field_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured.jsonl"
            path.write_text(
                json.dumps({"dream_id": "dream-1"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DreamValidationError, "named_characters"):
                StructuredDreamRepository(path).all()

            record = {"dream_id": "dream-1", "named_characters": []}
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DreamValidationError, "Duplicate"):
                StructuredDreamRepository(path).all()


class CharacterDictionaryRepositoryTests(unittest.TestCase):
    def test_loads_array_and_generated_envelope_formats(self) -> None:
        character = {
            "id": "maya",
            "name": "Maya",
            "aliases": ["May"],
            "relationship": "friend",
            "context": "A close friend.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "characters.json"
            for document in ([character], {"characters": [character]}):
                path.write_text(json.dumps(document), encoding="utf-8")
                records = CharacterDictionaryRepository(path).all()
                self.assertEqual(records[0]["name"], "Maya")

    def test_validates_ids_aliases_and_temporal_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "characters.json"
            duplicate = {"id": "same", "name": "Maya", "aliases": []}
            path.write_text(
                json.dumps([duplicate, duplicate]), encoding="utf-8"
            )
            with self.assertRaisesRegex(DreamValidationError, "Duplicate"):
                CharacterDictionaryRepository(path).all()

            invalid_history = {
                "id": "maya",
                "name": "Maya",
                "aliases": [],
                "relationship_history": [
                    {
                        "start_date": "2025-02-01",
                        "end_date": "2025-01-01",
                        "relationship": "friend",
                        "context": "",
                    }
                ],
            }
            path.write_text(json.dumps([invalid_history]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "start_date"):
                CharacterDictionaryRepository(path).all()


if __name__ == "__main__":
    unittest.main()
