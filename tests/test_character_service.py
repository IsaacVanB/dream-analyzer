from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from cli import build_character_lookup
from dream_analysis.characters import CharacterLookupService


class CharacterLookupServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "dream_id": "dream-1",
                "date_sort": "2024-02-03",
                "named_characters": [" Alice ", "alice", "Bob Smith"],
            },
            {
                "dream_id": "dream-2",
                "date_sort": "not-a-date",
                "named_characters": ["ALICE"],
            },
        ]

    def test_aggregation_deduplicates_each_dream_and_tracks_mentions(self) -> None:
        characters = CharacterLookupService().aggregate(self.records)

        self.assertEqual([item["name"] for item in characters], ["Alice", "Bob Smith"])
        self.assertEqual(
            characters[0]["mentions"],
            {
                "count": 2,
                "first_date": "2024-02-03",
                "last_date": "2024-02-03",
                "dream_ids": ["dream-1", "dream-2"],
            },
        )
        self.assertEqual(characters[1]["id"], "bob_smith")
        self.assertEqual(characters[0]["relationship"], "")

    def test_lookup_timestamp_is_injected_and_result_is_json_compatible(self) -> None:
        generated_at = datetime(2024, 3, 4, 12, 30, tzinfo=timezone.utc)

        lookup = CharacterLookupService().build_lookup(
            self.records,
            source="structured.jsonl",
            temporal_context=True,
            generated_at=generated_at,
        )

        self.assertEqual(lookup["generated_at"], "2024-03-04T12:30:00+00:00")
        self.assertIn("relationship_history", lookup["characters"][0])
        json.dumps(lookup)

    def test_cli_aggregation_wrapper_uses_the_service(self) -> None:
        expected = CharacterLookupService().aggregate(
            self.records,
            temporal_context=True,
        )

        actual = build_character_lookup.aggregate_characters(
            self.records,
            temporal_context=True,
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
