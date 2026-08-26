from __future__ import annotations

import unittest
from datetime import date

from dream_analysis.models import Dream, DreamValidationError


class DreamTests(unittest.TestCase):
    def test_from_record_converts_existing_json_shape(self) -> None:
        dream = Dream.from_record(
            {
                "dream_id": "dream-1",
                "date": "1/2/2024",
                "year": 2024,
                "month": 1,
                "day": 2,
                "date_precision": "day",
                "date_sort": "2024-01-02",
                "tags": ["house", "recurring"],
                "text": "A room appeared behind the pantry.",
                "word_count": 6,
            }
        )

        self.assertEqual(dream.dream_id, "dream-1")
        self.assertEqual(dream.tags, ("house", "recurring"))
        self.assertEqual(dream.date_sort, date(2024, 1, 2))
        self.assertEqual(dream.to_record()["date_sort"], "2024-01-02")

    def test_word_count_is_derived_when_missing(self) -> None:
        dream = Dream.from_record(
            {
                "dream_id": "dream-1",
                "date": "unknown",
                "text": "one two three",
            }
        )
        self.assertEqual(dream.word_count, 3)

    def test_rejects_invalid_tags(self) -> None:
        with self.assertRaisesRegex(DreamValidationError, "tags"):
            Dream.from_record(
                {
                    "dream_id": "dream-1",
                    "date": "1/2/2024",
                    "text": "Text",
                    "tags": "house",
                }
            )


if __name__ == "__main__":
    unittest.main()

