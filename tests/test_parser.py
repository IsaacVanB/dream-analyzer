from __future__ import annotations

import unittest

from cli import parse_dreams
from dream_analysis.parser import (
    JournalParser,
    detect_dream_separator_blank_lines,
    normalize_year,
    parse_journal,
)


class JournalParserTests(unittest.TestCase):
    def test_auto_detected_two_blank_separator_preserves_single_blank_lines(self) -> None:
        journal = (
            "\ufeff1/2/24\n"
            "#house #odd\n"
            "First line\n"
            "\n"
            "Second paragraph\n"
            "\n"
            "\n"
            "#school\n"
            "Second dream\n"
        )

        dreams = JournalParser().parse(journal)

        self.assertEqual(detect_dream_separator_blank_lines(journal), 2)
        self.assertEqual(len(dreams), 2)
        self.assertEqual(dreams[0]["dream_id"], "dream-2024-1-2-0")
        self.assertEqual(dreams[0]["tags"], ["house", "odd"])
        self.assertEqual(dreams[0]["text"], "First line\n\nSecond paragraph")
        self.assertEqual(dreams[0]["word_count"], 4)
        self.assertEqual(dreams[1]["dream_id"], "dream-2024-1-2-1")
        self.assertEqual(dreams[1]["tags"], ["school"])

    def test_explicit_single_blank_separator_splits_dreams(self) -> None:
        journal = "1/2/2024\nFirst dream\n\nSecond dream\n"

        dreams = JournalParser(dream_separator_blank_lines=1).parse(journal)

        self.assertEqual([dream["text"] for dream in dreams], ["First dream", "Second dream"])

    def test_partial_and_unknown_dates_keep_existing_record_shape(self) -> None:
        dreams = parse_journal(
            "0/0/00\nUnknown date\n"
            "0/0/2024\nYear only\n"
            "3/0/24\nMonth only\n"
        )

        self.assertEqual(
            {
                key: dreams[0][key]
                for key in ("year", "month", "day", "date_precision", "date_sort")
            },
            {
                "year": None,
                "month": None,
                "day": None,
                "date_precision": "unknown",
                "date_sort": None,
            },
        )
        self.assertEqual(dreams[1]["date_precision"], "year")
        self.assertEqual(dreams[1]["date_sort"], "2024-01-01")
        self.assertEqual(dreams[2]["date_precision"], "month")
        self.assertEqual(dreams[2]["date_sort"], "2024-03-01")

    def test_tags_are_only_consumed_at_the_start_of_a_dream(self) -> None:
        dream = parse_journal(
            "1/2/24\n#first\nText begins\n#part-of-text\n"
        )[0]

        self.assertEqual(dream["tags"], ["first"])
        self.assertEqual(dream["text"], "Text begins\n#part-of-text")

    def test_invalid_input_has_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "before first date"):
            JournalParser().parse("A dream without a date")
        with self.assertRaisesRegex(ValueError, "at least 1"):
            JournalParser(dream_separator_blank_lines=0)
        with self.assertRaisesRegex(TypeError, "journal_text"):
            JournalParser().parse(None)  # type: ignore[arg-type]

    def test_year_pivot_and_cli_compatibility_entry_point(self) -> None:
        journal = "1/2/69\nFuture\n1/2/70\nPast\n"

        service_records = JournalParser().parse(journal)
        cli_records = parse_dreams.parse_journal(journal)

        self.assertEqual(normalize_year("69"), 2069)
        self.assertEqual(normalize_year("70"), 1970)
        self.assertEqual(cli_records, service_records)


if __name__ == "__main__":
    unittest.main()
