from __future__ import annotations

import unittest
from datetime import date, datetime

from dream_analysis.dates import (
    filter_records_by_date,
    format_period_label,
    parse_date_bound,
    parse_date_value,
    record_date,
)


class DateUtilityTests(unittest.TestCase):
    def test_parse_date_value_supports_stored_and_cli_shapes(self) -> None:
        self.assertEqual(parse_date_value("2024-01-02"), date(2024, 1, 2))
        self.assertEqual(parse_date_value("1/2/2024"), date(2024, 1, 2))
        self.assertEqual(parse_date_value("1/2/24"), date(2024, 1, 2))
        self.assertEqual(
            parse_date_value(datetime(2024, 1, 2, 3, 4)),
            date(2024, 1, 2),
        )
        self.assertIsNone(parse_date_value("not-a-date"))

    def test_date_sort_is_authoritative_when_present(self) -> None:
        self.assertIsNone(
            record_date({"date_sort": None, "date": "1/2/2024"})
        )
        self.assertIsNone(
            record_date({"date_sort": "invalid", "date": "1/2/2024"})
        )
        self.assertEqual(
            record_date({"date": "1/2/2024"}),
            date(2024, 1, 2),
        )

    def test_filter_is_inclusive_and_omits_unknown_dates(self) -> None:
        records = [
            {"dream_id": "early", "date_sort": "2024-01-01"},
            {"dream_id": "middle", "date_sort": "2024-01-02"},
            {"dream_id": "late", "date_sort": "2024-01-03"},
            {"dream_id": "unknown", "date_sort": None},
        ]

        selected = filter_records_by_date(
            records,
            start_date="2024-01-02",
            end_date="2024-01-03",
        )

        self.assertEqual(
            [record["dream_id"] for record in selected],
            ["middle", "late"],
        )

    def test_invalid_bound_and_reversed_range_have_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid start_date"):
            parse_date_bound("invalid", argument_name="start_date")
        with self.assertRaisesRegex(ValueError, "before or equal"):
            filter_records_by_date(
                [],
                start_date="2024-02-01",
                end_date="2024-01-01",
            )

    def test_period_labels_are_shared_across_consumers(self) -> None:
        value = date(2024, 4, 1)
        self.assertEqual(format_period_label(value, frequency="M"), "4-2024")
        self.assertEqual(format_period_label(value, frequency="Q"), "Q2-2024")
        self.assertEqual(format_period_label(value, frequency="Y"), "2024")


if __name__ == "__main__":
    unittest.main()
