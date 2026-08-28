from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cli import check_dates, compute_stats, plot_tags, structure_dreams


class DataUtilityCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "dreams.jsonl"
        records = [
            {
                "dream_id": "dream-1",
                "date": "1/1/2024",
                "year": 2024,
                "month": 1,
                "day": 1,
                "date_sort": "2024-01-01",
                "tags": ["house"],
                "text": "First dream",
            },
            {
                "dream_id": "dream-2",
                "date": "2/1/2024",
                "year": 2024,
                "month": 2,
                "day": 1,
                "date_sort": "2024-02-01",
                "tags": ["school"],
                "text": "Second dream",
            },
        ]
        self.path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_migrated_loaders_return_the_same_records(self) -> None:
        stats_records = compute_stats.load_dreams(self.path)
        plot_records = plot_tags.load_dreams(self.path)
        feature_records = structure_dreams.load_dreams(self.path)

        self.assertEqual(stats_records, plot_records)
        self.assertEqual(plot_records, feature_records)

    def test_stats_and_plot_filters_share_date_behavior(self) -> None:
        records = compute_stats.load_dreams(self.path)

        stats_filtered = compute_stats.filter_dreams_by_date(
            records,
            start_date="2024-02-01",
        )
        plot_filtered = plot_tags.filter_dreams_by_date(
            records,
            start_date="2024-02-01",
        )

        self.assertEqual(
            [record["dream_id"] for record in stats_filtered],
            ["dream-2"],
        )
        self.assertEqual(stats_filtered, plot_filtered)

    def test_date_checker_preserves_line_aware_behavior(self) -> None:
        loaded = check_dates.load_dreams(self.path)

        self.assertEqual([line for line, _ in loaded], [1, 2])
        self.assertEqual(
            check_dates.parse_sort_date(loaded[0][1]).isoformat(),
            "2024-01-01",
        )


if __name__ == "__main__":
    unittest.main()
