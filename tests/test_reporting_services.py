from __future__ import annotations

import json
import unittest
from datetime import date

from dream_analysis.models import Dream
from dream_analysis.statistics import DreamStatisticsService
from dream_analysis.trends import TagTrendService, rank_tags


def dream(
    dream_id: str,
    day: date | None,
    *,
    tags: tuple[str, ...] = (),
    text: str = "A quiet room",
    word_count: int = 3,
) -> Dream:
    return Dream(
        dream_id=dream_id,
        date=day.isoformat() if day else "unknown",
        text=text,
        tags=tags,
        date_sort=day,
        word_count=word_count,
    )


class DreamStatisticsServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreams = [
            dream(
                "dream-1",
                date(2024, 1, 5),
                tags=("house", "flying"),
                text="A red house above trees",
                word_count=5,
            ),
            dream(
                "dream-2",
                date(2024, 4, 2),
                tags=("house",),
                text="Blue ocean below",
                word_count=3,
            ),
            dream("dream-3", None, tags=("unknown",)),
        ]

    def test_summary_is_filtered_grouped_and_json_compatible(self) -> None:
        result = DreamStatisticsService(self.dreams).summarize(
            frequency="Q",
            end_date="2024-03-31",
            common_words=2,
        )

        self.assertEqual(result["dream_count"], 1)
        self.assertEqual(result["excluded_unknown_date_count"], 1)
        self.assertEqual(result["date_min"], "2024-01-05")
        self.assertEqual(
            result["entries_per_period"],
            [{"period": "Q1-2024", "entries": 1}],
        )
        self.assertEqual(result["tag_stats"][0]["tag"], "flying")
        json.dumps(result)

    def test_filter_is_inclusive_and_validates_range(self) -> None:
        service = DreamStatisticsService(self.dreams)

        selected = service.filter_by_date(
            start_date="2024-04-02",
            end_date="2024-04-02",
        )

        self.assertEqual([item.dream_id for item in selected], ["dream-2"])
        with self.assertRaisesRegex(ValueError, "start_date"):
            service.filter_by_date(
                start_date="2024-05-01",
                end_date="2024-04-01",
            )


class TagTrendServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreams = [
            dream("dream-1", date(2024, 1, 5), tags=("house", "flying")),
            dream("dream-2", date(2024, 1, 20), tags=("house",)),
            dream("dream-3", date(2024, 4, 2), tags=("school",)),
            dream("dream-4", None, tags=("house",)),
        ]

    def test_raw_trend_has_period_totals_and_missing_tags(self) -> None:
        result = TagTrendService(self.dreams).analyze(
            frequency="Q",
            tags=["house", "missing"],
        )

        self.assertEqual(result["missing_tags"], ["missing"])
        self.assertEqual(result["excluded_unknown_date_count"], 1)
        self.assertEqual(
            result["periods"],
            [
                {
                    "period_start": "2024-01-01",
                    "period": "Q1-2024",
                    "dream_count": 2,
                    "values": {"house": 2, "missing": 0},
                },
                {
                    "period_start": "2024-04-01",
                    "period": "Q2-2024",
                    "dream_count": 1,
                    "values": {"house": 0, "missing": 0},
                },
            ],
        )
        json.dumps(result)

    def test_normalized_trend_and_tag_ranking(self) -> None:
        result = TagTrendService(self.dreams).analyze(
            tags=["flying"],
            normalize=True,
            end_date="2024-01-31",
        )

        self.assertEqual(result["periods"][0]["values"]["flying"], 50.0)
        self.assertEqual(rank_tags(self.dreams[:3], top_n=2), ["house", "flying"])


if __name__ == "__main__":
    unittest.main()
