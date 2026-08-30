from __future__ import annotations

import json
import unittest
from datetime import date

from dream_analysis.models import SearchResult
from dream_analysis.tools import DreamSearchTool


class FakeIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, date | None, date | None]] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, limit, start_date, end_date))
        return self.results


def result(text: str = "A hidden room appeared.") -> SearchResult:
    return SearchResult(
        dream_id="dream-1",
        document=text,
        metadata={"date": "1/2/2024"},
        distance=0.12345678,
    )


class DreamSearchToolTests(unittest.TestCase):
    def test_schema_exposes_a_bounded_query_and_optional_date_range(self) -> None:
        tool = DreamSearchTool(FakeIndex([]))

        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.schema["function"]["name"], "search_dreams")
        self.assertEqual(parameters["required"], ["query"])
        self.assertEqual(
            set(parameters["properties"]),
            {"query", "start_date", "end_date"},
        )
        self.assertEqual(parameters["properties"]["start_date"]["format"], "date")
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_bounded_json_compatible_evidence(self) -> None:
        index = FakeIndex([result("abcdefgh")])
        tool = DreamSearchTool(index, result_limit=4, max_chars_per_dream=4)

        output = tool.execute({"query": "  hidden room  "})

        self.assertEqual(index.calls, [("hidden room", 4, None, None)])
        self.assertEqual(output["result_count"], 1)
        self.assertIsNone(output["start_date"])
        self.assertIsNone(output["end_date"])
        self.assertEqual(output["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertTrue(output["dreams"][0]["truncated"])
        self.assertEqual(output["dreams"][0]["distance"], 0.123457)
        json.dumps(output)

    def test_execute_with_report_data_preserves_full_text(self) -> None:
        index = FakeIndex([result("abcdefgh")])
        tool = DreamSearchTool(index, result_limit=4, max_chars_per_dream=4)

        bounded, report = tool.execute_with_report_data({"query": "hidden room"})

        self.assertEqual(len(index.calls), 1)
        self.assertEqual(bounded["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertEqual(report["dreams"][0]["text"], "abcdefgh")
        self.assertFalse(report["dreams"][0]["truncated"])

    def test_execute_passes_normalized_inclusive_date_bounds(self) -> None:
        index = FakeIndex([result()])
        tool = DreamSearchTool(index, result_limit=4)

        output = tool.execute(
            {
                "query": "school",
                "start_date": "2024-02-01",
                "end_date": "2024-02-29",
            }
        )

        self.assertEqual(
            index.calls,
            [("school", 4, date(2024, 2, 1), date(2024, 2, 29))],
        )
        self.assertEqual(output["start_date"], "2024-02-01")
        self.assertEqual(output["end_date"], "2024-02-29")

    def test_execute_rejects_empty_queries_and_extra_arguments(self) -> None:
        tool = DreamSearchTool(FakeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            tool.execute({"query": " "})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"query": "house", "path": "/tmp"})
        with self.assertRaisesRegex(ValueError, "Invalid start_date"):
            tool.execute({"query": "house", "start_date": "last month"})
        with self.assertRaisesRegex(ValueError, "start_date"):
            tool.execute(
                {
                    "query": "house",
                    "start_date": "2024-03-01",
                    "end_date": "2024-02-01",
                }
            )

    def test_constructor_bounds_result_and_context_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            DreamSearchTool(FakeIndex([]), result_limit=21)
        with self.assertRaisesRegex(ValueError, "positive"):
            DreamSearchTool(FakeIndex([]), max_chars_per_dream=0)


if __name__ == "__main__":
    unittest.main()
