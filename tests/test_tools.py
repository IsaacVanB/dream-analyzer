from __future__ import annotations

import json
import unittest

from dream_analysis.models import SearchResult
from dream_analysis.tools import DreamSearchTool


class FakeIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.calls.append((query, limit))
        return self.results


def result(text: str = "A hidden room appeared.") -> SearchResult:
    return SearchResult(
        dream_id="dream-1",
        document=text,
        metadata={"date": "1/2/2024"},
        distance=0.12345678,
    )


class DreamSearchToolTests(unittest.TestCase):
    def test_schema_exposes_only_a_bounded_query(self) -> None:
        tool = DreamSearchTool(FakeIndex([]))

        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.schema["function"]["name"], "search_dreams")
        self.assertEqual(parameters["required"], ["query"])
        self.assertEqual(set(parameters["properties"]), {"query"})
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_bounded_json_compatible_evidence(self) -> None:
        index = FakeIndex([result("abcdefgh")])
        tool = DreamSearchTool(index, result_limit=4, max_chars_per_dream=4)

        output = tool.execute({"query": "  hidden room  "})

        self.assertEqual(index.calls, [("hidden room", 4)])
        self.assertEqual(output["result_count"], 1)
        self.assertEqual(output["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertTrue(output["dreams"][0]["truncated"])
        self.assertEqual(output["dreams"][0]["distance"], 0.123457)
        json.dumps(output)

    def test_execute_rejects_empty_queries_and_extra_arguments(self) -> None:
        tool = DreamSearchTool(FakeIndex([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            tool.execute({"query": " "})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"query": "house", "path": "/tmp"})

    def test_constructor_bounds_result_and_context_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            DreamSearchTool(FakeIndex([]), result_limit=21)
        with self.assertRaisesRegex(ValueError, "positive"):
            DreamSearchTool(FakeIndex([]), max_chars_per_dream=0)


if __name__ == "__main__":
    unittest.main()
