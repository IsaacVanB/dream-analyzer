from __future__ import annotations

import json
import unittest
from datetime import date

from dream_analysis.models import Dream, SearchResult
from dream_analysis.tools import (
    CharacterContextTool,
    CharacterMentionsTool,
    DreamByIdTool,
    DreamDateRangeTool,
    DreamSearchTool,
    DreamStatisticsTool,
    DreamTagTool,
    TagTrendTool,
)


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


class FakeRepository:
    def __init__(self, dreams: list[Dream]) -> None:
        self.dreams = dreams
        self.calls = []

    def tagged(self, tags, *, start=None, end=None):
        self.calls.append((tags, start, end))
        return self.dreams

    def between(self, start=None, end=None):
        self.calls.append((start, end))
        return self.dreams

    def all(self):
        self.calls.append("all")
        return self.dreams

    def get(self, dream_id):
        self.calls.append(dream_id)
        return self.dreams[0]


class FakeStructuredRepository:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def all(self):
        self.calls += 1
        return self.records


class FakeCharacterRepository:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def all(self):
        self.calls += 1
        return self.records


class DreamTagToolTests(unittest.TestCase):
    def test_schema_defines_an_exact_and_tag_lookup(self) -> None:
        tool = DreamTagTool(FakeRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.schema["function"]["name"], "get_dreams_by_tags")
        self.assertEqual(parameters["required"], ["tags"])
        self.assertEqual(parameters["properties"]["tags"]["minItems"], 1)
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_all_matches_with_bounded_model_text(self) -> None:
        dreams = [
            Dream(
                dream_id="one",
                date="1/1/2024",
                tags=("school", "lucid?"),
                text="abcdefgh",
            ),
            Dream(
                dream_id="two",
                date="1/2/2024",
                tags=("school", "lucid?", "late"),
                text="ijklmnop",
            ),
        ]
        repository = FakeRepository(dreams)
        tool = DreamTagTool(repository, max_chars_per_dream=4)

        bounded, report = tool.execute_with_report_data(
            {"tags": ["school", "lucid?"]}
        )

        self.assertEqual(repository.calls, [(["school", "lucid?"], None, None)])
        self.assertEqual(bounded["result_count"], 2)
        self.assertEqual(bounded["match"], "all")
        self.assertTrue(bounded["synthesis_include_all_matches"])
        self.assertEqual(bounded["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertEqual(report["dreams"][0]["text"], "abcdefgh")
        json.dumps(bounded)

    def test_execute_validates_tags_and_dates(self) -> None:
        tool = DreamTagTool(FakeRepository([]))

        with self.assertRaisesRegex(ValueError, "non-empty array"):
            tool.execute({"tags": []})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            tool.execute({"tags": ["school", "SCHOOL"]})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"tags": ["school"], "match": "any"})
        with self.assertRaisesRegex(ValueError, "Invalid start_date"):
            tool.execute({"tags": ["school"], "start_date": "yesterday"})


class DreamDateRangeToolTests(unittest.TestCase):
    def test_schema_requires_an_inclusive_date_range(self) -> None:
        tool = DreamDateRangeTool(FakeRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(
            tool.schema["function"]["name"], "get_dreams_by_date_range"
        )
        description = tool.schema["function"]["description"]
        self.assertIn("date is the sole retrieval criterion", description)
        self.assertIn("Never infer a date range", description)
        self.assertIn("Do not use this as a fallback", description)
        self.assertEqual(parameters["required"], ["start_date", "end_date"])
        self.assertEqual(parameters["properties"]["start_date"]["format"], "date")
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_every_match_and_preserves_full_report_text(self) -> None:
        dreams = [
            Dream(
                dream_id="one",
                date="7/1/2025",
                tags=("school",),
                text="abcdefgh",
            ),
            Dream(
                dream_id="two",
                date="7/31/2025",
                tags=("travel",),
                text="ijklmnop",
            ),
        ]
        repository = FakeRepository(dreams)
        tool = DreamDateRangeTool(repository, max_chars_per_dream=4)

        bounded, report = tool.execute_with_report_data(
            {"start_date": "2025-07-01", "end_date": "2025-07-31"}
        )

        self.assertEqual(
            repository.calls, [(date(2025, 7, 1), date(2025, 7, 31))]
        )
        self.assertEqual(bounded["result_count"], 2)
        self.assertEqual(bounded["match"], "all")
        self.assertTrue(bounded["synthesis_include_all_matches"])
        self.assertEqual(bounded["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertEqual(report["dreams"][0]["text"], "abcdefgh")
        json.dumps(bounded)

    def test_execute_rejects_missing_invalid_and_reversed_dates(self) -> None:
        tool = DreamDateRangeTool(FakeRepository([]))

        with self.assertRaisesRegex(ValueError, "required"):
            tool.execute({"start_date": "2025-07-01"})
        with self.assertRaisesRegex(ValueError, "Invalid start_date"):
            tool.execute(
                {"start_date": "last month", "end_date": "2025-07-31"}
            )
        with self.assertRaisesRegex(ValueError, "start_date"):
            tool.execute(
                {"start_date": "2025-08-01", "end_date": "2025-07-31"}
            )
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute(
                {
                    "start_date": "2025-07-01",
                    "end_date": "2025-07-31",
                    "query": "themes",
                }
            )


class DreamStatisticsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreams = [
            Dream(
                dream_id="one",
                date="1/5/2025",
                date_sort=date(2025, 1, 5),
                tags=("house", "flying"),
                text="red house trees",
                word_count=3,
            ),
            Dream(
                dream_id="two",
                date="2/5/2025",
                date_sort=date(2025, 2, 5),
                tags=("house", "school"),
                text="blue school hallway",
                word_count=3,
            ),
            Dream(
                dream_id="three",
                date="3/5/2025",
                date_sort=date(2025, 3, 5),
                tags=("travel",),
                text="green train station",
                word_count=3,
            ),
            Dream(
                dream_id="unknown",
                date="0/0/00",
                date_sort=None,
                tags=("unknown",),
                text="fragment without date",
                word_count=3,
            ),
        ]

    def test_schema_exposes_bounded_statistics_parameters(self) -> None:
        tool = DreamStatisticsTool(FakeRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.name, "get_dream_statistics")
        self.assertEqual(parameters["properties"]["frequency"]["enum"], ["M", "Q", "Y"])
        self.assertEqual(parameters["properties"]["common_words"]["maximum"], 50)
        self.assertEqual(parameters["properties"]["top_tags"]["maximum"], 50)
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_bounded_model_and_complete_report_statistics(self) -> None:
        repository = FakeRepository(self.dreams)
        tool = DreamStatisticsTool(repository, max_periods=2)

        bounded, report = tool.execute_with_report_data(
            {
                "frequency": "M",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "common_words": 2,
                "min_word_length": 3,
                "top_tags": 1,
            }
        )

        self.assertEqual(repository.calls, ["all"])
        self.assertEqual(bounded["evidence_type"], "dream_statistics")
        self.assertEqual(bounded["analysis"]["dream_count"], 3)
        self.assertEqual(len(bounded["analysis"]["entries_per_period"]), 2)
        self.assertEqual(bounded["analysis"]["entry_periods_omitted"], 1)
        self.assertEqual(len(bounded["analysis"]["tag_stats"]), 1)
        self.assertEqual(bounded["analysis"]["tag_stats_omitted"], 3)
        self.assertEqual(len(report["analysis"]["entries_per_period"]), 3)
        self.assertEqual(len(report["analysis"]["tag_stats"]), 4)
        self.assertTrue(any("unknown date" in item for item in bounded["warnings"]))
        self.assertTrue(any("journal tags" in item for item in bounded["warnings"]))
        self.assertTrue(any("middle periods" in item for item in bounded["warnings"]))
        json.dumps(bounded)
        json.dumps(report)

    def test_execute_uses_defaults_and_validates_arguments(self) -> None:
        tool = DreamStatisticsTool(FakeRepository(self.dreams))

        result = tool.execute({})

        self.assertEqual(
            result["parameters"],
            {
                "frequency": "M",
                "start_date": None,
                "end_date": None,
                "common_words": 20,
                "min_word_length": 3,
                "top_tags": 20,
            },
        )
        with self.assertRaisesRegex(ValueError, "frequency"):
            tool.execute({"frequency": "W"})
        with self.assertRaisesRegex(ValueError, "common_words"):
            tool.execute({"common_words": 51})
        with self.assertRaisesRegex(ValueError, "common_words"):
            tool.execute({"common_words": True})
        with self.assertRaisesRegex(ValueError, "start_date"):
            tool.execute(
                {"start_date": "2025-02-01", "end_date": "2025-01-01"}
            )
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"stopwords_path": "/tmp/words.txt"})


class TagTrendToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreams = [
            Dream(
                dream_id="one",
                date="1/5/2025",
                date_sort=date(2025, 1, 5),
                tags=("School",),
                text="school hallway",
                word_count=2,
            ),
            Dream(
                dream_id="two",
                date="3/5/2025",
                date_sort=date(2025, 3, 5),
                tags=("school", "house"),
                text="school house",
                word_count=2,
            ),
            Dream(
                dream_id="unknown",
                date="0/0/00",
                date_sort=None,
                tags=("school",),
                text="unknown date",
                word_count=2,
            ),
        ]

    def test_schema_exposes_bounded_trend_parameters(self) -> None:
        tool = TagTrendTool(FakeRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.name, "analyze_tag_trends")
        self.assertEqual(parameters["properties"]["frequency"]["enum"], ["M", "Q", "Y"])
        self.assertEqual(parameters["properties"]["tags"]["maxItems"], 10)
        self.assertEqual(parameters["properties"]["top_n"]["maximum"], 20)
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_normalizes_and_returns_complete_report_periods(self) -> None:
        repository = FakeRepository(self.dreams)
        tool = TagTrendTool(repository, max_periods=2)

        bounded, report = tool.execute_with_report_data(
            {
                "tags": ["SCHOOL", "missing"],
                "frequency": "M",
                "normalize": True,
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "top_n": 5,
            }
        )

        self.assertEqual(repository.calls, ["all"])
        self.assertEqual(bounded["evidence_type"], "tag_trends")
        self.assertEqual(bounded["analysis"]["value_unit"], "percent")
        self.assertEqual(bounded["analysis"]["missing_tags"], ["missing"])
        self.assertEqual(bounded["analysis"]["period_count"], 3)
        self.assertEqual(bounded["analysis"]["periods_omitted"], 1)
        self.assertEqual(len(bounded["analysis"]["periods"]), 2)
        self.assertEqual(len(report["analysis"]["periods"]), 3)
        self.assertEqual(report["analysis"]["periods"][1]["dream_count"], 0)
        self.assertTrue(any("No matching" in item for item in bounded["warnings"]))
        self.assertTrue(any("unknown date" in item for item in bounded["warnings"]))
        self.assertTrue(any("no dated dreams" in item for item in bounded["warnings"]))
        self.assertTrue(any("middle periods" in item for item in bounded["warnings"]))
        json.dumps(bounded)
        json.dumps(report)

    def test_execute_defaults_to_normalized_top_tags_and_validates(self) -> None:
        tool = TagTrendTool(FakeRepository(self.dreams))

        result = tool.execute({})

        self.assertTrue(result["parameters"]["normalize"])
        self.assertEqual(result["parameters"]["top_n"], 10)
        self.assertIsNone(result["parameters"]["tags"])
        with self.assertRaisesRegex(ValueError, "normalize"):
            tool.execute({"normalize": 1})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            tool.execute({"tags": ["school", "SCHOOL"]})
        with self.assertRaisesRegex(ValueError, "top_n"):
            tool.execute({"top_n": 21})
        with self.assertRaisesRegex(ValueError, "frequency"):
            tool.execute({"frequency": "W"})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"plot_path": "/tmp/trend.png"})


class CharacterContextToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "maya",
                "name": "Maya",
                "aliases": ["May"],
                "relationship": "close friend",
                "context": "A calm and curious investigator.",
                "mentions": {
                    "count": 3,
                    "first_date": "2024-01-01",
                    "last_date": "2025-01-01",
                    "dream_ids": ["one", "two", "three"],
                },
            }
        ]

    def test_schema_requires_bounded_names_and_optional_dream_date(self) -> None:
        tool = CharacterContextTool(FakeCharacterRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.name, "get_character_context")
        self.assertEqual(parameters["required"], ["names"])
        self.assertEqual(parameters["properties"]["names"]["maxItems"], 20)
        self.assertEqual(parameters["properties"]["dream_date"]["format"], "date")
        self.assertFalse(parameters["additionalProperties"])

    def test_matches_aliases_and_preserves_full_report_context(self) -> None:
        repository = FakeCharacterRepository(self.records)
        tool = CharacterContextTool(repository, max_context_chars=10)

        bounded, report = tool.execute_with_report_data(
            {"names": ["may", "Theo"]}
        )

        self.assertEqual(repository.calls, 1)
        self.assertEqual(bounded["evidence_type"], "character_context")
        self.assertEqual(bounded["analysis"]["matched_name_count"], 1)
        self.assertEqual(bounded["analysis"]["missing_names"], ["Theo"])
        character = bounded["analysis"]["characters"][0]
        self.assertEqual(character["name"], "Maya")
        self.assertEqual(character["matched_by"], "alias")
        self.assertEqual(character["matched_value"], "May")
        self.assertEqual(character["context"], "A calm and\n[TRUNCATED]")
        self.assertNotIn("dream_ids", character["mentions"])
        self.assertEqual(
            report["analysis"]["characters"][0]["context"],
            "A calm and curious investigator.",
        )
        self.assertTrue(any("Theo" in warning for warning in bounded["warnings"]))
        self.assertTrue(
            any("truncated" in warning for warning in bounded["warnings"])
        )
        json.dumps(bounded)
        json.dumps(report)

    def test_selects_date_bounded_history_and_reports_ambiguity(self) -> None:
        records = [
            {
                "id": "alex-one",
                "name": "Alexandra",
                "aliases": ["Alex"],
                "relationship_history": [
                    {
                        "start_date": None,
                        "end_date": "2024-12-31",
                        "relationship": "coworker",
                        "context": "Worked together.",
                    },
                    {
                        "start_date": "2025-01-01",
                        "end_date": None,
                        "relationship": "friend",
                        "context": "Stayed friends.",
                    },
                ],
            },
            {
                "id": "alex-two",
                "name": "Alexander",
                "aliases": ["Alex"],
                "relationship": "cousin",
                "context": "A different Alex.",
            },
        ]
        tool = CharacterContextTool(FakeCharacterRepository(records))

        result = tool.execute({"names": ["ALEX"], "dream_date": "2025-03-01"})

        self.assertEqual(len(result["analysis"]["characters"]), 2)
        self.assertEqual(result["analysis"]["ambiguous_names"], ["ALEX"])
        alexandra = result["analysis"]["characters"][0]
        self.assertEqual(len(alexandra["relationship_history"]), 1)
        self.assertEqual(
            alexandra["relationship_history"][0]["relationship"], "friend"
        )
        self.assertTrue(any("ambiguous" in warning for warning in result["warnings"]))

    def test_validates_arguments(self) -> None:
        tool = CharacterContextTool(FakeCharacterRepository(self.records))

        with self.assertRaisesRegex(ValueError, "non-empty array"):
            tool.execute({"names": []})
        with self.assertRaisesRegex(ValueError, "duplicates"):
            tool.execute({"names": ["Maya", "maya"]})
        with self.assertRaisesRegex(ValueError, "Invalid dream_date"):
            tool.execute({"names": ["Maya"], "dream_date": "last year"})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"names": ["Maya"], "path": "/tmp/characters.json"})


class CharacterMentionsToolTests(unittest.TestCase):
    def test_schema_exposes_bounded_character_parameters(self) -> None:
        tool = CharacterMentionsTool(
            FakeRepository([]), FakeStructuredRepository([])
        )
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.name, "get_character_mentions")
        self.assertEqual(parameters["properties"]["names"]["maxItems"], 20)
        self.assertEqual(parameters["properties"]["limit"]["maximum"], 50)
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_reports_coverage_and_bounds_dream_ids(self) -> None:
        dreams = [
            Dream(
                dream_id=f"dream-{index}",
                date=f"1/{index + 1}/2025",
                date_sort=date(2025, 1, index + 1),
                text="Dream",
            )
            for index in range(22)
        ]
        dreams.append(
            Dream(
                dream_id="unstructured",
                date="2/1/2025",
                date_sort=date(2025, 2, 1),
                text="Dream",
            )
        )
        records = [
            {
                "dream_id": f"dream-{index}",
                "date_sort": "1999-01-01",
                "named_characters": ["Maya", "maya"],
            }
            for index in range(22)
        ]
        records.append(
            {
                "dream_id": "orphan",
                "date_sort": "2025-01-01",
                "named_characters": ["Theo"],
            }
        )
        dream_repository = FakeRepository(dreams)
        structured_repository = FakeStructuredRepository(records)
        tool = CharacterMentionsTool(dream_repository, structured_repository)

        bounded, report = tool.execute_with_report_data(
            {
                "names": ["maya", "Theo"],
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "minimum_mentions": 1,
                "limit": 25,
            }
        )

        coverage = bounded["analysis"]["coverage"]
        self.assertEqual(dream_repository.calls, ["all"])
        self.assertEqual(structured_repository.calls, 1)
        self.assertEqual(coverage["parsed_dream_count"], 23)
        self.assertEqual(coverage["structured_current_dream_count"], 22)
        self.assertEqual(coverage["orphaned_structured_record_count"], 1)
        self.assertEqual(bounded["analysis"]["missing_names"], ["Theo"])
        self.assertEqual(bounded["analysis"]["characters"][0]["name"], "Maya")
        self.assertEqual(
            bounded["analysis"]["characters"][0]["mentions"]["count"], 22
        )
        self.assertEqual(
            len(bounded["analysis"]["characters"][0]["mentions"]["dream_ids"]),
            20,
        )
        self.assertEqual(
            bounded["analysis"]["characters"][0]["mentions"]["dream_ids_omitted"],
            2,
        )
        self.assertEqual(
            len(report["analysis"]["characters"][0]["mentions"]["dream_ids"]),
            22,
        )
        self.assertTrue(any("current dreams" in item for item in bounded["warnings"]))
        self.assertTrue(any("do not match" in item for item in bounded["warnings"]))
        self.assertTrue(any("Theo" in item for item in bounded["warnings"]))
        self.assertTrue(any("truncated" in item for item in bounded["warnings"]))
        json.dumps(bounded)
        json.dumps(report)

    def test_execute_uses_current_dates_and_validates_arguments(self) -> None:
        dreams = [
            Dream(
                dream_id="one",
                date="5/1/2025",
                date_sort=date(2025, 5, 1),
                text="Dream",
            ),
            Dream(
                dream_id="unknown",
                date="0/0/00",
                date_sort=None,
                text="Dream",
            ),
        ]
        records = [
            {
                "dream_id": "one",
                "date_sort": "1999-01-01",
                "named_characters": ["Maya"],
            },
            {
                "dream_id": "unknown",
                "date_sort": "1999-01-02",
                "named_characters": ["Theo"],
            },
        ]
        tool = CharacterMentionsTool(
            FakeRepository(dreams), FakeStructuredRepository(records)
        )

        result = tool.execute(
            {"start_date": "2025-01-01", "end_date": "2025-12-31"}
        )

        self.assertEqual(
            result["analysis"]["characters"][0]["mentions"]["first_date"],
            "2025-05-01",
        )
        self.assertEqual(result["analysis"]["excluded_unknown_date_count"], 1)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            tool.execute({"names": ["Maya", "maya"]})
        with self.assertRaisesRegex(ValueError, "limit"):
            tool.execute({"limit": 51})
        with self.assertRaisesRegex(ValueError, "minimum_mentions"):
            tool.execute({"minimum_mentions": False})
        with self.assertRaisesRegex(ValueError, "start_date"):
            tool.execute(
                {"start_date": "2025-02-01", "end_date": "2025-01-01"}
            )
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"temporal_context": True})


class DreamByIdToolTests(unittest.TestCase):
    def test_schema_requires_only_an_exact_dream_id(self) -> None:
        tool = DreamByIdTool(FakeRepository([]))
        parameters = tool.schema["function"]["parameters"]

        self.assertEqual(tool.schema["function"]["name"], "get_dream_by_id")
        self.assertEqual(parameters["required"], ["dream_id"])
        self.assertEqual(set(parameters["properties"]), {"dream_id"})
        self.assertFalse(parameters["additionalProperties"])

    def test_execute_returns_bounded_and_full_text_from_one_lookup(self) -> None:
        dream = Dream(
            dream_id="dream-2025-1-9-0",
            date="1/9/2025",
            tags=("school",),
            text="abcdefgh",
        )
        repository = FakeRepository([dream])
        tool = DreamByIdTool(repository, max_chars_per_dream=4)

        bounded, report = tool.execute_with_report_data(
            {"dream_id": "  dream-2025-1-9-0  "}
        )

        self.assertEqual(repository.calls, ["dream-2025-1-9-0"])
        self.assertEqual(bounded["requested_dream_id"], "dream-2025-1-9-0")
        self.assertEqual(bounded["result_count"], 1)
        self.assertEqual(bounded["dreams"][0]["text"], "abcd\n[TRUNCATED]")
        self.assertEqual(report["dreams"][0]["text"], "abcdefgh")
        json.dumps(bounded)

    def test_execute_rejects_invalid_arguments(self) -> None:
        tool = DreamByIdTool(FakeRepository([]))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            tool.execute({"dream_id": " "})
        with self.assertRaisesRegex(ValueError, "unexpected arguments"):
            tool.execute({"dream_id": "one", "path": "/tmp"})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            tool.execute({"dream_id": "x" * 201})


if __name__ == "__main__":
    unittest.main()
