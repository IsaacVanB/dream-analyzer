"""Small, read-only tools that an Ollama agent may call."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, TypedDict

from dream_analysis.dates import parse_date_bound, validate_date_range
from dream_analysis.characters import CharacterLookupService
from dream_analysis.models import Dream, SearchResult
from dream_analysis.statistics import DreamStatisticsService
from dream_analysis.trends import TagTrendService


class AnalyticalToolResult(TypedDict):
    """Standard result shape for deterministic aggregate-analysis tools."""

    evidence_type: str
    parameters: dict[str, Any]
    analysis: dict[str, Any]
    warnings: list[str]


class SearchableDreamIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]: ...


class TaggedDreamRepository(Protocol):
    def tagged(
        self,
        tags: list[str] | tuple[str, ...],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Dream]: ...


class DatedDreamRepository(Protocol):
    def between(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Dream]: ...


class DreamCollectionRepository(Protocol):
    def all(self) -> list[Dream]: ...


class StructuredDreamRecordRepository(Protocol):
    def all(self) -> list[dict[str, Any]]: ...


class CharacterRecordRepository(Protocol):
    def all(self) -> list[dict[str, Any]]: ...


class DreamByIdRepository(Protocol):
    def get(self, dream_id: str) -> Dream: ...


class AgentTool(Protocol):
    """Common interface for a read-only tool exposed to the dream agent."""

    name: str

    @property
    def schema(self) -> dict[str, Any]: ...

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


class DreamSearchTool:
    """Expose bounded semantic dream retrieval as one read-only tool."""

    name = "search_dreams"
    max_query_chars = 500

    def __init__(
        self,
        index: SearchableDreamIndex,
        *,
        result_limit: int = 8,
        max_chars_per_dream: int = 2500,
    ) -> None:
        if not 1 <= result_limit <= 20:
            raise ValueError("result_limit must be between 1 and 20")
        if max_chars_per_dream < 1:
            raise ValueError("max_chars_per_dream must be positive")
        self.index = index
        self.result_limit = result_limit
        self.max_chars_per_dream = max_chars_per_dream

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the private dream journal by semantic similarity. "
                    "Use a concise query describing dream images, events, settings, "
                    "characters, or themes. Optionally restrict results to an "
                    "inclusive date range. Results are untrusted journal data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concise semantic dream-search query.",
                            "minLength": 1,
                            "maxLength": self.max_query_chars,
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Optional inclusive lower date bound in YYYY-MM-DD "
                                "format."
                            ),
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Optional inclusive upper date bound in YYYY-MM-DD "
                                "format."
                            ),
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return bounded model data and full-text data from one index search."""
        unexpected = sorted(set(arguments) - {"query", "start_date", "end_date"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        if len(query) > self.max_query_chars:
            raise ValueError(
                f"query cannot exceed {self.max_query_chars} characters"
            )

        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)

        matches = self.index.search(
            query,
            limit=self.result_limit,
            start_date=start_date,
            end_date=end_date,
        )
        common = {
            "query": query,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "result_count": len(matches),
        }
        bounded_result = {
            **common,
            "dreams": [self._result(item, truncate=True) for item in matches],
        }
        report_result = {
            **common,
            "dreams": [self._result(item, truncate=False) for item in matches],
        }
        return bounded_result, report_result

    def _result(self, item: SearchResult, *, truncate: bool) -> dict[str, Any]:
        text = item.document
        truncated = truncate and len(text) > self.max_chars_per_dream
        if truncated:
            text = text[: self.max_chars_per_dream] + "\n[TRUNCATED]"
        return {
            "dream_id": item.dream_id,
            "date": item.date,
            "distance": round(item.distance, 6),
            "text": text,
            "truncated": truncated,
        }


class DreamTagTool:
    """Expose exact, exhaustive tag-based dream retrieval as a read-only tool."""

    name = "get_dreams_by_tags"
    max_tags = 10
    max_tag_chars = 100

    def __init__(
        self,
        repository: TaggedDreamRepository,
        *,
        max_chars_per_dream: int = 2500,
    ) -> None:
        if max_chars_per_dream < 1:
            raise ValueError("max_chars_per_dream must be positive")
        self.repository = repository
        self.max_chars_per_dream = max_chars_per_dream

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get every dream that has all of the specified journal tags. "
                    "Use this for exact tag requests, not semantic topics. Tag "
                    "matching is case-insensitive, exact, and uses AND when more "
                    "than one tag is supplied. Punctuation is part of a tag, so "
                    "'lucid?' must include the question mark."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "description": (
                                "One or more exact tags all dreams must have."
                            ),
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_tag_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_tags,
                            "uniqueItems": True,
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive lower date bound.",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive upper date bound.",
                        },
                    },
                    "required": ["tags"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unexpected = sorted(set(arguments) - {"tags", "start_date", "end_date"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")

        raw_tags = arguments.get("tags")
        if not isinstance(raw_tags, list) or not raw_tags:
            raise ValueError("tags must be a non-empty array of strings")
        if len(raw_tags) > self.max_tags:
            raise ValueError(f"tags cannot contain more than {self.max_tags} items")
        if any(not isinstance(tag, str) or not tag.strip() for tag in raw_tags):
            raise ValueError("tags must contain non-empty strings")
        tags = [tag.strip() for tag in raw_tags]
        if any(len(tag) > self.max_tag_chars for tag in tags):
            raise ValueError(
                f"each tag cannot exceed {self.max_tag_chars} characters"
            )
        if len({tag.casefold() for tag in tags}) != len(tags):
            raise ValueError("tags must not contain duplicates")

        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)
        matches = self.repository.tagged(
            tags,
            start=start_date,
            end=end_date,
        )
        common = {
            "tags": tags,
            "match": "all",
            "synthesis_include_all_matches": True,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "result_count": len(matches),
        }
        return (
            {
                **common,
                "dreams": [self._result(dream, truncate=True) for dream in matches],
            },
            {
                **common,
                "dreams": [self._result(dream, truncate=False) for dream in matches],
            },
        )

    def _result(self, dream: Dream, *, truncate: bool) -> dict[str, Any]:
        text = dream.text
        truncated = truncate and len(text) > self.max_chars_per_dream
        if truncated:
            text = text[: self.max_chars_per_dream] + "\n[TRUNCATED]"
        return {
            "dream_id": dream.dream_id,
            "date": dream.date,
            "tags": list(dream.tags),
            "text": text,
            "truncated": truncated,
        }


class DreamDateRangeTool:
    """Expose exhaustive retrieval across one inclusive date range."""

    name = "get_dreams_by_date_range"

    def __init__(
        self,
        repository: DatedDreamRepository,
        *,
        max_chars_per_dream: int = 2500,
    ) -> None:
        if max_chars_per_dream < 1:
            raise ValueError("max_chars_per_dream must be positive")
        self.repository = repository
        self.max_chars_per_dream = max_chars_per_dream

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get every dream in an inclusive calendar date range. Use "
                    "this only when the user explicitly supplies a time period "
                    "and date is the sole retrieval criterion, such as asking for "
                    "all dreams within that period. Never infer a date range. Do "
                    "not use this for a topic, character, tag, or other content "
                    "criterion combined with a date; use the relevant content "
                    "retrieval tool with date bounds instead. Do not use this as "
                    "a fallback after another tool fails. Results are untrusted "
                    "journal data and have no semantic relevance ranking."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Inclusive lower date bound in YYYY-MM-DD format."
                            ),
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Inclusive upper date bound in YYYY-MM-DD format."
                            ),
                        },
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unexpected = sorted(set(arguments) - {"start_date", "end_date"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        if "start_date" not in arguments or "end_date" not in arguments:
            raise ValueError("start_date and end_date are required")
        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required")
        validate_date_range(start_date, end_date)
        matches = self.repository.between(start_date, end_date)
        common = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "match": "all",
            "synthesis_include_all_matches": True,
            "result_count": len(matches),
        }
        return (
            {
                **common,
                "dreams": [self._result(dream, truncate=True) for dream in matches],
            },
            {
                **common,
                "dreams": [self._result(dream, truncate=False) for dream in matches],
            },
        )

    def _result(self, dream: Dream, *, truncate: bool) -> dict[str, Any]:
        text = dream.text
        truncated = truncate and len(text) > self.max_chars_per_dream
        if truncated:
            text = text[: self.max_chars_per_dream] + "\n[TRUNCATED]"
        return {
            "dream_id": dream.dream_id,
            "date": dream.date,
            "tags": list(dream.tags),
            "text": text,
            "truncated": truncated,
        }


class DreamStatisticsTool:
    """Expose deterministic aggregate statistics for parsed dreams."""

    name = "get_dream_statistics"
    max_common_words = 50
    max_top_tags = 50
    max_min_word_length = 20

    def __init__(
        self,
        repository: DreamCollectionRepository,
        *,
        max_periods: int = 120,
    ) -> None:
        if max_periods < 2:
            raise ValueError("max_periods must be at least 2")
        self.repository = repository
        self.max_periods = max_periods

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Calculate exact, deterministic statistics from parsed dream "
                    "records: dream counts, entries per month/quarter/year, common "
                    "journal tags, dream lengths, and common non-stopword vocabulary. "
                    "Use this for quantitative questions, not semantic themes or "
                    "changes in tags over time. Optional dates are inclusive."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "string",
                            "enum": ["M", "Q", "Y"],
                            "description": (
                                "Period grouping: M=month, Q=quarter, Y=year."
                            ),
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive lower date bound.",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive upper date bound.",
                        },
                        "common_words": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": self.max_common_words,
                            "description": (
                                "Number of common non-stopword tokens to return."
                            ),
                        },
                        "min_word_length": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_min_word_length,
                            "description": "Minimum token length for common words.",
                        },
                        "top_tags": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_top_tags,
                            "description": "Maximum journal tags shown to the model.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        allowed = {
            "frequency",
            "start_date",
            "end_date",
            "common_words",
            "min_word_length",
            "top_tags",
        }
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")

        frequency = arguments.get("frequency", "M")
        if frequency not in {"M", "Q", "Y"}:
            raise ValueError("frequency must be one of: M, Q, Y")
        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)
        common_words = self._bounded_integer(
            arguments.get("common_words", 20),
            name="common_words",
            minimum=0,
            maximum=self.max_common_words,
        )
        min_word_length = self._bounded_integer(
            arguments.get("min_word_length", 3),
            name="min_word_length",
            minimum=1,
            maximum=self.max_min_word_length,
        )
        top_tags = self._bounded_integer(
            arguments.get("top_tags", 20),
            name="top_tags",
            minimum=1,
            maximum=self.max_top_tags,
        )
        parameters = {
            "frequency": frequency,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "common_words": common_words,
            "min_word_length": min_word_length,
            "top_tags": top_tags,
        }
        analysis = DreamStatisticsService(self.repository.all()).summarize(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            common_words=common_words,
            min_word_length=min_word_length,
        )

        report_warnings = self._source_warnings(analysis)
        report: AnalyticalToolResult = {
            "evidence_type": "dream_statistics",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": report_warnings,
        }

        bounded_analysis = dict(analysis)
        tag_stats = list(analysis["tag_stats"])
        periods = list(analysis["entries_per_period"])
        bounded_analysis["tag_count"] = len(tag_stats)
        bounded_analysis["tag_stats"] = tag_stats[:top_tags]
        bounded_analysis["tag_stats_omitted"] = max(0, len(tag_stats) - top_tags)
        bounded_periods, omitted_periods = self._bound_periods(periods)
        bounded_analysis["entries_per_period"] = bounded_periods
        bounded_analysis["entry_period_count"] = len(periods)
        bounded_analysis["entry_periods_omitted"] = omitted_periods

        warnings = list(report_warnings)
        if len(tag_stats) > top_tags:
            warnings.append(
                f"Showing the top {top_tags} of {len(tag_stats)} journal tags; "
                "the saved report retains all tags."
            )
        if omitted_periods:
            warnings.append(
                f"Showing the earliest and latest {self.max_periods} of "
                f"{len(periods)} periods; {omitted_periods} middle periods are "
                "omitted from model evidence but retained in the saved report."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "dream_statistics",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": warnings,
        }
        return bounded, report

    @staticmethod
    def _bounded_integer(
        value: Any,
        *,
        name: str,
        minimum: int,
        maximum: int,
    ) -> int:
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _source_warnings(analysis: dict[str, Any]) -> list[str]:
        excluded = int(analysis.get("excluded_unknown_date_count", 0))
        if not excluded:
            return []
        noun = "dream has" if excluded == 1 else "dreams have"
        return [
            f"{excluded} {noun} an unknown date and cannot be included in "
            "date-based statistics."
        ]

    def _bound_periods(
        self, periods: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        if len(periods) <= self.max_periods:
            return periods, 0
        first_count = self.max_periods // 2
        last_count = self.max_periods - first_count
        return (
            periods[:first_count] + periods[-last_count:],
            len(periods) - self.max_periods,
        )


class TagTrendTool:
    """Expose deterministic journal-tag frequencies over time."""

    name = "analyze_tag_trends"
    max_tags = 10
    max_tag_chars = 100
    max_top_n = 20

    def __init__(
        self,
        repository: DreamCollectionRepository,
        *,
        max_periods: int = 120,
    ) -> None:
        if max_periods < 2:
            raise ValueError("max_periods must be at least 2")
        self.repository = repository
        self.max_periods = max_periods

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Analyze exact journal-tag frequencies over time using "
                    "deterministic counts or percentages. Use this for questions "
                    "about whether tagged dream topics increased, decreased, or "
                    "changed across months, quarters, or years. This analyzes "
                    "journal tags, not semantic themes inferred from dream text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "description": (
                                "Optional exact tags to compare case-insensitively. "
                                "When omitted, the most frequent tags are selected."
                            ),
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_tag_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_tags,
                            "uniqueItems": True,
                        },
                        "frequency": {
                            "type": "string",
                            "enum": ["M", "Q", "Y"],
                            "description": (
                                "Period grouping: M=month, Q=quarter, Y=year."
                            ),
                        },
                        "normalize": {
                            "type": "boolean",
                            "description": (
                                "Use percent of dreams per period instead of raw "
                                "counts. Defaults to true for fair comparisons."
                            ),
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive lower date bound.",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive upper date bound.",
                        },
                        "top_n": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_top_n,
                            "description": (
                                "Number of frequent tags selected when tags are "
                                "omitted. Ignored when explicit tags are supplied."
                            ),
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        allowed = {
            "tags",
            "frequency",
            "normalize",
            "start_date",
            "end_date",
            "top_n",
        }
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")

        tags = self._tags(arguments.get("tags"))
        frequency = arguments.get("frequency", "M")
        if frequency not in {"M", "Q", "Y"}:
            raise ValueError("frequency must be one of: M, Q, Y")
        normalize = arguments.get("normalize", True)
        if type(normalize) is not bool:
            raise ValueError("normalize must be a boolean")
        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)
        top_n = self._bounded_integer(arguments.get("top_n", 10), name="top_n")
        parameters = {
            "tags": tags,
            "frequency": frequency,
            "normalize": normalize,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "top_n": top_n,
        }
        analysis = TagTrendService(self.repository.all()).analyze(
            frequency=frequency,
            tags=tags,
            top_n=top_n,
            normalize=normalize,
            include_empty_periods=True,
            start_date=start_date,
            end_date=end_date,
        )

        report_warnings = self._source_warnings(analysis)
        report: AnalyticalToolResult = {
            "evidence_type": "tag_trends",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": report_warnings,
        }
        periods = list(analysis["periods"])
        bounded_periods, omitted_periods = self._bound_periods(periods)
        bounded_analysis = dict(analysis)
        bounded_analysis["periods"] = bounded_periods
        bounded_analysis["period_count"] = len(periods)
        bounded_analysis["periods_omitted"] = omitted_periods
        warnings = list(report_warnings)
        if omitted_periods:
            warnings.append(
                f"Showing the earliest and latest {self.max_periods} of "
                f"{len(periods)} periods; {omitted_periods} middle periods are "
                "omitted from model evidence but retained in the saved report."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "tag_trends",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": warnings,
        }
        return bounded, report

    def _tags(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError("tags must be a non-empty array of strings")
        if len(value) > self.max_tags:
            raise ValueError(f"tags cannot contain more than {self.max_tags} items")
        if any(not isinstance(tag, str) or not tag.strip() for tag in value):
            raise ValueError("tags must contain non-empty strings")
        tags = [tag.strip() for tag in value]
        if any(len(tag) > self.max_tag_chars for tag in tags):
            raise ValueError(
                f"each tag cannot exceed {self.max_tag_chars} characters"
            )
        if len({tag.casefold() for tag in tags}) != len(tags):
            raise ValueError("tags must not contain case-insensitive duplicates")
        return tags

    def _bounded_integer(self, value: Any, *, name: str) -> int:
        if type(value) is not int or not 1 <= value <= self.max_top_n:
            raise ValueError(f"{name} must be between 1 and {self.max_top_n}")
        return value

    @staticmethod
    def _source_warnings(analysis: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        missing = list(analysis.get("missing_tags", []))
        if missing:
            warnings.append(
                "No matching journal tag was found for: "
                + ", ".join(str(tag) for tag in missing)
                + "."
            )
        unknown = int(analysis.get("excluded_unknown_date_count", 0))
        if unknown:
            noun = "dream has" if unknown == 1 else "dreams have"
            warnings.append(
                f"{unknown} {noun} an unknown date and cannot be included in "
                "tag trends."
            )
        empty_periods = sum(
            period.get("dream_count") == 0
            for period in analysis.get("periods", [])
        )
        if empty_periods:
            warnings.append(
                f"{empty_periods} period(s) contain no dated dreams; they are "
                "included explicitly with zero values."
            )
        return warnings

    def _bound_periods(
        self, periods: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        if len(periods) <= self.max_periods:
            return periods, 0
        first_count = self.max_periods // 2
        last_count = self.max_periods - first_count
        return (
            periods[:first_count] + periods[-last_count:],
            len(periods) - self.max_periods,
        )


class CharacterContextTool:
    """Expose manually curated character context by canonical name or alias."""

    name = "get_character_context"
    max_names = 20
    max_name_chars = 100
    max_aliases = 20
    max_history_entries = 20

    def __init__(
        self,
        repository: CharacterRecordRepository,
        *,
        max_relationship_chars: int = 500,
        max_context_chars: int = 2000,
    ) -> None:
        if max_relationship_chars < 1:
            raise ValueError("max_relationship_chars must be positive")
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be positive")
        self.repository = repository
        self.max_relationship_chars = max_relationship_chars
        self.max_context_chars = max_context_chars

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up manually curated background about named characters "
                    "in the private dream journal. Match canonical names and "
                    "aliases case-insensitively. Use this after a dream retrieval "
                    "mentions a named character, or when the user asks who that "
                    "person is. An optional dream_date selects relationship context "
                    "that applied on that date. This is contextual reference data, "
                    "not evidence that the person appeared in a particular dream."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": "Character names or aliases to look up.",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_name_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_names,
                            "uniqueItems": True,
                        },
                        "dream_date": {
                            "type": "string",
                            "format": "date",
                            "description": (
                                "Optional date of the dream in YYYY-MM-DD format, "
                                "used to select date-bounded relationship history."
                            ),
                        },
                    },
                    "required": ["names"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        unexpected = sorted(set(arguments) - {"names", "dream_date"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        names = self._names(arguments.get("names"))
        dream_date = parse_date_bound(
            arguments.get("dream_date"), argument_name="dream_date"
        )
        records = self.repository.all()
        index = self._index(records)

        characters: list[dict[str, Any]] = []
        missing_names: list[str] = []
        ambiguous_names: list[str] = []
        for requested_name in names:
            candidates = index.get(requested_name.casefold(), [])
            if not candidates:
                missing_names.append(requested_name)
                continue
            if len(candidates) > 1:
                ambiguous_names.append(requested_name)
            for character, matched_by, matched_value in candidates:
                characters.append(
                    self._result(
                        character,
                        requested_name=requested_name,
                        matched_by=matched_by,
                        matched_value=matched_value,
                        dream_date=dream_date,
                    )
                )

        parameters = {
            "names": names,
            "dream_date": dream_date.isoformat() if dream_date else None,
        }
        analysis = {
            "dictionary_character_count": len(records),
            "requested_name_count": len(names),
            "matched_name_count": len(names) - len(missing_names),
            "missing_names": missing_names,
            "ambiguous_names": ambiguous_names,
            "characters": characters,
        }
        warnings = self._warnings(
            missing_names=missing_names,
            ambiguous_names=ambiguous_names,
            characters=characters,
            dream_date=dream_date,
        )
        report: AnalyticalToolResult = {
            "evidence_type": "character_context",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": warnings,
        }

        bounded_characters = [self._bounded(character) for character in characters]
        bounded_analysis = {**analysis, "characters": bounded_characters}
        bounded_warnings = list(warnings)
        truncated = sum(
            bool(character.get("content_truncated"))
            for character in bounded_characters
        )
        if truncated:
            bounded_warnings.append(
                f"Context was truncated for {truncated} character result(s) in "
                "model evidence; the saved report retains complete content."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "character_context",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": bounded_warnings,
        }
        return bounded, report

    def _names(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("names must be a non-empty array of strings")
        if len(value) > self.max_names:
            raise ValueError(f"names cannot contain more than {self.max_names} items")
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("names must contain non-empty strings")
        names = [name.strip() for name in value]
        if any(len(name) > self.max_name_chars for name in names):
            raise ValueError(
                f"each name cannot exceed {self.max_name_chars} characters"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("names must not contain case-insensitive duplicates")
        return names

    @staticmethod
    def _index(
        records: list[dict[str, Any]],
    ) -> dict[str, list[tuple[dict[str, Any], str, str]]]:
        index: dict[str, list[tuple[dict[str, Any], str, str]]] = {}
        for character in records:
            values = [("name", str(character["name"]))]
            values.extend(("alias", str(alias)) for alias in character.get("aliases", []))
            seen_values: set[str] = set()
            for matched_by, value in values:
                identity = value.strip().casefold()
                if identity in seen_values:
                    continue
                seen_values.add(identity)
                index.setdefault(identity, []).append(
                    (character, matched_by, value.strip())
                )
        return index

    def _result(
        self,
        character: dict[str, Any],
        *,
        requested_name: str,
        matched_by: str,
        matched_value: str,
        dream_date: date | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested_name": requested_name,
            "matched_by": matched_by,
            "matched_value": matched_value,
            "id": character["id"],
            "name": character["name"],
            "aliases": list(character.get("aliases", [])),
        }
        if "relationship_history" in character:
            history = [dict(entry) for entry in character["relationship_history"]]
            if dream_date is not None:
                history = [
                    entry
                    for entry in history
                    if self._history_applies(entry, dream_date)
                ]
            result["relationship_history"] = history
        else:
            result["relationship"] = character.get("relationship", "")
            result["context"] = character.get("context", "")
        if isinstance(character.get("mentions"), dict):
            mentions = character["mentions"]
            result["mentions"] = {
                key: mentions.get(key)
                for key in ("count", "first_date", "last_date")
                if key in mentions
            }
        return result

    @staticmethod
    def _history_applies(entry: dict[str, Any], dream_date: date) -> bool:
        start = parse_date_bound(entry.get("start_date"), argument_name="start_date")
        end = parse_date_bound(entry.get("end_date"), argument_name="end_date")
        return (start is None or start <= dream_date) and (
            end is None or dream_date <= end
        )

    def _bounded(self, character: dict[str, Any]) -> dict[str, Any]:
        bounded = dict(character)
        truncated = False
        aliases = list(character.get("aliases", []))
        bounded["aliases"] = aliases[: self.max_aliases]
        if len(aliases) > self.max_aliases:
            bounded["aliases_omitted"] = len(aliases) - self.max_aliases
            truncated = True
        for field, maximum in (
            ("relationship", self.max_relationship_chars),
            ("context", self.max_context_chars),
        ):
            if isinstance(character.get(field), str) and len(character[field]) > maximum:
                bounded[field] = character[field][:maximum] + "\n[TRUNCATED]"
                truncated = True
        if "relationship_history" in character:
            history = []
            full_history = list(character["relationship_history"])
            for entry in full_history[: self.max_history_entries]:
                bounded_entry = dict(entry)
                for field, maximum in (
                    ("relationship", self.max_relationship_chars),
                    ("context", self.max_context_chars),
                ):
                    if len(bounded_entry.get(field, "")) > maximum:
                        bounded_entry[field] = (
                            bounded_entry[field][:maximum] + "\n[TRUNCATED]"
                        )
                        truncated = True
                history.append(bounded_entry)
            bounded["relationship_history"] = history
            if len(full_history) > self.max_history_entries:
                bounded["relationship_history_entries_omitted"] = (
                    len(full_history) - self.max_history_entries
                )
                truncated = True
        bounded["content_truncated"] = truncated
        return bounded

    @staticmethod
    def _warnings(
        *,
        missing_names: list[str],
        ambiguous_names: list[str],
        characters: list[dict[str, Any]],
        dream_date: date | None,
    ) -> list[str]:
        warnings: list[str] = []
        if missing_names:
            warnings.append(
                "No character dictionary entry was found for: "
                + ", ".join(missing_names)
                + "."
            )
        if ambiguous_names:
            warnings.append(
                "Multiple character entries matched: "
                + ", ".join(ambiguous_names)
                + ". Treat them as ambiguous."
            )
        if dream_date is not None:
            without_history = [
                character["name"]
                for character in characters
                if "relationship_history" in character
                and not character["relationship_history"]
            ]
            if without_history:
                warnings.append(
                    f"No relationship history applies on {dream_date.isoformat()} "
                    "for: " + ", ".join(without_history) + "."
                )
        return warnings


class CharacterMentionsTool:
    """Expose named-character mentions from structured dream records."""

    name = "get_character_mentions"
    max_names = 20
    max_name_chars = 100
    max_limit = 50
    max_dream_ids_per_character = 20

    def __init__(
        self,
        dream_repository: DreamCollectionRepository,
        structured_repository: StructuredDreamRecordRepository,
    ) -> None:
        self.dream_repository = dream_repository
        self.structured_repository = structured_repository

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Count explicit proper-name character mentions extracted from "
                    "structured dream records. Use this for who appears most often, "
                    "how often named characters appear, first or last appearances, "
                    "or named-character activity within an inclusive date range. "
                    "Results cover named_characters only and report structuring "
                    "coverage; they do not use manually edited character profiles."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "description": (
                                "Optional exact character names matched "
                                "case-insensitively. Omit to rank all names."
                            ),
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": self.max_name_chars,
                            },
                            "minItems": 1,
                            "maxItems": self.max_names,
                            "uniqueItems": True,
                        },
                        "start_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive lower date bound.",
                        },
                        "end_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Optional inclusive upper date bound.",
                        },
                        "minimum_mentions": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "Minimum distinct-dream mentions required."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self.max_limit,
                            "description": "Maximum ranked characters to return.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[AnalyticalToolResult, AnalyticalToolResult]:
        allowed = {
            "names",
            "start_date",
            "end_date",
            "minimum_mentions",
            "limit",
        }
        unexpected = sorted(set(arguments) - allowed)
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        names = self._names(arguments.get("names"))
        start_date = parse_date_bound(
            arguments.get("start_date"), argument_name="start_date"
        )
        end_date = parse_date_bound(
            arguments.get("end_date"), argument_name="end_date"
        )
        validate_date_range(start_date, end_date)
        minimum_mentions = self._positive_integer(
            arguments.get("minimum_mentions", 1),
            name="minimum_mentions",
        )
        limit = self._positive_integer(
            arguments.get("limit", 25),
            name="limit",
            maximum=self.max_limit,
        )
        parameters = {
            "names": names,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "minimum_mentions": minimum_mentions,
            "limit": limit,
        }

        dreams = self.dream_repository.all()
        current_by_id = {dream.dream_id: dream for dream in dreams}
        structured = self.structured_repository.all()
        current_structured = [
            record
            for record in structured
            if str(record.get("dream_id")) in current_by_id
        ]
        coverage = self._coverage(
            parsed_count=len(dreams),
            structured_count=len(structured),
            current_structured_count=len(current_structured),
        )

        selected_records: list[dict[str, Any]] = []
        excluded_unknown_dates = 0
        for record in current_structured:
            dream = current_by_id[str(record["dream_id"])]
            if start_date is not None or end_date is not None:
                if dream.date_sort is None:
                    excluded_unknown_dates += 1
                    continue
                if start_date is not None and dream.date_sort < start_date:
                    continue
                if end_date is not None and dream.date_sort > end_date:
                    continue
            normalized = dict(record)
            normalized["date_sort"] = (
                dream.date_sort.isoformat() if dream.date_sort else None
            )
            selected_records.append(normalized)

        characters = CharacterLookupService().aggregate(
            selected_records,
            temporal_context=False,
        )
        requested = {name.casefold() for name in names or []}
        available = {character["name"].casefold() for character in characters}
        if requested:
            characters = [
                character
                for character in characters
                if character["name"].casefold() in requested
            ]
        characters = [
            self._character_result(character)
            for character in characters
            if int(character["mentions"]["count"]) >= minimum_mentions
        ]
        characters.sort(
            key=lambda character: (
                -int(character["mentions"]["count"]),
                str(character["name"]).casefold(),
            )
        )
        character_count = len(characters)
        characters = characters[:limit]
        missing_names = [
            name for name in names or [] if name.casefold() not in available
        ]
        analysis = {
            "coverage": coverage,
            "selected_structured_dream_count": len(selected_records),
            "excluded_unknown_date_count": excluded_unknown_dates,
            "start_date": parameters["start_date"],
            "end_date": parameters["end_date"],
            "minimum_mentions": minimum_mentions,
            "character_count": character_count,
            "characters_omitted_by_limit": max(0, character_count - limit),
            "missing_names": missing_names,
            "characters": characters,
        }
        report_warnings = self._warnings(
            coverage=coverage,
            missing_names=missing_names,
            excluded_unknown_dates=excluded_unknown_dates,
            characters_omitted=max(0, character_count - limit),
        )
        report: AnalyticalToolResult = {
            "evidence_type": "character_mentions",
            "parameters": parameters,
            "analysis": analysis,
            "warnings": report_warnings,
        }

        bounded_characters = [
            self._bound_character(character) for character in characters
        ]
        bounded_analysis = {**analysis, "characters": bounded_characters}
        truncated_lists = sum(
            character["mentions"]["dream_ids_omitted"] > 0
            for character in bounded_characters
        )
        warnings = list(report_warnings)
        if truncated_lists:
            warnings.append(
                f"Dream ID lists are truncated for {truncated_lists} "
                "character(s) in model evidence; the saved report retains all IDs."
            )
        bounded: AnalyticalToolResult = {
            "evidence_type": "character_mentions",
            "parameters": parameters,
            "analysis": bounded_analysis,
            "warnings": warnings,
        }
        return bounded, report

    def _names(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list) or not value:
            raise ValueError("names must be a non-empty array of strings")
        if len(value) > self.max_names:
            raise ValueError(f"names cannot contain more than {self.max_names} items")
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("names must contain non-empty strings")
        names = [name.strip() for name in value]
        if any(len(name) > self.max_name_chars for name in names):
            raise ValueError(
                f"each name cannot exceed {self.max_name_chars} characters"
            )
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("names must not contain case-insensitive duplicates")
        return names

    @staticmethod
    def _positive_integer(
        value: Any,
        *,
        name: str,
        maximum: int | None = None,
    ) -> int:
        if type(value) is not int or value < 1 or (
            maximum is not None and value > maximum
        ):
            detail = "a positive integer"
            if maximum is not None:
                detail = f"between 1 and {maximum}"
            raise ValueError(f"{name} must be {detail}")
        return value

    @staticmethod
    def _coverage(
        *,
        parsed_count: int,
        structured_count: int,
        current_structured_count: int,
    ) -> dict[str, Any]:
        return {
            "parsed_dream_count": parsed_count,
            "structured_record_count": structured_count,
            "structured_current_dream_count": current_structured_count,
            "unstructured_dream_count": max(
                0, parsed_count - current_structured_count
            ),
            "orphaned_structured_record_count": max(
                0, structured_count - current_structured_count
            ),
            "coverage_percentage": round(
                (current_structured_count / parsed_count) * 100, 2
            )
            if parsed_count
            else 0.0,
        }

    @staticmethod
    def _character_result(character: dict[str, Any]) -> dict[str, Any]:
        mentions = dict(character["mentions"])
        return {
            "id": character["id"],
            "name": character["name"],
            "mentions": mentions,
        }

    def _bound_character(self, character: dict[str, Any]) -> dict[str, Any]:
        mentions = dict(character["mentions"])
        dream_ids = list(mentions["dream_ids"])
        mentions["dream_ids"] = dream_ids[: self.max_dream_ids_per_character]
        mentions["dream_ids_omitted"] = max(
            0, len(dream_ids) - self.max_dream_ids_per_character
        )
        return {**character, "mentions": mentions}

    @staticmethod
    def _warnings(
        *,
        coverage: dict[str, Any],
        missing_names: list[str],
        excluded_unknown_dates: int,
        characters_omitted: int,
    ) -> list[str]:
        warnings: list[str] = []
        if coverage["coverage_percentage"] < 100:
            warnings.append(
                "Character results cover "
                f"{coverage['structured_current_dream_count']} of "
                f"{coverage['parsed_dream_count']} current dreams "
                f"({coverage['coverage_percentage']}%). Unstructured dreams "
                "cannot contribute named-character mentions."
            )
        if coverage["orphaned_structured_record_count"]:
            warnings.append(
                f"{coverage['orphaned_structured_record_count']} structured "
                "record(s) do not match a current dream ID and were excluded."
            )
        if missing_names:
            warnings.append(
                "No structured mention was found in the selected range for: "
                + ", ".join(missing_names)
                + "."
            )
        if excluded_unknown_dates:
            warnings.append(
                f"{excluded_unknown_dates} structured current dream(s) with an "
                "unknown date were excluded by the requested date range."
            )
        if characters_omitted:
            warnings.append(
                f"{characters_omitted} character(s) were omitted by the requested "
                "limit."
            )
        return warnings


class DreamByIdTool:
    """Expose exact retrieval of one dream by its stable ID."""

    name = "get_dream_by_id"
    max_id_chars = 200

    def __init__(
        self,
        repository: DreamByIdRepository,
        *,
        max_chars_per_dream: int = 2500,
    ) -> None:
        if max_chars_per_dream < 1:
            raise ValueError("max_chars_per_dream must be positive")
        self.repository = repository
        self.max_chars_per_dream = max_chars_per_dream

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get one dream by its exact dream_id. Use this whenever the "
                    "user supplies a specific ID, including when they ask to "
                    "retrieve, summarize, or analyze that dream."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dream_id": {
                            "type": "string",
                            "description": (
                                "Exact stable dream ID, such as dream-2025-1-9-0."
                            ),
                            "minLength": 1,
                            "maxLength": self.max_id_chars,
                        }
                    },
                    "required": ["dream_id"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result, _ = self.execute_with_report_data(arguments)
        return result

    def execute_with_report_data(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unexpected = sorted(set(arguments) - {"dream_id"})
        if unexpected:
            raise ValueError(f"unexpected arguments: {', '.join(unexpected)}")
        dream_id = arguments.get("dream_id")
        if not isinstance(dream_id, str) or not dream_id.strip():
            raise ValueError("dream_id must be a non-empty string")
        dream_id = dream_id.strip()
        if len(dream_id) > self.max_id_chars:
            raise ValueError(
                f"dream_id cannot exceed {self.max_id_chars} characters"
            )

        dream = self.repository.get(dream_id)
        common = {"requested_dream_id": dream_id, "result_count": 1}
        return (
            {
                **common,
                "dreams": [self._result(dream, truncate=True)],
            },
            {
                **common,
                "dreams": [self._result(dream, truncate=False)],
            },
        )

    def _result(self, dream: Dream, *, truncate: bool) -> dict[str, Any]:
        text = dream.text
        truncated = truncate and len(text) > self.max_chars_per_dream
        if truncated:
            text = text[: self.max_chars_per_dream] + "\n[TRUNCATED]"
        return {
            "dream_id": dream.dream_id,
            "date": dream.date,
            "tags": list(dream.tags),
            "text": text,
            "truncated": truncated,
        }
