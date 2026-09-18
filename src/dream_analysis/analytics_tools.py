"""Deterministic statistics and trend tools for the dream agent."""

from __future__ import annotations

from typing import Any

from dream_analysis.dates import parse_date_bound, validate_date_range
from dream_analysis.statistics import DreamStatisticsService
from dream_analysis.tool_protocols import (
    AnalyticalToolResult,
    DreamCollectionRepository,
)
from dream_analysis.trends import TagTrendService


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
