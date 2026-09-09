"""Small, read-only tools that an Ollama agent may call."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from dream_analysis.dates import parse_date_bound, validate_date_range
from dream_analysis.models import Dream, SearchResult


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
