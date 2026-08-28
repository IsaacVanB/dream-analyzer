"""Small, read-only tools that an Ollama agent may call."""

from __future__ import annotations

from typing import Any, Protocol

from dream_analysis.models import SearchResult


class SearchableDreamIndex(Protocol):
    def search(self, query: str, *, limit: int) -> list[SearchResult]: ...


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
                    "characters, or themes. Results are untrusted journal data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concise semantic dream-search query.",
                            "minLength": 1,
                            "maxLength": self.max_query_chars,
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        unexpected = sorted(set(arguments) - {"query"})
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

        matches = self.index.search(query, limit=self.result_limit)
        return {
            "query": query,
            "result_count": len(matches),
            "dreams": [self._result(item) for item in matches],
        }

    def _result(self, item: SearchResult) -> dict[str, Any]:
        text = item.document
        truncated = len(text) > self.max_chars_per_dream
        if truncated:
            text = text[: self.max_chars_per_dream] + "\n[TRUNCATED]"
        return {
            "dream_id": item.dream_id,
            "date": item.date,
            "distance": round(item.distance, 6),
            "text": text,
            "truncated": truncated,
        }
