"""Shared protocols and result types for dream-agent tools."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, TypedDict

from dream_analysis.bm25 import Bm25SearchResult
from dream_analysis.models import Dream, SearchResult


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

    def rank_ids(
        self,
        query: str,
        dream_ids: list[str] | tuple[str, ...],
    ) -> list[SearchResult]: ...


class SearchableDreamKeywordIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[Bm25SearchResult]: ...


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
