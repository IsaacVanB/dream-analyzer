"""Reusable services for the dream-analysis project."""

from dream_analysis.analysis import SingleDreamAnalysisService
from dream_analysis.agent import AgentResponse, DreamRagAgent
from dream_analysis.artifacts import write_json_atomic, write_text_atomic
from dream_analysis.characters import CharacterLookupService
from dream_analysis.config import Settings
from dream_analysis.dates import (
    filter_records_by_date,
    filter_dreams_by_date,
    format_period_label,
    parse_date_bound,
    period_start,
    record_date,
    validate_date_range,
)
from dream_analysis.index import DreamIndex, EmbeddingModelMismatchError
from dream_analysis.models import Dream, DreamValidationError, RelatedDream, SearchResult
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.parser import JournalParser
from dream_analysis.rag import DirectRagService
from dream_analysis.repository import (
    DreamNotFoundError,
    DreamRepository,
    load_jsonl_objects,
)
from dream_analysis.statistics import DreamStatisticsService
from dream_analysis.trends import TagTrendService
from dream_analysis.tools import DreamSearchTool

__all__ = [
    "Dream",
    "DreamRagAgent",
    "DreamSearchTool",
    "DreamIndex",
    "DreamNotFoundError",
    "DreamRepository",
    "DreamValidationError",
    "DreamStatisticsService",
    "DirectRagService",
    "EmbeddingModelMismatchError",
    "filter_records_by_date",
    "filter_dreams_by_date",
    "format_period_label",
    "load_jsonl_objects",
    "OllamaGateway",
    "JournalParser",
    "parse_date_bound",
    "period_start",
    "RelatedDream",
    "record_date",
    "SearchResult",
    "Settings",
    "SingleDreamAnalysisService",
    "TagTrendService",
    "AgentResponse",
    "validate_date_range",
    "CharacterLookupService",
    "write_json_atomic",
    "write_text_atomic",
]
