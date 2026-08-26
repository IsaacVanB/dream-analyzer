"""Reusable services for the dream-analysis project."""

from dream_analysis.analysis import SingleDreamAnalysisService
from dream_analysis.config import Settings
from dream_analysis.dates import (
    filter_records_by_date,
    format_period_label,
    parse_date_bound,
    record_date,
    validate_date_range,
)
from dream_analysis.index import DreamIndex, EmbeddingModelMismatchError
from dream_analysis.models import Dream, DreamValidationError, RelatedDream, SearchResult
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.rag import DirectRagService
from dream_analysis.repository import (
    DreamNotFoundError,
    DreamRepository,
    load_jsonl_objects,
)

__all__ = [
    "Dream",
    "DreamIndex",
    "DreamNotFoundError",
    "DreamRepository",
    "DreamValidationError",
    "DirectRagService",
    "EmbeddingModelMismatchError",
    "filter_records_by_date",
    "format_period_label",
    "load_jsonl_objects",
    "OllamaGateway",
    "parse_date_bound",
    "RelatedDream",
    "record_date",
    "SearchResult",
    "Settings",
    "SingleDreamAnalysisService",
    "validate_date_range",
]
