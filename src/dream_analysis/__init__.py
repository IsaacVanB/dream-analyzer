"""Reusable services for the dream-analysis project."""

from dream_analysis.config import Settings
from dream_analysis.index import DreamIndex, EmbeddingModelMismatchError
from dream_analysis.models import Dream, DreamValidationError, SearchResult
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.repository import DreamNotFoundError, DreamRepository

__all__ = [
    "Dream",
    "DreamIndex",
    "DreamNotFoundError",
    "DreamRepository",
    "DreamValidationError",
    "EmbeddingModelMismatchError",
    "OllamaGateway",
    "SearchResult",
    "Settings",
]
