"""Persistent Chroma vector index for dream journal entries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from dream_analysis.models import Dream, SearchResult
from dream_analysis.ollama_client import OllamaGateway


DREAM_TEXT_SEPARATOR = "--- DREAM TEXT ---"


class EmbeddingModelMismatchError(ValueError):
    """Raised when a query model differs from the collection's model."""


def build_document(dream: Dream) -> str:
    """Build the existing display document stored alongside each embedding."""
    tag_text = ", ".join(dream.tags)
    return (
        f"DREAM_ID: {dream.dream_id}\n"
        f"DATE: {dream.date}\n"
        f"YEAR: {dream.year if dream.year is not None else ''}\n"
        f"MONTH: {dream.month if dream.month is not None else ''}\n"
        f"DAY: {dream.day if dream.day is not None else ''}\n"
        f"DATE_PRECISION: {dream.date_precision}\n"
        f"TAGS: {tag_text}\n\n"
        f"{DREAM_TEXT_SEPARATOR}\n\n"
        f"{dream.text}"
    )


def build_metadata(dream: Dream) -> dict[str, str | int]:
    """Build Chroma-compatible metadata using the project's existing shape."""
    return {
        "date": dream.date,
        "year": dream.year or 0,
        "month": dream.month or 0,
        "day": dream.day or 0,
        "date_precision": dream.date_precision,
        "date_sort": dream.date_sort.isoformat() if dream.date_sort else "",
        "tags": ", ".join(dream.tags),
        "word_count": dream.word_count,
    }


def validate_collection_embedding_model(
    collection: Any,
    *,
    collection_name: str,
    embedding_model: str,
) -> None:
    """Ensure queries use the model that produced stored embeddings."""
    metadata = collection.metadata or {}
    indexed_model = metadata.get("embedding_model")
    if indexed_model == embedding_model:
        return

    detail = (
        "does not record an embedding model"
        if indexed_model is None
        else f"was built with embedding model {indexed_model!r}"
    )
    raise EmbeddingModelMismatchError(
        f"ChromaDB collection {collection_name!r} {detail}, but the requested "
        f"embedding model is {embedding_model!r}. Rebuild a separate collection "
        "with matching collection and embedding model values."
    )


class DreamIndex:
    """Build and query one embedding-model-specific Chroma collection."""

    def __init__(
        self,
        *,
        path: Path | str,
        collection_name: str,
        embedding_model: str,
        ollama_gateway: OllamaGateway,
        client: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        if not embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.ollama = ollama_gateway
        self.client = client or chromadb.PersistentClient(path=self.path)

    def rebuild(
        self,
        dreams: Sequence[Dream],
        *,
        batch_size: int = 32,
        progress: Callable[[int, int, Dream], None] | None = None,
    ) -> int:
        """Embed all dreams, then replace the logical collection.

        Embeddings are completed before the existing collection is removed. A
        failed Ollama request therefore leaves the current collection intact.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dream_list = list(dreams)
        ids = [dream.dream_id for dream in dream_list]
        if len(ids) != len(set(ids)):
            raise ValueError("dream IDs must be unique")

        embeddings: list[list[float]] = []
        for offset in range(0, len(dream_list), batch_size):
            batch = dream_list[offset : offset + batch_size]
            if progress is not None:
                for index, dream in enumerate(batch, start=offset + 1):
                    progress(index, len(dream_list), dream)
            embeddings.extend(
                self.ollama.embed_many(
                    [dream.text for dream in batch],
                    model=self.embedding_model,
                )
            )

        collection = self._recreate_collection()
        if dream_list:
            collection.add(
                ids=ids,
                documents=[build_document(dream) for dream in dream_list],
                metadatas=[build_metadata(dream) for dream in dream_list],
                embeddings=embeddings,
            )
        return len(dream_list)

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError("limit must be positive")

        collection = self.client.get_collection(name=self.collection_name)
        validate_collection_embedding_model(
            collection,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        result_count = min(limit, collection.count())
        if result_count == 0:
            return []

        query_embedding = self.ollama.embed_one(
            query,
            model=self.embedding_model,
        )
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(
                dream_id=str(dream_id),
                document=str(document or ""),
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for dream_id, document, metadata, distance in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]

    def _recreate_collection(self) -> Any:
        try:
            self.client.delete_collection(self.collection_name)
        except (NotFoundError, ValueError):
            pass
        return self.client.create_collection(
            name=self.collection_name,
            metadata={
                "embedding_source": "dream_text",
                "embedding_model": self.embedding_model,
            },
        )
