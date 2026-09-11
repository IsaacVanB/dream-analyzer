"""Persistent Chroma vector index for dream journal entries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from dream_analysis.dates import record_date, parse_date_value, validate_date_range
from dream_analysis.imports import dream_text_hash
from dream_analysis.models import Dream, RelatedDream, SearchResult
from dream_analysis.ollama_client import OllamaGateway


DREAM_TEXT_SEPARATOR = "--- DREAM TEXT ---"


class EmbeddingModelMismatchError(ValueError):
    """Raised when a query model differs from the collection's model."""


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for two equal-length vectors."""
    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match.")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def extract_dream_text(document: str) -> str:
    """Extract raw dream text from a formatted Chroma document."""
    _, separator, dream_text = document.partition(DREAM_TEXT_SEPARATOR)
    return (dream_text if separator else document).strip()


def parse_index_date(value: Any) -> date | None:
    """Parse a Chroma ISO date, returning None for missing/invalid values."""
    return parse_date_value(value)


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


def build_metadata(
    dream: Dream,
    *,
    embedded_text_hash: str | None = None,
) -> dict[str, str | int]:
    """Build Chroma-compatible metadata using the project's existing shape."""
    current_text_hash = dream_text_hash(dream.text)
    return {
        "date": dream.date,
        "year": dream.year or 0,
        "month": dream.month or 0,
        "day": dream.day or 0,
        "date_precision": dream.date_precision,
        "date_sort": dream.date_sort.isoformat() if dream.date_sort else "",
        "tags": ", ".join(dream.tags),
        "word_count": dream.word_count,
        "current_text_hash": current_text_hash,
        "embedded_text_hash": embedded_text_hash or current_text_hash,
    }


@dataclass(frozen=True, slots=True)
class IndexSyncResult:
    """Summary of an incremental Chroma synchronization."""

    embedded: int
    renamed: int
    updated: int
    unchanged: int
    pruned: int
    orphaned_ids: tuple[str, ...]


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

    def sync(
        self,
        dreams: Sequence[Dream],
        *,
        batch_size: int = 32,
        prune: bool = False,
        progress: Callable[[int, int, Dream], None] | None = None,
    ) -> IndexSyncResult:
        """Add missing dreams and refresh documents without re-embedding old IDs.

        Existing embeddings are deliberately retained when their source text has
        received a correction. Metadata records both the current text hash and
        the hash of the text that produced the retained vector.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dream_list = list(dreams)
        ids = [dream.dream_id for dream in dream_list]
        if len(ids) != len(set(ids)):
            raise ValueError("dream IDs must be unique")

        try:
            collection = self.client.get_collection(name=self.collection_name)
            validate_collection_embedding_model(
                collection,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
            )
        except (NotFoundError, ValueError) as exc:
            if isinstance(exc, EmbeddingModelMismatchError):
                raise
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "embedding_source": "dream_text",
                    "embedding_model": self.embedding_model,
                },
            )

        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        stored_embeddings = stored.get("embeddings")
        if stored_embeddings is None:
            raise ValueError("The ChromaDB collection contains no embeddings.")
        stored_by_id = {
            str(dream_id): (
                str(document or ""),
                dict(metadata or {}),
                [float(value) for value in embedding],
            )
            for dream_id, document, metadata, embedding in zip(
                stored["ids"],
                stored.get("documents") or [],
                stored.get("metadatas") or [],
                stored_embeddings,
            )
        }
        current_ids = set(ids)
        orphaned_ids = set(stored_by_id) - current_ids
        orphan_ids_by_text_hash: dict[str, list[str]] = {}
        for orphan_id in orphaned_ids:
            orphan_document = stored_by_id[orphan_id][0]
            text_hash = dream_text_hash(extract_dream_text(orphan_document))
            orphan_ids_by_text_hash.setdefault(text_hash, []).append(orphan_id)

        missing = [dream for dream in dream_list if dream.dream_id not in stored_by_id]
        renamed = 0
        renamed_ids: set[str] = set()
        consumed_orphans: set[str] = set()
        for dream in missing:
            matches = orphan_ids_by_text_hash.get(dream_text_hash(dream.text), [])
            available = [item for item in matches if item not in consumed_orphans]
            if len(available) != 1:
                continue
            orphan_id = available[0]
            old_document, old_metadata, old_embedding = stored_by_id[orphan_id]
            embedded_hash = str(
                old_metadata.get("embedded_text_hash")
                or dream_text_hash(extract_dream_text(old_document))
            )
            collection.add(
                ids=[dream.dream_id],
                documents=[build_document(dream)],
                metadatas=[
                    build_metadata(dream, embedded_text_hash=embedded_hash)
                ],
                embeddings=[old_embedding],
            )
            collection.delete(ids=[orphan_id])
            renamed_ids.add(dream.dream_id)
            consumed_orphans.add(orphan_id)
            renamed += 1

        missing = [dream for dream in missing if dream.dream_id not in renamed_ids]
        updated = 0
        unchanged = 0

        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            if progress is not None:
                for index, dream in enumerate(batch, start=offset + 1):
                    progress(index, len(missing), dream)
            embeddings = self.ollama.embed_many(
                [dream.text for dream in batch], model=self.embedding_model
            )
            collection.add(
                ids=[dream.dream_id for dream in batch],
                documents=[build_document(dream) for dream in batch],
                metadatas=[build_metadata(dream) for dream in batch],
                embeddings=embeddings,
            )

        for dream in dream_list:
            existing = stored_by_id.get(dream.dream_id)
            if existing is None:
                continue
            old_document, old_metadata, old_embedding = existing
            embedded_hash = str(
                old_metadata.get("embedded_text_hash")
                or dream_text_hash(extract_dream_text(old_document))
            )
            document = build_document(dream)
            metadata = build_metadata(dream, embedded_text_hash=embedded_hash)
            if old_document == document and old_metadata == metadata:
                unchanged += 1
                continue
            collection.update(
                ids=[dream.dream_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[old_embedding],
            )
            updated += 1

        remaining_orphans = orphaned_ids - consumed_orphans
        pruned = 0
        if prune and remaining_orphans:
            collection.delete(ids=sorted(remaining_orphans))
            pruned = len(remaining_orphans)
            remaining_orphans.clear()
        orphaned = tuple(sorted(remaining_orphans))
        return IndexSyncResult(
            embedded=len(missing),
            renamed=renamed,
            updated=updated,
            unchanged=unchanged,
            pruned=pruned,
            orphaned_ids=orphaned,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        """Return semantic matches inside an optional inclusive date range."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        if limit < 1:
            raise ValueError("limit must be positive")
        validate_date_range(start_date, end_date)

        collection = self.client.get_collection(name=self.collection_name)
        validate_collection_embedding_model(
            collection,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        collection_count = collection.count()
        date_filtered = start_date is not None or end_date is not None
        # Chroma stores date_sort as an ISO string, while its range operators
        # accept numeric operands. Ask it to rank the full collection when a
        # date window is present, then apply the exact range locally. The final
        # result remains bounded by limit and correctly ranked within the range.
        result_count = (
            collection_count if date_filtered else min(limit, collection_count)
        )
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
        matches: list[SearchResult] = []
        for dream_id, document, metadata, distance in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            metadata = dict(metadata or {})
            if date_filtered:
                indexed_date = record_date(metadata)
                if indexed_date is None:
                    continue
                if start_date is not None and indexed_date < start_date:
                    continue
                if end_date is not None and indexed_date > end_date:
                    continue
            matches.append(
                SearchResult(
                    dream_id=str(dream_id),
                    document=str(document or ""),
                    metadata=metadata,
                    distance=float(distance),
                )
            )
            if len(matches) == limit:
                break
        return matches

    def rank_ids(
        self,
        query: str,
        dream_ids: Sequence[str],
    ) -> list[SearchResult]:
        """Semantically rank a specified set of indexed dream IDs."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")
        allowed_ids = {str(dream_id) for dream_id in dream_ids}
        if not allowed_ids:
            return []

        collection = self.client.get_collection(name=self.collection_name)
        validate_collection_embedding_model(
            collection,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        collection_count = collection.count()
        if collection_count == 0:
            return []
        query_embedding = self.ollama.embed_one(
            query.strip(),
            model=self.embedding_model,
        )
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=collection_count,
            include=["documents", "metadatas", "distances"],
        )
        matches: list[SearchResult] = []
        for dream_id, document, metadata, distance in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            if str(dream_id) not in allowed_ids:
                continue
            matches.append(
                SearchResult(
                    dream_id=str(dream_id),
                    document=str(document or ""),
                    metadata=dict(metadata or {}),
                    distance=float(distance),
                )
            )
        return matches

    def related(
        self,
        text: str,
        *,
        limit: int,
        similarity_threshold: float = 0.5,
        target_dream_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RelatedDream]:
        """Return the most cosine-similar dreams meeting optional filters."""
        if limit < 0:
            raise ValueError("limit cannot be negative")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        validate_date_range(start_date, end_date)
        if limit == 0:
            return []
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text cannot be empty")

        collection = self.client.get_collection(name=self.collection_name)
        validate_collection_embedding_model(
            collection,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        records = collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = records.get("embeddings")
        if embeddings is None:
            raise ValueError("The ChromaDB collection contains no embeddings.")

        target_embedding: list[float] | None = None
        if target_dream_id is not None:
            for indexed_id, embedding in zip(records["ids"], embeddings):
                if indexed_id == target_dream_id:
                    target_embedding = [float(value) for value in embedding]
                    break
        if target_embedding is None:
            target_embedding = self.ollama.embed_one(
                text,
                model=self.embedding_model,
            )

        related: list[RelatedDream] = []
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        for dream_id, document, metadata, embedding in zip(
            records["ids"], documents, metadatas, embeddings
        ):
            if dream_id == target_dream_id:
                continue
            metadata = metadata or {}
            related_date = parse_index_date(metadata.get("date_sort"))
            if start_date is not None or end_date is not None:
                if related_date is None:
                    continue
                if start_date is not None and related_date < start_date:
                    continue
                if end_date is not None and related_date > end_date:
                    continue

            similarity = cosine_similarity(
                target_embedding,
                [float(value) for value in embedding],
            )
            if similarity < similarity_threshold:
                continue
            related.append(
                RelatedDream(
                    dream_id=str(dream_id),
                    date=str(metadata.get("date", "unknown")),
                    similarity=similarity,
                    text=extract_dream_text(str(document or "")),
                )
            )

        related.sort(key=lambda item: item.similarity, reverse=True)
        return related[:limit]

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
