from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from dream_analysis.index import (
    DREAM_TEXT_SEPARATOR,
    DreamIndex,
    EmbeddingModelMismatchError,
    build_document,
    build_metadata,
)
from dream_analysis.models import Dream


class FakeEmbeddingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def embed_many(self, texts, *, model=None):
        values = list(texts)
        self.calls.append((values, model))
        return [self.vector(text) for text in values]

    def embed_one(self, text, *, model=None):
        self.calls.append(([text], model))
        return self.vector(text)

    @staticmethod
    def vector(text: str) -> list[float]:
        return [float(text.count("room")), float(text.count("school")), 1.0]


def make_dream(dream_id: str, text: str, day: int) -> Dream:
    return Dream(
        dream_id=dream_id,
        date=f"1/{day}/2024",
        text=text,
        tags=("test",),
        year=2024,
        month=1,
        day=day,
        date_precision="day",
        date_sort=date(2024, 1, day),
        word_count=len(text.split()),
    )


class DreamIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "chroma"
        self.gateway = FakeEmbeddingGateway()
        self.index = DreamIndex(
            path=self.path,
            collection_name="test_dreams",
            embedding_model="test-embed",
            ollama_gateway=self.gateway,
        )
        self.dreams = [
            make_dream("room-dream", "hidden room room", 1),
            make_dream("school-dream", "late for school", 2),
        ]

    def test_document_and_metadata_preserve_existing_index_shape(self) -> None:
        document = build_document(self.dreams[0])
        metadata = build_metadata(self.dreams[0])

        self.assertIn(DREAM_TEXT_SEPARATOR, document)
        self.assertTrue(document.endswith("hidden room room"))
        self.assertEqual(metadata["date_sort"], "2024-01-01")
        self.assertEqual(metadata["tags"], "test")

    def test_rebuild_batches_embeddings_and_search_returns_typed_results(self) -> None:
        progress: list[tuple[int, int, str]] = []
        count = self.index.rebuild(
            self.dreams,
            batch_size=1,
            progress=lambda current, total, dream: progress.append(
                (current, total, dream.dream_id)
            ),
        )
        matches = self.index.search("room", limit=10)

        self.assertEqual(count, 2)
        self.assertEqual(len(self.gateway.calls), 3)
        self.assertEqual(progress[-1], (2, 2, "school-dream"))
        self.assertEqual(matches[0].dream_id, "room-dream")
        self.assertEqual(matches[0].date, "1/1/2024")
        self.assertEqual(len(matches), 2)

    def test_search_rejects_mismatched_embedding_model(self) -> None:
        self.index.rebuild(self.dreams)
        other_index = DreamIndex(
            path=self.path,
            collection_name="test_dreams",
            embedding_model="different-embed",
            ollama_gateway=self.gateway,
        )

        with self.assertRaises(EmbeddingModelMismatchError):
            other_index.search("room")

    def test_search_applies_inclusive_date_bounds_before_limiting(self) -> None:
        self.index.rebuild(self.dreams)

        matches = self.index.search(
            "room",
            limit=1,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )

        self.assertEqual([item.dream_id for item in matches], ["school-dream"])

    def test_search_rejects_reversed_date_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_date"):
            self.index.search(
                "room",
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 1),
            )

    def test_related_reuses_target_embedding_and_applies_filters(self) -> None:
        self.index.rebuild(self.dreams)
        self.gateway.calls.clear()

        related = self.index.related(
            self.dreams[0].text,
            limit=5,
            similarity_threshold=0.3,
            target_dream_id="room-dream",
            start_date=date(2024, 1, 2),
        )

        self.assertEqual([item.dream_id for item in related], ["school-dream"])
        self.assertEqual(related[0].text, "late for school")
        self.assertEqual(self.gateway.calls, [])

    def test_related_embeds_direct_text_when_target_is_not_indexed(self) -> None:
        self.index.rebuild(self.dreams)
        self.gateway.calls.clear()

        related = self.index.related(
            "room",
            limit=1,
            similarity_threshold=0.8,
        )

        self.assertEqual(related[0].dream_id, "room-dream")
        self.assertEqual(self.gateway.calls, [(["room"], "test-embed")])

    def test_sync_embeds_only_missing_ids_and_reuses_edited_embedding(self) -> None:
        self.index.rebuild(self.dreams)
        original = self.index.client.get_collection("test_dreams").get(
            ids=["room-dream"], include=["metadatas"]
        )["metadatas"][0]
        edited = make_dream("room-dream", "hidden room rooms", 1)
        added = make_dream("new-dream", "a new school room", 3)
        self.gateway.calls.clear()

        result = self.index.sync([edited, self.dreams[1], added], batch_size=10)

        self.assertEqual(result.embedded, 1)
        self.assertEqual(result.updated, 1)
        self.assertEqual(self.gateway.calls, [([added.text], "test-embed")])
        metadata = self.index.client.get_collection("test_dreams").get(
            ids=["room-dream"], include=["metadatas"]
        )["metadatas"][0]
        self.assertEqual(metadata["embedded_text_hash"], original["embedded_text_hash"])
        self.assertNotEqual(metadata["current_text_hash"], metadata["embedded_text_hash"])

    def test_sync_migrates_exact_text_to_a_corrected_id_without_embedding(self) -> None:
        old = make_dream("dream-2024-1-1-0", "same remembered dream", 1)
        corrected = make_dream("dream-2024-1-2-0", "same remembered dream", 2)
        self.index.rebuild([old])
        self.gateway.calls.clear()

        result = self.index.sync([corrected])

        self.assertEqual(result.embedded, 0)
        self.assertEqual(result.renamed, 1)
        self.assertEqual(result.orphaned_ids, ())
        self.assertEqual(self.gateway.calls, [])
        ids = self.index.client.get_collection("test_dreams").get()["ids"]
        self.assertEqual(ids, [corrected.dream_id])

    def test_sync_prunes_unmatched_orphans_only_when_requested(self) -> None:
        self.index.rebuild(self.dreams)

        retained = self.index.sync([self.dreams[0]])
        pruned = self.index.sync([self.dreams[0]], prune=True)

        self.assertEqual(retained.orphaned_ids, ("school-dream",))
        self.assertEqual(retained.pruned, 0)
        self.assertEqual(pruned.pruned, 1)
        self.assertEqual(pruned.orphaned_ids, ())


if __name__ == "__main__":
    unittest.main()
