from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dream_analysis.cluster_reporting import render_report, write_report
from dream_analysis.clustering import (
    ClusteringSettings,
    DreamClusteringService,
    DreamEmbeddingCollection,
    DreamEmbeddingRepository,
    automatic_label,
    enriched_tags,
    representative_indices,
)


class FakeCollection:
    metadata = {"embedding_model": "test-embed"}

    def get(self, *, include: list[str]) -> dict:
        self.include = include
        return {
            "ids": ["dream-1", "dream-2"],
            "documents": [
                "DATE: 2024-01-01\n\n--- DREAM TEXT ---\n\nA hidden room.",
                "A school hallway.",
            ],
            "metadatas": [{"tags": "house"}, {"tags": "school"}],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
        }


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_collection(self, *, name: str) -> FakeCollection:
        self.name = name
        return self.collection


class ClusteringServiceTests(unittest.TestCase):
    def test_repository_loads_and_normalizes_chroma_records(self) -> None:
        client = FakeClient()

        loaded = DreamEmbeddingRepository(
            path="unused", collection_name="dreams", client=client
        ).load()

        self.assertEqual(client.name, "dreams")
        self.assertEqual(loaded.documents, ["A hidden room.", "A school hallway."])
        self.assertEqual(loaded.vectors.shape, (2, 2))
        self.assertEqual(loaded.metadata["embedding_model"], "test-embed")

    def test_evidence_helpers_are_deterministic(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        labels = np.asarray([0, 0, 1])
        metadatas = [
            {"tags": "house, room"},
            {"tags": "house"},
            {"tags": "school"},
        ]

        self.assertEqual(representative_indices(vectors, labels, 0, count=1), [0])
        self.assertEqual(enriched_tags(metadatas, labels, top_n=1)[0][0][0], "house")
        self.assertEqual(
            automatic_label(["hidden room"], [("house", 2.0, 2)]),
            "house / hidden room",
        )

    def test_service_returns_analysis_without_writing_files(self) -> None:
        collection = DreamEmbeddingCollection(
            name="dreams",
            ids=["one", "two", "three"],
            documents=["room room", "room door", "school class"],
            metadatas=[{"tags": "house"}, {"tags": "house"}, {"tags": "school"}],
            vectors=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            metadata={"embedding_model": "embed"},
        )
        projected = (
            np.asarray([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]]),
            np.asarray([0, 0, -1]),
            np.asarray([0.9, 0.8, 0.0]),
            2,
        )
        service = DreamClusteringService(
            settings=ClusteringSettings(
                pca_dimensions=2,
                min_cluster_size=2,
                top_terms=2,
                top_tags=2,
                representative_dreams=1,
            )
        )

        with patch("dream_analysis.clustering.project_and_cluster", return_value=projected):
            analysis = service.analyze(collection)

        self.assertEqual(analysis.cluster_labels[0], "house / room")
        self.assertEqual(analysis.representatives[0], [0])
        self.assertEqual(analysis.metrics["cluster_count"], 1)
        self.assertEqual(analysis.settings["pca_dimensions"], 2)

    def test_report_can_be_rendered_or_written(self) -> None:
        arguments = {
            "collection_name": "dreams",
            "collection_metadata": {"embedding_model": "embed"},
            "settings": {"pca_dimensions": 2},
            "metrics": {
                "dream_count": 2,
                "cluster_count": 1,
                "noise_count": 0,
                "noise_fraction": 0.0,
                "silhouette_cosine_non_noise": None,
            },
            "labels": np.asarray([0, 0]),
            "ids": ["one", "two"],
            "documents": ["A hidden room", "A secret door"],
            "metadatas": [{"date": "2024-01-01"}, {"date": "2024-01-02"}],
            "representatives": {0: [0]},
            "terms": {0: ["hidden room"]},
            "tags": {0: [("house", 2.0, 2)]},
            "cluster_labels": {0: "Hidden rooms"},
        }

        rendered = render_report(**arguments)
        self.assertIn("## Cluster 0: Hidden rooms", rendered)
        self.assertIn("**one** (2024-01-01): A hidden room", rendered)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, **arguments)
            self.assertEqual(path.read_text(encoding="utf-8"), rendered)


if __name__ == "__main__":
    unittest.main()
