"""Reusable services for clustering stored dream embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from dream_analysis.index import extract_dream_text
from dream_analysis.ollama_client import OllamaGateway


@dataclass(frozen=True, slots=True)
class DreamEmbeddingCollection:
    """Dream documents, metadata, and embeddings loaded from one collection."""

    name: str
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    vectors: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClusteringSettings:
    """Numerical and evidence-selection settings for a clustering run."""

    pca_dimensions: int = 50
    n_neighbors: int = 15
    min_dist: float = 0.1
    min_cluster_size: int = 5
    min_samples: int | None = 1
    random_state: int = 42
    top_terms: int = 8
    top_tags: int = 5
    representative_dreams: int = 3

    def validate(self) -> None:
        if self.pca_dimensions < 2:
            raise ValueError("pca_dimensions must be at least 2")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if self.min_samples is not None and self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if self.top_terms < 0 or self.top_tags < 0:
            raise ValueError("top_terms and top_tags cannot be negative")
        if self.representative_dreams < 1:
            raise ValueError("representative_dreams must be at least 1")


@dataclass(frozen=True, slots=True)
class ClusterAnalysis:
    """Complete computed result, independent of any output format."""

    collection: DreamEmbeddingCollection
    coordinates: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    representatives: dict[int, list[int]]
    terms: dict[int, list[str]]
    tags: dict[int, list[tuple[str, float, int]]]
    cluster_labels: dict[int, str]
    metrics: dict[str, Any]
    settings: dict[str, Any]


class DreamEmbeddingRepository:
    """Load clustering input from Chroma without exposing Chroma to the CLI."""

    def __init__(
        self,
        *,
        path: Path | str,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.client = client

    def load(self) -> DreamEmbeddingCollection:
        client = self.client or chromadb.PersistentClient(path=str(self.path))
        collection = client.get_collection(name=self.collection_name)
        records = collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = records.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            raise ValueError(
                f"Collection {self.collection_name!r} contains no embeddings."
            )

        ids = [str(value) for value in records["ids"]]
        documents = [
            extract_dream_text(value or "") for value in records["documents"]
        ]
        metadatas = [value or {} for value in records["metadatas"]]
        vectors = np.asarray(embeddings, dtype=np.float64)
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise ValueError(
                "Stored embeddings do not form a dream-by-dimension matrix."
            )
        return DreamEmbeddingCollection(
            name=self.collection_name,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            vectors=vectors,
            metadata=collection.metadata or {},
        )


def load_collection(
    *, chroma_path: str, collection_name: str
) -> tuple[list[str], list[str], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Compatibility wrapper returning the original tuple-based result."""
    loaded = DreamEmbeddingRepository(
        path=chroma_path, collection_name=collection_name
    ).load()
    return (
        loaded.ids,
        loaded.documents,
        loaded.metadatas,
        loaded.vectors,
        loaded.metadata,
    )


def project_and_cluster(
    vectors: np.ndarray,
    *,
    pca_dimensions: int,
    n_neighbors: int,
    min_dist: float,
    min_cluster_size: int,
    min_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize, reduce, project, and cluster an embedding matrix."""
    numba_cache_dir = Path("/tmp/dream_analysis_numba")
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir.resolve()))
    try:
        import hdbscan
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "Clustering dependencies are missing. Reinstall dream-analysis."
        ) from exc

    if vectors.shape[0] < 3:
        raise ValueError("At least three dreams are required for clustering.")
    normalized = normalize(vectors, norm="l2")
    pca_count = max(2, min(pca_dimensions, *normalized.shape))
    reduced = PCA(n_components=pca_count, random_state=random_state).fit_transform(
        normalized
    )
    neighbors = max(2, min(n_neighbors, len(reduced) - 1))
    coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(reduced)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    return coordinates, labels, probabilities, pca_count


def parse_tags(metadata: dict[str, Any]) -> list[str]:
    """Return normalized tags from Chroma metadata."""
    raw = metadata.get("tags", "")
    if isinstance(raw, list):
        return [str(tag).strip() for tag in raw if str(tag).strip()]
    return [tag.strip() for tag in str(raw).split(",") if tag.strip()]


def representative_indices(
    vectors: np.ndarray, labels: np.ndarray, cluster: int, *, count: int
) -> list[int]:
    """Select cluster members closest to its cosine centroid."""
    members = np.flatnonzero(labels == cluster)
    member_vectors = normalize(vectors[members], norm="l2")
    centroid = normalize(member_vectors.mean(axis=0).reshape(1, -1))[0]
    similarities = member_vectors @ centroid
    return members[np.argsort(-similarities)[:count]].tolist()


def distinctive_terms(
    documents: list[str], labels: np.ndarray, *, top_n: int
) -> dict[int, list[str]]:
    """Find terms whose mean TF-IDF is elevated within each cluster."""
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {int(cluster): [] for cluster in sorted(set(labels)) if cluster >= 0}
    terms = np.asarray(vectorizer.get_feature_names_out())
    global_mean = np.asarray(matrix.mean(axis=0)).ravel()
    output: dict[int, list[str]] = {}
    for cluster in sorted(set(labels)):
        if cluster < 0:
            continue
        cluster_mean = np.asarray(matrix[labels == cluster].mean(axis=0)).ravel()
        scores = cluster_mean - global_mean
        output[int(cluster)] = terms[np.argsort(-scores)[:top_n]].tolist()
    return output


def enriched_tags(
    metadatas: list[dict[str, Any]], labels: np.ndarray, *, top_n: int
) -> dict[int, list[tuple[str, float, int]]]:
    """Rank tags by prevalence in a cluster relative to the collection."""
    all_counts = Counter(
        tag for metadata in metadatas for tag in set(parse_tags(metadata))
    )
    output: dict[int, list[tuple[str, float, int]]] = {}
    total = len(metadatas)
    for cluster in sorted(set(labels)):
        if cluster < 0:
            continue
        members = np.flatnonzero(labels == cluster)
        counts = Counter(
            tag for index in members for tag in set(parse_tags(metadatas[index]))
        )
        scored = []
        for tag, count in counts.items():
            baseline = all_counts[tag] / total
            enrichment = (count / len(members)) / baseline if baseline else 0.0
            scored.append((tag, enrichment, count))
        output[int(cluster)] = sorted(
            scored, key=lambda row: (-row[1], -row[2], row[0])
        )[:top_n]
    return output


def automatic_label(terms: list[str], tags: list[tuple[str, float, int]]) -> str:
    """Create a deterministic evidence-based label."""
    evidence = [tag for tag, _, _ in tags[:2]] + terms[:3]
    return " / ".join(dict.fromkeys(evidence)) or "unlabeled theme"


def llm_label(
    *,
    model: str,
    terms: list[str],
    tags: list[tuple[str, float, int]],
    texts: list[str],
    gateway: OllamaGateway | None = None,
) -> str:
    """Ask Ollama for a short content-only label for one cluster."""
    excerpts = "\n\n".join(text[:700] for text in texts)
    prompt = f"""Give this dream cluster a short, descriptive theme label of 2-7 words.
Do not diagnose or infer hidden psychological meaning. Describe only recurring content.
Distinctive terms: {', '.join(terms)}
Overrepresented tags: {', '.join(tag for tag, _, _ in tags) or 'none'}
Representative dreams:
{excerpts}
Return only the label."""
    ollama_gateway = gateway or OllamaGateway()
    response = ollama_gateway.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 30},
    )
    content = ollama_gateway.message_content(response).strip()
    label = content.splitlines()[0].strip().strip('"') if content else ""
    return label[:100] or automatic_label(terms, tags)


def quality_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Calculate high-level clustering quality statistics."""
    clusters = sorted(cluster for cluster in set(labels.tolist()) if cluster >= 0)
    non_noise = labels >= 0
    silhouette: float | None = None
    if len(clusters) >= 2 and non_noise.sum() > len(clusters):
        silhouette = float(
            silhouette_score(
                normalize(vectors[non_noise]),
                labels[non_noise],
                metric="cosine",
            )
        )
    return {
        "dream_count": int(len(labels)),
        "cluster_count": len(clusters),
        "noise_count": int((labels < 0).sum()),
        "noise_fraction": float((labels < 0).mean()),
        "silhouette_cosine_non_noise": silhouette,
    }


class DreamClusteringService:
    """Run clustering, evidence extraction, and optional Ollama labeling."""

    def __init__(
        self,
        *,
        settings: ClusteringSettings,
        label_clusters: bool = False,
        label_model: str = "qwen3:8b",
        ollama_gateway: OllamaGateway | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.label_clusters = label_clusters
        self.label_model = label_model
        self.ollama_gateway = ollama_gateway

    def analyze(self, collection: DreamEmbeddingCollection) -> ClusterAnalysis:
        """Compute a complete cluster analysis without writing artifacts."""
        settings = self.settings
        coordinates, labels, probabilities, actual_pca = project_and_cluster(
            collection.vectors,
            pca_dimensions=settings.pca_dimensions,
            n_neighbors=settings.n_neighbors,
            min_dist=settings.min_dist,
            min_cluster_size=settings.min_cluster_size,
            min_samples=settings.min_samples,
            random_state=settings.random_state,
        )
        terms = distinctive_terms(
            collection.documents, labels, top_n=settings.top_terms
        )
        tags = enriched_tags(collection.metadatas, labels, top_n=settings.top_tags)
        representatives = {
            int(cluster): representative_indices(
                collection.vectors,
                labels,
                int(cluster),
                count=settings.representative_dreams,
            )
            for cluster in sorted(set(labels))
            if cluster >= 0
        }

        gateway = self.ollama_gateway
        if self.label_clusters and gateway is None:
            gateway = OllamaGateway()
        cluster_labels = {}
        for cluster, indices in representatives.items():
            cluster_labels[cluster] = (
                llm_label(
                    model=self.label_model,
                    terms=terms[cluster],
                    tags=tags[cluster],
                    texts=[collection.documents[index] for index in indices],
                    gateway=gateway,
                )
                if self.label_clusters
                else automatic_label(terms[cluster], tags[cluster])
            )

        recorded_settings = {
            "pca_dimensions": actual_pca,
            "umap_neighbors": settings.n_neighbors,
            "umap_min_dist": settings.min_dist,
            "min_cluster_size": settings.min_cluster_size,
            "min_samples": settings.min_samples,
            "random_state": settings.random_state,
            "llm_labels": self.label_clusters,
            "label_model": self.label_model if self.label_clusters else None,
        }
        return ClusterAnalysis(
            collection=collection,
            coordinates=coordinates,
            labels=labels,
            probabilities=probabilities,
            representatives=representatives,
            terms=terms,
            tags=tags,
            cluster_labels=cluster_labels,
            metrics=quality_metrics(collection.vectors, labels),
            settings=recorded_settings,
        )
