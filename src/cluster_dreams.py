#!/usr/bin/env python3
"""Cluster stored dream embeddings and generate evidence-backed theme reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

MPLCONFIGDIR = Path("/tmp/dream_analysis_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))
NUMBA_CACHE_DIR = Path("/tmp/dream_analysis_numba")
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dreams"
OUTPUT_DIR = Path("outputs/clusters")
DREAM_TEXT_SEPARATOR = "--- DREAM TEXT ---"


def extract_dream_text(document: str) -> str:
    _, separator, dream_text = document.partition(DREAM_TEXT_SEPARATOR)
    return (dream_text if separator else document).strip()


def load_collection(
    *, chroma_path: str, collection_name: str
) -> tuple[list[str], list[str], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    collection = chromadb.PersistentClient(path=chroma_path).get_collection(
        name=collection_name
    )
    records = collection.get(include=["documents", "metadatas", "embeddings"])
    embeddings = records.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise ValueError(f"Collection {collection_name!r} contains no embeddings.")

    ids = [str(value) for value in records["ids"]]
    documents = [extract_dream_text(value or "") for value in records["documents"]]
    metadatas = [value or {} for value in records["metadatas"]]
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
        raise ValueError("Stored embeddings do not form a dream-by-dimension matrix.")
    return ids, documents, metadatas, vectors, collection.metadata or {}


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
    try:
        import hdbscan
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "Clustering dependencies are missing. Run: pip install -r requirements.txt"
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
    raw = metadata.get("tags", "")
    if isinstance(raw, list):
        return [str(tag).strip() for tag in raw if str(tag).strip()]
    return [tag.strip() for tag in str(raw).split(",") if tag.strip()]


def representative_indices(
    vectors: np.ndarray, labels: np.ndarray, cluster: int, *, count: int
) -> list[int]:
    members = np.flatnonzero(labels == cluster)
    member_vectors = normalize(vectors[members], norm="l2")
    centroid = normalize(member_vectors.mean(axis=0).reshape(1, -1))[0]
    similarities = member_vectors @ centroid
    return members[np.argsort(-similarities)[:count]].tolist()


def distinctive_terms(
    documents: list[str], labels: np.ndarray, *, top_n: int
) -> dict[int, list[str]]:
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
    all_counts = Counter(tag for metadata in metadatas for tag in set(parse_tags(metadata)))
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
        output[int(cluster)] = sorted(scored, key=lambda row: (-row[1], -row[2], row[0]))[
            :top_n
        ]
    return output


def automatic_label(terms: list[str], tags: list[tuple[str, float, int]]) -> str:
    evidence = [tag for tag, _, _ in tags[:2]] + terms[:3]
    return " / ".join(dict.fromkeys(evidence)) or "unlabeled theme"


def llm_label(
    *, model: str, terms: list[str], tags: list[tuple[str, float, int]], texts: list[str]
) -> str:
    import ollama

    excerpts = "\n\n".join(text[:700] for text in texts)
    prompt = f"""Give this dream cluster a short, descriptive theme label of 2-7 words.
Do not diagnose or infer hidden psychological meaning. Describe only recurring content.
Distinctive terms: {', '.join(terms)}
Overrepresented tags: {', '.join(tag for tag, _, _ in tags) or 'none'}
Representative dreams:
{excerpts}
Return only the label."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 30},
    )
    label = str(response["message"]["content"]).strip().strip('"').splitlines()[0]
    return label[:100] or automatic_label(terms, tags)


def quality_metrics(vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    clusters = sorted(cluster for cluster in set(labels.tolist()) if cluster >= 0)
    non_noise = labels >= 0
    silhouette: float | None = None
    if len(clusters) >= 2 and non_noise.sum() > len(clusters):
        silhouette = float(
            silhouette_score(normalize(vectors[non_noise]), labels[non_noise], metric="cosine")
        )
    return {
        "dream_count": int(len(labels)),
        "cluster_count": len(clusters),
        "noise_count": int((labels < 0).sum()),
        "noise_fraction": float((labels < 0).mean()),
        "silhouette_cosine_non_noise": silhouette,
    }


def write_csv(
    path: Path,
    *,
    ids: list[str],
    metadatas: list[dict[str, Any]],
    coordinates: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cluster_labels: dict[int, str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["dream_id", "date", "year", "tags", "cluster", "cluster_label", "confidence", "x", "y"],
        )
        writer.writeheader()
        for index, dream_id in enumerate(ids):
            cluster = int(labels[index])
            writer.writerow(
                {
                    "dream_id": dream_id,
                    "date": metadatas[index].get("date", ""),
                    "year": metadatas[index].get("year", ""),
                    "tags": metadatas[index].get("tags", ""),
                    "cluster": cluster,
                    "cluster_label": "noise / ambiguous" if cluster < 0 else cluster_labels[cluster],
                    "confidence": f"{probabilities[index]:.6f}",
                    "x": f"{coordinates[index, 0]:.6f}",
                    "y": f"{coordinates[index, 1]:.6f}",
                }
            )


def plot_maps(
    output_dir: Path,
    *,
    coordinates: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadatas: list[dict[str, Any]],
    cluster_labels: dict[int, str],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 8))
    clusters = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20")
    for position, cluster in enumerate(clusters):
        mask = labels == cluster
        name = "noise / ambiguous" if cluster < 0 else f"{cluster}: {cluster_labels[cluster]}"
        color = "#aaaaaa" if cluster < 0 else cmap(position % 20)
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=35, alpha=np.maximum(0.3, probabilities[mask]), c=[color], label=name)
    ax.set(title="Dream embedding clusters", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    years = np.asarray([int(metadata.get("year") or 0) for metadata in metadatas])
    fig, ax = plt.subplots(figsize=(10, 8))
    points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=years, cmap="viridis", s=38, alpha=0.8)
    ax.set(title="Dream embedding map by year", xlabel="UMAP 1", ylabel="UMAP 2")
    fig.colorbar(points, ax=ax, label="Year")
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_by_year.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_html(
    path: Path,
    *,
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]],
    coordinates: np.ndarray, labels: np.ndarray, probabilities: np.ndarray,
    cluster_labels: dict[int, str],
) -> None:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError("Plotly is required for the interactive map.") from exc
    rows = []
    for index, dream_id in enumerate(ids):
        cluster = int(labels[index])
        rows.append({
            "x": coordinates[index, 0], "y": coordinates[index, 1],
            "dream_id": dream_id, "date": metadatas[index].get("date", ""),
            "tags": metadatas[index].get("tags", ""),
            "cluster": "noise / ambiguous" if cluster < 0 else f"{cluster}: {cluster_labels[cluster]}",
            "confidence": probabilities[index],
            "excerpt": html.escape(" ".join(documents[index].split())[:300]),
        })
    figure = px.scatter(rows, x="x", y="y", color="cluster", opacity=0.8,
                        hover_data=["dream_id", "date", "tags", "confidence", "excerpt"],
                        title="Dream embedding clusters")
    figure.update_traces(marker={"size": 9})
    # Keep the journal visualization self-contained and usable without a network.
    figure.write_html(path, include_plotlyjs=True)


def write_report(
    path: Path, *, collection_name: str, collection_metadata: dict[str, Any],
    settings: dict[str, Any], metrics: dict[str, Any], labels: np.ndarray,
    ids: list[str], documents: list[str], metadatas: list[dict[str, Any]],
    representatives: dict[int, list[int]], terms: dict[int, list[str]],
    tags: dict[int, list[tuple[str, float, int]]], cluster_labels: dict[int, str],
) -> None:
    lines = ["# Dream embedding cluster report", "", f"- Collection: `{collection_name}`",
             f"- Embedding model: `{collection_metadata.get('embedding_model', 'unknown')}`",
             f"- Dreams: {metrics['dream_count']}", f"- Clusters: {metrics['cluster_count']}",
             f"- Noise/ambiguous dreams: {metrics['noise_count']} ({metrics['noise_fraction']:.1%})",
             f"- Cosine silhouette (non-noise): {metrics['silhouette_cosine_non_noise'] if metrics['silhouette_cosine_non_noise'] is not None else 'n/a'}",
             "", "## Settings", "", "```json", json.dumps(settings, indent=2), "```", "",
             "> Cluster labels are descriptive summaries of shared content, not psychological interpretations.", ""]
    for cluster in sorted(representatives):
        member_count = int((labels == cluster).sum())
        lines += [f"## Cluster {cluster}: {cluster_labels[cluster]}", "", f"Dreams: {member_count}", "",
                  f"Distinctive terms: {', '.join(terms[cluster]) or 'none'}", "",
                  "Overrepresented tags: " + (", ".join(f"`{tag}` ({ratio:.1f}× baseline; {count})" for tag, ratio, count in tags[cluster]) or "none"), "", "Representative dreams:", ""]
        for index in representatives[cluster]:
            excerpt = " ".join(documents[index].split())[:400]
            lines += [f"- **{ids[index]}** ({metadatas[index].get('date', 'unknown')}): {excerpt}"]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    ids, documents, metadatas, vectors, collection_metadata = load_collection(
        chroma_path=args.chroma_path, collection_name=args.collection_name
    )
    coordinates, labels, probabilities, actual_pca = project_and_cluster(
        vectors, pca_dimensions=args.pca_dimensions, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples, random_state=args.random_state,
    )
    terms = distinctive_terms(documents, labels, top_n=args.top_terms)
    tags = enriched_tags(metadatas, labels, top_n=args.top_tags)
    representatives = {
        int(cluster): representative_indices(vectors, labels, int(cluster), count=args.representative_dreams)
        for cluster in sorted(set(labels)) if cluster >= 0
    }
    cluster_labels = {}
    for cluster, indices in representatives.items():
        cluster_labels[cluster] = (
            llm_label(model=args.label_model, terms=terms[cluster], tags=tags[cluster], texts=[documents[i] for i in indices])
            if args.label_clusters else automatic_label(terms[cluster], tags[cluster])
        )
    settings = {
        "pca_dimensions": actual_pca, "umap_neighbors": args.n_neighbors,
        "umap_min_dist": args.min_dist, "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples, "random_state": args.random_state,
        "llm_labels": args.label_clusters, "label_model": args.label_model if args.label_clusters else None,
    }
    metrics = quality_metrics(vectors, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": args.output_dir / "dream_clusters.csv",
        "html": args.output_dir / "embedding_map.html",
        "report": args.output_dir / "cluster_report.md",
    }
    write_csv(paths["csv"], ids=ids, metadatas=metadatas, coordinates=coordinates,
              labels=labels, probabilities=probabilities, cluster_labels=cluster_labels)
    plot_maps(args.output_dir, coordinates=coordinates, labels=labels,
              probabilities=probabilities, metadatas=metadatas, cluster_labels=cluster_labels)
    write_html(paths["html"], ids=ids, documents=documents, metadatas=metadatas,
               coordinates=coordinates, labels=labels, probabilities=probabilities,
               cluster_labels=cluster_labels)
    write_report(paths["report"], collection_name=args.collection_name,
                 collection_metadata=collection_metadata, settings=settings, metrics=metrics,
                 labels=labels, ids=ids, documents=documents, metadatas=metadatas,
                 representatives=representatives, terms=terms, tags=tags,
                 cluster_labels=cluster_labels)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster stored dream embeddings and report candidate themes.")
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pca-dimensions", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="HDBSCAN conservativeness; raise this to classify more dreams as noise (default: 1).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--top-tags", type=int, default=5)
    parser.add_argument("--representative-dreams", type=int, default=3)
    parser.add_argument("--label-clusters", action="store_true", help="Use Ollama for content-only cluster labels.")
    parser.add_argument("--label-model", default="qwen3:8b")
    args = parser.parse_args()
    if args.min_cluster_size < 2:
        parser.error("--min-cluster-size must be at least 2")
    paths = run(args)
    print("Cluster analysis complete:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
