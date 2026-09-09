"""Artifact generation for dream cluster analyses."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dream_analysis.artifacts import write_text_atomic
from dream_analysis.clustering import ClusterAnalysis


MPLCONFIGDIR = Path("/tmp/dream_analysis_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    """Write one cluster assignment row per dream."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "dream_id",
                "date",
                "year",
                "tags",
                "cluster",
                "cluster_label",
                "confidence",
                "x",
                "y",
            ],
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
                    "cluster_label": (
                        "noise / ambiguous"
                        if cluster < 0
                        else cluster_labels[cluster]
                    ),
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
    """Write static cluster and year projections."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(figsize=(11, 8))
    clusters = sorted(set(labels.tolist()))
    color_map = plt.get_cmap("tab20")
    for position, cluster in enumerate(clusters):
        mask = labels == cluster
        name = (
            "noise / ambiguous"
            if cluster < 0
            else f"{cluster}: {cluster_labels[cluster]}"
        )
        color = "#aaaaaa" if cluster < 0 else color_map(position % 20)
        axes.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=35,
            alpha=np.maximum(0.3, probabilities[mask]),
            c=[color],
            label=name,
        )
    axes.set(title="Dream embedding clusters", xlabel="UMAP 1", ylabel="UMAP 2")
    axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(
        output_dir / "embedding_clusters.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    years = np.asarray([int(metadata.get("year") or 0) for metadata in metadatas])
    figure, axes = plt.subplots(figsize=(10, 8))
    points = axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=years,
        cmap="viridis",
        s=38,
        alpha=0.8,
    )
    axes.set(title="Dream embedding map by year", xlabel="UMAP 1", ylabel="UMAP 2")
    figure.colorbar(points, ax=axes, label="Year")
    figure.tight_layout()
    figure.savefig(output_dir / "embedding_by_year.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_html(
    path: Path,
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    coordinates: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cluster_labels: dict[int, str],
) -> None:
    """Write a self-contained interactive Plotly map."""
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError("Plotly is required for the interactive map.") from exc
    rows = []
    for index, dream_id in enumerate(ids):
        cluster = int(labels[index])
        rows.append(
            {
                "x": coordinates[index, 0],
                "y": coordinates[index, 1],
                "dream_id": dream_id,
                "date": metadatas[index].get("date", ""),
                "tags": metadatas[index].get("tags", ""),
                "cluster": (
                    "noise / ambiguous"
                    if cluster < 0
                    else f"{cluster}: {cluster_labels[cluster]}"
                ),
                "confidence": probabilities[index],
                "excerpt": html.escape(" ".join(documents[index].split())[:300]),
            }
        )
    figure = px.scatter(
        rows,
        x="x",
        y="y",
        color="cluster",
        opacity=0.8,
        hover_data=["dream_id", "date", "tags", "confidence", "excerpt"],
        title="Dream embedding clusters",
    )
    figure.update_traces(marker={"size": 9})
    figure.write_html(path, include_plotlyjs=True)


def render_report(
    *,
    collection_name: str,
    collection_metadata: dict[str, Any],
    settings: dict[str, Any],
    metrics: dict[str, Any],
    labels: np.ndarray,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    representatives: dict[int, list[int]],
    terms: dict[int, list[str]],
    tags: dict[int, list[tuple[str, float, int]]],
    cluster_labels: dict[int, str],
) -> str:
    """Render the evidence report without performing file I/O."""
    silhouette = metrics["silhouette_cosine_non_noise"]
    lines = [
        "# Dream embedding cluster report",
        "",
        f"- Collection: `{collection_name}`",
        f"- Embedding model: `{collection_metadata.get('embedding_model', 'unknown')}`",
        f"- Dreams: {metrics['dream_count']}",
        f"- Clusters: {metrics['cluster_count']}",
        (
            f"- Noise/ambiguous dreams: {metrics['noise_count']} "
            f"({metrics['noise_fraction']:.1%})"
        ),
        f"- Cosine silhouette (non-noise): {silhouette if silhouette is not None else 'n/a'}",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(settings, indent=2),
        "```",
        "",
        (
            "> Cluster labels are descriptive summaries of shared content, "
            "not psychological interpretations."
        ),
        "",
    ]
    for cluster in sorted(representatives):
        member_count = int((labels == cluster).sum())
        tag_summary = ", ".join(
            f"`{tag}` ({ratio:.1f}× baseline; {count})"
            for tag, ratio, count in tags[cluster]
        )
        lines += [
            f"## Cluster {cluster}: {cluster_labels[cluster]}",
            "",
            f"Dreams: {member_count}",
            "",
            f"Distinctive terms: {', '.join(terms[cluster]) or 'none'}",
            "",
            "Overrepresented tags: " + (tag_summary or "none"),
            "",
            "Representative dreams:",
            "",
        ]
        for index in representatives[cluster]:
            excerpt = " ".join(documents[index].split())[:400]
            lines.append(
                f"- **{ids[index]}** "
                f"({metadatas[index].get('date', 'unknown')}): {excerpt}"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    collection_name: str,
    collection_metadata: dict[str, Any],
    settings: dict[str, Any],
    metrics: dict[str, Any],
    labels: np.ndarray,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    representatives: dict[int, list[int]],
    terms: dict[int, list[str]],
    tags: dict[int, list[tuple[str, float, int]]],
    cluster_labels: dict[int, str],
) -> None:
    """Write the Markdown evidence report."""
    content = render_report(
        collection_name=collection_name,
        collection_metadata=collection_metadata,
        settings=settings,
        metrics=metrics,
        labels=labels,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        representatives=representatives,
        terms=terms,
        tags=tags,
        cluster_labels=cluster_labels,
    )
    write_text_atomic(path, content)


class ClusterReportService:
    """Generate every user-facing artifact for a cluster analysis."""

    def write(self, analysis: ClusterAnalysis, output_dir: Path) -> dict[str, Path]:
        """Write CSV, HTML, PNG, and Markdown outputs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "csv": output_dir / "dream_clusters.csv",
            "html": output_dir / "embedding_map.html",
            "report": output_dir / "cluster_report.md",
        }
        collection = analysis.collection
        write_csv(
            paths["csv"],
            ids=collection.ids,
            metadatas=collection.metadatas,
            coordinates=analysis.coordinates,
            labels=analysis.labels,
            probabilities=analysis.probabilities,
            cluster_labels=analysis.cluster_labels,
        )
        plot_maps(
            output_dir,
            coordinates=analysis.coordinates,
            labels=analysis.labels,
            probabilities=analysis.probabilities,
            metadatas=collection.metadatas,
            cluster_labels=analysis.cluster_labels,
        )
        write_html(
            paths["html"],
            ids=collection.ids,
            documents=collection.documents,
            metadatas=collection.metadatas,
            coordinates=analysis.coordinates,
            labels=analysis.labels,
            probabilities=analysis.probabilities,
            cluster_labels=analysis.cluster_labels,
        )
        write_report(
            paths["report"],
            collection_name=collection.name,
            collection_metadata=collection.metadata,
            settings=analysis.settings,
            metrics=analysis.metrics,
            labels=analysis.labels,
            ids=collection.ids,
            documents=collection.documents,
            metadatas=collection.metadatas,
            representatives=analysis.representatives,
            terms=analysis.terms,
            tags=analysis.tags,
            cluster_labels=analysis.cluster_labels,
        )
        return paths
