#!/usr/bin/env python3
"""Cluster stored dream embeddings and generate evidence-backed theme reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from dream_analysis.cluster_reporting import (
    ClusterReportService,
    plot_maps,
    render_report,
    write_csv,
    write_html,
    write_report,
)
from dream_analysis.clustering import (
    ClusteringSettings,
    DreamClusteringService,
    DreamEmbeddingRepository,
    automatic_label,
    distinctive_terms,
    enriched_tags,
    llm_label,
    load_collection,
    parse_tags,
    project_and_cluster,
    quality_metrics,
    representative_indices,
)
from dream_analysis.index import DREAM_TEXT_SEPARATOR, extract_dream_text


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dreams"
OUTPUT_DIR = Path("outputs/clusters")


def run(args: argparse.Namespace) -> dict[str, Path]:
    """Load data, run the clustering service, and write its artifacts."""
    collection = DreamEmbeddingRepository(
        path=args.chroma_path,
        collection_name=args.collection_name,
    ).load()
    settings = ClusteringSettings(
        pca_dimensions=args.pca_dimensions,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        random_state=args.random_state,
        top_terms=args.top_terms,
        top_tags=args.top_tags,
        representative_dreams=args.representative_dreams,
    )
    analysis = DreamClusteringService(
        settings=settings,
        label_clusters=args.label_clusters,
        label_model=args.label_model,
    ).analyze(collection)
    return ClusterReportService().write(analysis, args.output_dir)


def build_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Build the command-line parser separately for reuse and testing."""
    description = "Cluster stored dream embeddings and report candidate themes."
    parser = parser or argparse.ArgumentParser(description=description)
    parser.description = description
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
        help=(
            "HDBSCAN conservativeness; raise this to classify more dreams as "
            "noise (default: 1)."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--top-tags", type=int, default=5)
    parser.add_argument("--representative-dreams", type=int, default=3)
    parser.add_argument(
        "--label-clusters",
        action="store_true",
        help="Use Ollama for content-only cluster labels.",
    )
    parser.add_argument("--label-model", default="qwen3:8b")
    return parser


def run_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> dict[str, Path]:
    parser = parser or build_parser()
    try:
        paths = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print("Cluster analysis complete:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


def main() -> None:
    parser = build_parser()
    run_command(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
