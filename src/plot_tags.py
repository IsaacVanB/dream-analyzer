#!/usr/bin/env python3
"""Plot dream tag frequencies over time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path("/tmp/dream_analysis_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DREAMS_PATH = Path("data/dreams.jsonl")
OUTPUT_PATH = Path("outputs/plots/tag_frequency.png")


def load_dreams(path: Path) -> list[dict[str, Any]]:
    dreams: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                try:
                    dreams.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return dreams


def top_tags(dreams: list[dict[str, Any]], *, top_n: int) -> list[str]:
    counts: dict[str, int] = {}
    for dream in dreams:
        for tag in dream.get("tags", []):
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return [
        tag
        for tag, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
    ]


def tag_counts_over_time(
    dreams: list[dict[str, Any]],
    *,
    freq: str = "M",
    tags: list[str] | None = None,
    top_n: int = 10,
    normalize: bool = False,
) -> pd.DataFrame:
    if not dreams:
        raise ValueError("No dreams found.")

    selected_tags = tags or top_tags(dreams, top_n=top_n)
    if not selected_tags:
        raise ValueError("No tags found in dreams.")

    rows: list[dict[str, Any]] = []
    dream_periods: list[pd.Timestamp] = []

    for dream in dreams:
        date = pd.to_datetime(dream["date"])
        period = date.to_period(freq).to_timestamp()
        dream_periods.append(period)

        dream_tags = {str(tag) for tag in dream.get("tags", [])}
        for tag in selected_tags:
            if tag in dream_tags:
                rows.append({"period": period, "tag": tag, "count": 1})

    if rows:
        counts = (
            pd.DataFrame(rows)
            .groupby(["period", "tag"])["count"]
            .sum()
            .unstack(fill_value=0)
        )
    else:
        counts = pd.DataFrame(columns=selected_tags, dtype=float)

    all_periods = pd.Index(sorted(set(dream_periods)), name="period")
    counts = counts.reindex(index=all_periods, columns=selected_tags, fill_value=0)

    if normalize:
        dreams_per_period = pd.Series(dream_periods).value_counts().sort_index()
        counts = counts.div(dreams_per_period, axis=0).fillna(0) * 100

    return counts


def plot_tag_counts(
    counts: pd.DataFrame,
    *,
    output_path: Path,
    title: str | None = None,
    normalize: bool = False,
    show: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    counts.plot(ax=ax, marker="o")

    ax.set_title(title or "Dream Tag Frequency Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent of dreams" if normalize else "Dream count")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Tag", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def make_tag_frequency_plot(
    *,
    dreams_path: Path = DREAMS_PATH,
    output_path: Path = OUTPUT_PATH,
    freq: str = "M",
    tags: list[str] | None = None,
    top_n: int = 10,
    normalize: bool = False,
    title: str | None = None,
    show: bool = False,
) -> dict[str, Any]:
    dreams = load_dreams(dreams_path)
    selected_tags = tags or top_tags(dreams, top_n=top_n)
    counts = tag_counts_over_time(
        dreams,
        freq=freq,
        tags=selected_tags,
        top_n=top_n,
        normalize=normalize,
    )
    plot_path = plot_tag_counts(
        counts,
        output_path=output_path,
        title=title,
        normalize=normalize,
        show=show,
    )

    available_tags = set(top_tags(dreams, top_n=10_000))
    missing_tags = [tag for tag in selected_tags if tag not in available_tags]

    dates = pd.to_datetime([dream["date"] for dream in dreams])
    return {
        "plot_path": str(plot_path),
        "tags": selected_tags,
        "missing_tags": missing_tags,
        "freq": freq,
        "normalize": normalize,
        "dream_count": len(dreams),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dream tag frequencies over time."
    )
    parser.add_argument(
        "--dreams-path",
        type=Path,
        default=DREAMS_PATH,
        help="Path to parsed dream JSONL records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path where the plot image should be saved.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Specific tags to plot. Defaults to the top tags.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top tags to plot when --tags is omitted.",
    )
    parser.add_argument(
        "--freq",
        choices=["M", "Q", "Y"],
        default="M",
        help="Time grouping frequency: M=month, Q=quarter, Y=year.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Plot percent of dreams per period instead of raw counts.",
    )
    parser.add_argument(
        "--title",
        help="Optional plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    args = parser.parse_args()

    metadata = make_tag_frequency_plot(
        dreams_path=args.dreams_path,
        output_path=args.output,
        freq=args.freq,
        tags=args.tags,
        top_n=args.top_n,
        normalize=args.normalize,
        title=args.title,
        show=args.show,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
