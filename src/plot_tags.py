#!/usr/bin/env python3
"""Plot dream tag frequencies over time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

MPLCONFIGDIR = Path("/tmp/dream_analysis_matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from dream_analysis.config import Settings
from dream_analysis.dates import (
    filter_records_by_date,
    format_period_label as shared_period_label,
    parse_date_bound,
    record_date,
)
from dream_analysis.repository import DreamRepository


plt.style.use("seaborn-v0_8-whitegrid")

DEFAULT_SETTINGS = Settings()
DREAMS_PATH = DEFAULT_SETTINGS.dreams_path
OUTPUT_PATH = DEFAULT_SETTINGS.output_path / "plots/tag_frequency.png"


def load_dreams(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper returning validated record dictionaries."""
    return DreamRepository(path).records()


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


def parse_date_filter(date_text: str | None, *, argument_name: str) -> pd.Timestamp | None:
    parsed = parse_date_bound(date_text, argument_name=argument_name)
    return pd.Timestamp(parsed) if parsed is not None else None


def dream_plot_date(dream: dict[str, Any]) -> pd.Timestamp | None:
    parsed = record_date(dream)
    return pd.Timestamp(parsed) if parsed is not None else None


def filter_dreams_by_date(
    dreams: list[dict[str, Any]],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    return filter_records_by_date(
        dreams,
        start_date=start_date,
        end_date=end_date,
    )


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
        date = dream_plot_date(dream)
        if date is None:
            continue
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


def dream_counts_over_time(dreams: list[dict[str, Any]], *, freq: str = "M") -> pd.Series:
    if not dreams:
        raise ValueError("No dreams found.")

    periods: list[pd.Timestamp] = []
    for dream in dreams:
        date = dream_plot_date(dream)
        if date is not None:
            periods.append(date.to_period(freq).to_timestamp())

    all_periods = pd.Index(sorted(set(periods)), name="period")
    return pd.Series(periods).value_counts().sort_index().reindex(all_periods, fill_value=0)


def plot_tag_counts(
    counts: pd.DataFrame,
    *,
    output_path: Path,
    freq: str = "M",
    total_counts: pd.Series | None = None,
    title: str | None = None,
    normalize: bool = False,
    show: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [format_period_label(period, freq=freq) for period in counts.index]
    plot_counts = counts.copy()
    plot_counts.index = labels

    width = max(11, len(plot_counts.index) * 0.28)
    fig, ax = plt.subplots(figsize=(width, 6))
    tag_colors = plt.get_cmap("Set2").colors
    if normalize or total_counts is None:
        plot_counts.plot(kind="bar", ax=ax, width=0.85, color=tag_colors)
    else:
        x_positions = np.arange(len(plot_counts.index))
        total_values = total_counts.reindex(counts.index, fill_value=0).to_numpy()
        ax.bar(
            x_positions,
            total_values,
            width=0.85,
            color="#e8e8e8",
            edgecolor="#c7c7c7",
            linewidth=0.8,
            label="total dreams",
            zorder=1,
        )

        tag_width = 0.7 / max(len(plot_counts.columns), 1)
        offsets = (
            np.arange(len(plot_counts.columns)) - (len(plot_counts.columns) - 1) / 2
        ) * tag_width
        for index, (offset, tag) in enumerate(zip(offsets, plot_counts.columns)):
            ax.bar(
                x_positions + offset,
                plot_counts[tag].to_numpy(),
                width=tag_width,
                color=tag_colors[index % len(tag_colors)],
                label=tag,
                zorder=2,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_counts.index)

    ax.set_title(title or "Dream Tag Frequency Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent of dreams" if normalize else "Dream count")
    style_axis(ax)
    ax.legend(title="Tag", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c0c0c0")
    ax.spines["bottom"].set_color("#c0c0c0")


def format_period_label(period: pd.Timestamp, *, freq: str = "M") -> str:
    return shared_period_label(period, frequency=freq)


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
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    all_dreams = load_dreams(dreams_path)
    excluded_unknown_date_count = sum(
        1 for dream in all_dreams if dream_plot_date(dream) is None
    )
    dreams = filter_dreams_by_date(
        all_dreams,
        start_date=start_date,
        end_date=end_date,
    )
    selected_tags = tags or top_tags(dreams, top_n=top_n)
    counts = tag_counts_over_time(
        dreams,
        freq=freq,
        tags=selected_tags,
        top_n=top_n,
        normalize=normalize,
    )
    total_counts = dream_counts_over_time(dreams, freq=freq)
    plot_path = plot_tag_counts(
        counts,
        output_path=output_path,
        freq=freq,
        total_counts=total_counts,
        title=title,
        normalize=normalize,
        show=show,
    )

    available_tags = set(top_tags(dreams, top_n=10_000))
    missing_tags = [tag for tag in selected_tags if tag not in available_tags]

    dates = pd.Series([dream_plot_date(dream) for dream in dreams])
    return {
        "plot_path": str(plot_path),
        "tags": selected_tags,
        "missing_tags": missing_tags,
        "freq": freq,
        "normalize": normalize,
        "includes_total_dream_bars": not normalize,
        "dream_count": len(dreams),
        "excluded_unknown_date_count": excluded_unknown_date_count,
        "start_date": start_date,
        "end_date": end_date,
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
        "--start-date",
        help="Only include dreams on or after this date, e.g. 2023-01-01.",
    )
    parser.add_argument(
        "--end-date",
        help="Only include dreams on or before this date, e.g. 2023-12-31.",
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
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
