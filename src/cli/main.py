"""Consolidated command-line interface for Dream Analyzer."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from cli import (
    analyze_dream,
    build_chroma_db,
    cluster_dreams,
    compute_stats,
    dream_agent,
    parse_dreams,
    plot_tags,
)


CommandHandler = Callable[[argparse.Namespace, argparse.ArgumentParser], Any]


def _add_command(
    subparsers: Any,
    name: str,
    help_text: str,
    configure: Callable[[argparse.ArgumentParser], argparse.ArgumentParser],
    handler: CommandHandler,
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    configure(parser)
    parser.set_defaults(_command_handler=handler, _command_parser=parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser using arguments owned by legacy commands."""
    parser = argparse.ArgumentParser(
        prog="dream-analyzer",
        description="Parse, index, query, and analyze a local dream journal.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )
    _add_command(
        subparsers,
        "parse",
        "parse a journal text file into JSONL",
        parse_dreams.build_parser,
        parse_dreams.run,
    )
    _add_command(
        subparsers,
        "index",
        "synchronize or rebuild the vector index",
        build_chroma_db.build_parser,
        build_chroma_db.run,
    )
    _add_command(
        subparsers,
        "ask",
        "ask an agent a question about the journal",
        dream_agent.build_parser,
        dream_agent.run,
    )
    _add_command(
        subparsers,
        "analyze",
        "analyze one dream by ID or supplied text",
        analyze_dream.build_parser,
        analyze_dream.run,
    )
    _add_command(
        subparsers,
        "stats",
        "compute summary statistics",
        compute_stats.build_parser,
        compute_stats.run,
    )
    _add_command(
        subparsers,
        "trends",
        "plot dream-tag trends over time",
        plot_tags.build_parser,
        plot_tags.run,
    )
    _add_command(
        subparsers,
        "cluster",
        "cluster indexed dream embeddings",
        cluster_dreams.build_parser,
        cluster_dreams.run_command,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    """Parse arguments and dispatch to the selected legacy command adapter."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._command_handler(args, args._command_parser)


if __name__ == "__main__":
    main()
