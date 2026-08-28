#!/usr/bin/env python3
"""Answer dream-journal questions with a read-only Ollama search agent."""

from __future__ import annotations

import argparse

from dream_analysis.agent import DreamRagAgent, ToolExecution
from dream_analysis.config import Settings
from dream_analysis.index import DreamIndex
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.tools import DreamSearchTool


DEFAULT_SETTINGS = Settings()


def build_agent(
    *,
    chroma_path: str,
    collection_name: str,
    embed_model: str,
    top_k: int,
    max_chars_per_dream: int,
) -> DreamRagAgent:
    gateway = OllamaGateway()
    index = DreamIndex(
        path=chroma_path,
        collection_name=collection_name,
        embedding_model=embed_model,
        ollama_gateway=gateway,
    )
    return DreamRagAgent(
        ollama_gateway=gateway,
        search_tool=DreamSearchTool(
            index,
            result_limit=top_k,
            max_chars_per_dream=max_chars_per_dream,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using an Ollama agent with read-only dream search."
        )
    )
    parser.add_argument("question", help="Question to answer from the dream journal.")
    parser.add_argument(
        "--chroma-path",
        default=str(DEFAULT_SETTINGS.index.path),
        help="Path to the persistent ChromaDB database.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_SETTINGS.index.collection_name,
        help="Name of the ChromaDB collection to query.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_SETTINGS.ollama.embedding_model,
        help="Ollama embedding model used by the selected collection.",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_SETTINGS.ollama.chat_model,
        help="Tool-capable Ollama chat model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum dreams returned by each search call.",
    )
    parser.add_argument(
        "--max-chars-per-dream",
        type=int,
        default=2500,
        help="Maximum characters returned for each retrieved dream.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=3,
        help="Maximum search calls allowed for one answer.",
    )
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.question.strip():
        parser.error("question cannot be empty")
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    if args.max_chars_per_dream < 1:
        parser.error("--max-chars-per-dream must be positive")
    if args.max_tool_calls < 1:
        parser.error("--max-tool-calls must be positive")
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")


def print_searches(executions: tuple[ToolExecution, ...]) -> None:
    for index, execution in enumerate(executions, start=1):
        result = execution.result
        print(f"\nSearch {index}: {execution.arguments.get('query', '<missing>')}")
        if not result.get("ok"):
            print(f"  ERROR: {result.get('error', 'unknown tool error')}")
            continue
        dreams = result.get("dreams", [])
        for item in dreams:
            print(
                f"  - {item['dream_id']} | {item['date']} | "
                f"distance={item['distance']:.4f}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    agent = build_agent(
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_chars_per_dream=args.max_chars_per_dream,
    )
    response = agent.answer(
        args.question,
        chat_model=args.chat_model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        max_tool_calls=args.max_tool_calls,
    )
    print_searches(response.tool_executions)
    print("\n--- ANSWER ---\n")
    print(response.answer)


if __name__ == "__main__":
    main()
