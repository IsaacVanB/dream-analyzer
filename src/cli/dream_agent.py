#!/usr/bin/env python3
"""Answer dream-journal questions with a read-only Ollama search agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dream_analysis.agent import (
    AgentResponse,
    AgentTraceError,
    AgentTurnTrace,
    DreamRagAgent,
    ToolExecution,
    ToolRequest,
)
from dream_analysis.artifacts import write_text_atomic
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
        help="Maximum characters per retrieved dream returned to the model.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=3,
        help="Maximum search calls allowed for one answer.",
    )
    parser.add_argument(
        "--max-synthesis-dreams",
        type=int,
        default=10,
        help="Maximum unique dreams included in the final synthesis prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where a Markdown report should be saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print assistant messages and Ollama response diagnostics, and include "
            "them in the Markdown report."
        ),
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
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
    if not 1 <= args.max_synthesis_dreams <= 20:
        parser.error("--max-synthesis-dreams must be between 1 and 20")
    if args.num_ctx < 1:
        parser.error("--num-ctx must be positive")
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")


def print_searches(executions: tuple[ToolExecution, ...]) -> None:
    for index, execution in enumerate(executions, start=1):
        result = execution.result
        cached = " [CACHED DUPLICATE]" if execution.cached else ""
        print(
            f"\nSearch {index}{cached}: "
            f"{execution.arguments.get('query', '<missing>')}"
        )
        if execution.arguments.get("start_date") or execution.arguments.get(
            "end_date"
        ):
            print(
                "  Date range: "
                f"{execution.arguments.get('start_date', 'earliest')} through "
                f"{execution.arguments.get('end_date', 'latest')}"
            )
        if not result.get("ok"):
            print(f"  ERROR: {result.get('error', 'unknown tool error')}")
            continue
        dreams = result.get("dreams", [])
        for item in dreams:
            print(
                f"  - {item['dream_id']} | {item['date']} | "
                f"distance={item['distance']:.4f}"
            )


def print_pending_tool_calls(calls: tuple[ToolRequest, ...]) -> None:
    print("\n--- UNEXECUTED TOOL CALLS ---")
    if not calls:
        print("None")
        return
    for index, call in enumerate(calls, start=1):
        arguments = json.dumps(
            dict(call.arguments),
            ensure_ascii=False,
            sort_keys=True,
        )
        print(f"{index}. {call.name}: {arguments}")


def print_turn_trace(turns: tuple[AgentTurnTrace, ...]) -> None:
    print("\n--- AGENT TURN TRACE ---")
    if not turns:
        print("No agent turns were recorded.")
        return
    for index, turn in enumerate(turns, start=1):
        phase = "final synthesis" if turn.forced_synthesis else "tool loop"
        print(f"\nTurn {index} ({phase}, tools_enabled={turn.tools_enabled}):")
        if turn.request_prompt is not None:
            print("Request prompt:")
            print(turn.request_prompt)
        print("Assistant message:")
        print(json.dumps(dict(turn.assistant_message), ensure_ascii=False, indent=2))
        print("Diagnostics:")
        print(json.dumps(dict(turn.diagnostics), ensure_ascii=False, indent=2))


def format_markdown_report(
    question: str,
    response: AgentResponse,
    *,
    settings: Mapping[str, Any],
    trace_error: AgentTraceError | None = None,
    include_debug: bool = False,
) -> str:
    pending_calls = (
        trace_error.pending_tool_calls
        if trace_error is not None
        else response.unexecuted_tool_calls
    )
    lines = [
        (
            "# Dream Agent Partial Response"
            if trace_error is not None
            else "# Dream Agent Response"
        ),
        "",
        "## Question",
        "",
        *[f"> {line}" if line else ">" for line in question.strip().splitlines()],
        "",
        "## Settings",
        "",
    ]
    for name, value in settings.items():
        lines.append(f"- **{name}:** `{_inline_code(value)}`")

    if trace_error is not None:
        lines.extend(
            [
                "",
                "## Status",
                "",
                f"Error: {_markdown_text(trace_error)}",
                "",
                f"- Completed tool calls: `{len(trace_error.completed_executions)}`",
                f"- Unexecuted tool calls: `{len(pending_calls)}`",
            ]
        )
        max_tool_calls = getattr(trace_error, "max_tool_calls", None)
        if max_tool_calls is not None:
            lines.append(f"- Configured limit: `{max_tool_calls}`")
    elif response.forced_synthesis:
        lines.extend(
            [
                "",
                "## Status",
                "",
                "- Ranked final synthesis: `True`",
                "- Reason: "
                f"{_markdown_text(response.forced_synthesis_reason or 'unknown')}",
                f"- Unexecuted tool calls: `{len(pending_calls)}`",
            ]
        )

    lines.extend(["", "## Searches", ""])
    if not response.tool_executions:
        lines.extend(["No searches were recorded.", ""])
    retrieved_dreams: dict[str, dict[str, Any]] = {}
    for index, execution in enumerate(response.tool_executions, start=1):
        query = execution.arguments.get("query", "<missing>")
        cached = " — Cached Duplicate" if execution.cached else ""
        lines.extend(
            [
                f"### Search {index}{cached}",
                "",
                f"Query: `{_inline_code(query)}`",
                "",
            ]
        )
        if execution.arguments.get("start_date") or execution.arguments.get(
            "end_date"
        ):
            start_date = execution.arguments.get("start_date", "earliest")
            end_date = execution.arguments.get("end_date", "latest")
            lines.extend(
                [
                    "Date range: "
                    f"`{_inline_code(start_date)}` through "
                    f"`{_inline_code(end_date)}`",
                    "",
                ]
            )
        result = execution.report_result or execution.result
        if not result.get("ok"):
            error = _markdown_text(result.get("error", "unknown tool error"))
            lines.extend(
                [f"Error: {error}", ""]
            )
            continue
        lines.extend(
            [
                "| dream_id | date | distance |",
                "|---|---|---:|",
            ]
        )
        for dream in result.get("dreams", []):
            lines.append(
                f"| {_markdown_cell(dream['dream_id'])} "
                f"| {_markdown_cell(dream['date'])} "
                f"| {float(dream['distance']):.4f} |"
            )
        if not result.get("dreams"):
            lines.append("| *(no results)* |  |  |")
        lines.append("")

        for dream in result.get("dreams", []):
            dream_id = str(dream["dream_id"])
            if dream_id not in retrieved_dreams:
                retrieved_dreams[dream_id] = {
                    "dream": dream,
                    "searches": [index],
                }
            else:
                retrieved_dreams[dream_id]["searches"].append(index)

    if retrieved_dreams:
        lines.extend(["## Full Retrieved Dream Text", ""])
    for dream_index, item in enumerate(retrieved_dreams.values(), start=1):
        dream = item["dream"]
        search_numbers = ", ".join(str(value) for value in item["searches"])
        lines.extend(
            [
                (
                    f"### {dream_index}. Dream "
                    f"`{_inline_code(dream['dream_id'])}`"
                ),
                "",
                f"- Date: `{_inline_code(dream['date'])}`",
                f"- Retrieved by searches: `{search_numbers}`",
                "",
                _markdown_code_block(dream.get("text", "")),
                "",
            ]
        )

    if pending_calls:
        lines.extend(["## Unexecuted Tool Calls", ""])
        for index, call in enumerate(pending_calls, start=1):
            lines.extend(
                [
                    f"### Tool Call {index}: `{_inline_code(call.name)}`",
                    "",
                    "````json",
                    json.dumps(
                        dict(call.arguments),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "````",
                    "",
                ]
            )

    if include_debug:
        lines.extend(["## Agent Turn Trace", ""])
        if not response.turn_traces:
            lines.extend(["No agent turns were recorded.", ""])
        for index, turn in enumerate(response.turn_traces, start=1):
            phase = "Final Synthesis" if turn.forced_synthesis else "Tool Loop"
            lines.extend(
                [
                    f"### Turn {index}: {phase}",
                    "",
                    f"- Tools enabled: `{turn.tools_enabled}`",
                    "",
                    "````json",
                    json.dumps(
                        {
                            "request_prompt": turn.request_prompt,
                            "assistant_message": dict(turn.assistant_message),
                            "diagnostics": dict(turn.diagnostics),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "````",
                    "",
                ]
            )

    lines.extend(["## Answer", "", response.answer.rstrip(), ""])
    return "\n".join(lines)


def save_markdown_report(path: Path, report: str) -> Path:
    return write_text_atomic(path, report.rstrip() + "\n")


def _inline_code(value: Any) -> str:
    return str(value).replace("`", "'").replace("\n", " ")


def _markdown_text(value: Any) -> str:
    return str(value).replace("\n", " ")


def _markdown_cell(value: Any) -> str:
    return _markdown_text(value).replace("|", "\\|")


def _markdown_code_block(value: Any) -> str:
    text = str(value)
    longest_run = 0
    current_run = 0
    for character in text:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * max(4, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"


def report_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "Chat model": args.chat_model,
        "Embedding model": args.embed_model,
        "Collection": args.collection_name,
        "Chroma path": args.chroma_path,
        "Results per search": args.top_k,
        "Maximum model characters per dream": args.max_chars_per_dream,
        "Maximum tool calls": args.max_tool_calls,
        "Maximum synthesis dreams": args.max_synthesis_dreams,
        "Debug trace": args.debug,
    }


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
    try:
        response = agent.answer(
            args.question,
            chat_model=args.chat_model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            temperature=args.temperature,
            max_tool_calls=args.max_tool_calls,
            max_synthesis_dreams=args.max_synthesis_dreams,
        )
    except AgentTraceError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print_searches(exc.completed_executions)
        print_pending_tool_calls(exc.pending_tool_calls)
        if args.debug:
            print_turn_trace(exc.turn_traces)
        partial_response = AgentResponse(
            answer=f"[No final answer: {exc}]",
            tool_executions=exc.completed_executions,
            assistant_messages=exc.assistant_messages,
            turn_traces=exc.turn_traces,
            unexecuted_tool_calls=exc.pending_tool_calls,
        )
        if args.output is not None:
            report = format_markdown_report(
                args.question,
                partial_response,
                settings=report_settings(args),
                trace_error=exc,
                include_debug=args.debug,
            )
            output_path = save_markdown_report(args.output, report)
            print(f"\nSaved partial Markdown report to {output_path}")
        raise SystemExit(2) from None

    print_searches(response.tool_executions)
    if response.unexecuted_tool_calls:
        print_pending_tool_calls(response.unexecuted_tool_calls)
    if response.forced_synthesis:
        print("\nFinal ranked synthesis.")
        print(f"Reason: {response.forced_synthesis_reason or 'unknown'}")
    if args.debug:
        print_turn_trace(response.turn_traces)
    print("\n--- ANSWER ---\n")
    print(response.answer)
    if args.output is not None:
        report = format_markdown_report(
            args.question,
            response,
            settings=report_settings(args),
            include_debug=args.debug,
        )
        output_path = save_markdown_report(args.output, report)
        print(f"\nSaved Markdown report to {output_path}")


if __name__ == "__main__":
    main()
