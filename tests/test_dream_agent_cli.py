from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import dream_agent
from dream_analysis.agent import (
    AgentResponse,
    AgentToolLimitError,
    ToolExecution,
    ToolRequest,
)


class DreamAgentCliTests(unittest.TestCase):
    def response(self) -> AgentResponse:
        return AgentResponse(
            answer="A grounded **answer**.",
            tool_executions=(
                ToolExecution(
                    name="search_dreams",
                    arguments={
                        "query": "hidden room",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31",
                    },
                    result={
                        "ok": True,
                        "query": "hidden room",
                        "result_count": 1,
                        "dreams": [
                            {
                                "dream_id": "dream|1",
                                "date": "1/2/2024",
                                "distance": 0.125,
                                "text": "A hidden room.",
                                "truncated": False,
                            }
                        ],
                    },
                ),
            ),
        )

    def limit_error(self) -> AgentToolLimitError:
        return AgentToolLimitError(
            "Ollama exceeded the limit of 1 tool calls",
            max_tool_calls=1,
            completed_executions=self.response().tool_executions,
            pending_tool_calls=(
                ToolRequest(
                    name="search_dreams",
                    arguments={"query": "school", "start_date": "2024-01-01"},
                ),
            ),
            assistant_messages=(
                {
                    "role": "assistant",
                    "content": "I should search another topic.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_dreams",
                                "arguments": {"query": "school"},
                            }
                        }
                    ],
                },
            ),
        )

    def test_parser_accepts_an_optional_output_path(self) -> None:
        args = dream_agent.build_parser().parse_args(
            ["What happened?", "--output", "outputs/agent/report.md"]
        )

        self.assertEqual(args.output, Path("outputs/agent/report.md"))

    def test_parser_accepts_debug_tracing(self) -> None:
        args = dream_agent.build_parser().parse_args(["What happened?", "--debug"])

        self.assertTrue(args.debug)

    def test_markdown_report_contains_expected_sections(self) -> None:
        report = dream_agent.format_markdown_report(
            "What hidden rooms recur?",
            self.response(),
            settings={"Chat model": "qwen3:8b", "Results per search": 8},
        )

        self.assertIn("# Dream Agent Response", report)
        self.assertIn("> What hidden rooms recur?", report)
        self.assertIn("**Chat model:** `qwen3:8b`", report)
        self.assertIn("Query: `hidden room`", report)
        self.assertIn(
            "Date range: `2024-01-01` through `2024-01-31`",
            report,
        )
        self.assertIn("dream\\|1", report)
        self.assertIn("A grounded **answer**.", report)

    def test_partial_markdown_contains_limit_trace(self) -> None:
        error = self.limit_error()
        response = AgentResponse(
            answer=f"[No final answer: {error}]",
            tool_executions=error.completed_executions,
            assistant_messages=error.assistant_messages,
        )

        report = dream_agent.format_markdown_report(
            "Compare hidden rooms and schools",
            response,
            settings={"Maximum tool calls": 1},
            limit_error=error,
            include_debug=True,
        )

        self.assertIn("# Dream Agent Partial Response", report)
        self.assertIn("Completed tool calls: `1`", report)
        self.assertIn("## Unexecuted Tool Calls", report)
        self.assertIn('"query": "school"', report)
        self.assertIn("## Assistant Trace", report)
        self.assertIn("I should search another topic.", report)
        self.assertIn("[No final answer:", report)

    def test_main_saves_partial_report_when_limit_is_exceeded(self) -> None:
        error = self.limit_error()

        class LimitAgent:
            @staticmethod
            def answer(*args, **kwargs):
                raise error

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "partial.md"
            argv = [
                "dream_agent.py",
                "Compare hidden rooms and schools",
                "--max-tool-calls",
                "1",
                "--debug",
                "--output",
                str(output_path),
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(dream_agent, "build_agent", return_value=LimitAgent()),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                dream_agent.main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("UNEXECUTED TOOL CALLS", stdout.getvalue())
            self.assertIn("ASSISTANT TRACE", stdout.getvalue())
            self.assertIn("exceeded the limit", stderr.getvalue())
            self.assertIn("Partial Response", output_path.read_text(encoding="utf-8"))

    def test_save_markdown_report_creates_parent_and_replaces_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "report.md"

            saved = dream_agent.save_markdown_report(path, "# First report")
            dream_agent.save_markdown_report(path, "# New report")

            self.assertEqual(saved, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "# New report\n")
            self.assertFalse(path.with_name(".report.md.tmp").exists())


if __name__ == "__main__":
    unittest.main()
