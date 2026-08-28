from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import dream_agent
from dream_analysis.agent import AgentResponse, ToolExecution


class DreamAgentCliTests(unittest.TestCase):
    def response(self) -> AgentResponse:
        return AgentResponse(
            answer="A grounded **answer**.",
            tool_executions=(
                ToolExecution(
                    name="search_dreams",
                    arguments={"query": "hidden room"},
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

    def test_parser_accepts_an_optional_output_path(self) -> None:
        args = dream_agent.build_parser().parse_args(
            ["What happened?", "--output", "outputs/agent/report.md"]
        )

        self.assertEqual(args.output, Path("outputs/agent/report.md"))

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
        self.assertIn("dream\\|1", report)
        self.assertIn("A grounded **answer**.", report)

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
