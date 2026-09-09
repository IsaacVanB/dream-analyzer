from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import main as consolidated_cli


class ConsolidatedCliTests(unittest.TestCase):
    def test_top_level_help_lists_commands(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            consolidated_cli.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        for command in (
            "parse",
            "index",
            "ask",
            "analyze",
            "stats",
            "trends",
            "cluster",
        ):
            self.assertIn(command, output.getvalue())

    def test_subcommand_help_reuses_command_options(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            consolidated_cli.main(["index", "--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--rebuild", output.getvalue())
        self.assertIn("--collection-name", output.getvalue())

    @patch("cli.main.compute_stats.run")
    def test_dispatches_to_selected_command(self, run_stats) -> None:
        consolidated_cli.main(["stats", "--freq", "Y"])

        run_stats.assert_called_once()
        args, command_parser = run_stats.call_args.args
        self.assertEqual(args.command, "stats")
        self.assertEqual(args.freq, "Y")
        self.assertEqual(command_parser.prog, "dream-analyzer stats")

    def test_representative_arguments_for_major_subcommands(self) -> None:
        parser = consolidated_cli.build_parser()

        parsed = parser.parse_args(["parse", "journal.txt", "dreams.jsonl"])
        self.assertEqual(parsed.input, Path("journal.txt"))
        self.assertEqual(parsed.output, Path("dreams.jsonl"))

        parsed = parser.parse_args(["index", "--rebuild", "--batch-size", "4"])
        self.assertTrue(parsed.rebuild)
        self.assertEqual(parsed.batch_size, 4)

        parsed = parser.parse_args(["ask", "What themes recur?", "--top-k", "6"])
        self.assertEqual(parsed.question, "What themes recur?")
        self.assertEqual(parsed.top_k, 6)

        parsed = parser.parse_args(
            ["analyze", "dream-2024-1-2-0", "--related-dreams", "3"]
        )
        self.assertEqual(parsed.dream_id_argument, "dream-2024-1-2-0")
        self.assertEqual(parsed.related_dreams, 3)

        parsed = parser.parse_args(["stats", "--freq", "Q", "--common-words", "12"])
        self.assertEqual((parsed.freq, parsed.common_words), ("Q", 12))

        parsed = parser.parse_args(["trends", "--tag", "school", "--normalize"])
        self.assertEqual(parsed.tags, ["school"])
        self.assertTrue(parsed.normalize)

        parsed = parser.parse_args(["cluster", "--min-cluster-size", "7"])
        self.assertEqual(parsed.min_cluster_size, 7)


if __name__ == "__main__":
    unittest.main()
