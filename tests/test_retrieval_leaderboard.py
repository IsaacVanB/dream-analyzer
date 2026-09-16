from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dream_analysis.retrieval_leaderboard import (
    build_leaderboard,
    markdown_leaderboard,
    write_leaderboard,
)


def report(
    *,
    name: str,
    created_at: str,
    query_fingerprint: str,
    dream_fingerprint: str,
    r_precision: float,
    recall_at_5: float,
    recall_at_10: float,
    routing: float | None,
    errors: int,
    latency: float,
    category_r_precision: float,
) -> dict:
    return {
        "created_at": created_at,
        "experiment": {"name": name, "note": f"note for {name}"},
        "fingerprints": {
            "queries_sha256": query_fingerprint,
            "dreams_sha256": dream_fingerprint,
        },
        "settings": {"retrieval_mode": "agent", "chat_model": "test"},
        "summary": {
            "query_count": 2,
            "evaluated_query_count": 2 - errors,
            "error_query_count": errors,
            "r_precision": r_precision,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "strategy_accuracy": routing,
        },
        "category_summaries": {
            "literal": {"r_precision": category_r_precision}
        },
        "queries": [
            {"status": "ok", "retrieval_seconds": latency},
            {"status": "ok", "retrieval_seconds": latency},
        ],
    }


class RetrievalLeaderboardTests(unittest.TestCase):
    def test_groups_compatible_runs_and_highlights_each_best_metric(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = report(
                name="semantic baseline",
                created_at="2026-09-15T10:00:00-04:00",
                query_fingerprint="q" * 64,
                dream_fingerprint="d" * 64,
                r_precision=0.5,
                recall_at_5=0.8,
                recall_at_10=0.7,
                routing=0.7,
                errors=1,
                latency=0.1,
                category_r_precision=0.4,
            )
            second = report(
                name="hybrid experiment",
                created_at="2026-09-15T11:00:00-04:00",
                query_fingerprint="q" * 64,
                dream_fingerprint="d" * 64,
                r_precision=0.7,
                recall_at_5=0.6,
                recall_at_10=0.9,
                routing=0.9,
                errors=0,
                latency=0.2,
                category_r_precision=0.8,
            )
            incompatible = report(
                name="old suite",
                created_at="2026-09-14T09:00:00-04:00",
                query_fingerprint="x" * 64,
                dream_fingerprint="d" * 64,
                r_precision=0.99,
                recall_at_5=0.99,
                recall_at_10=0.99,
                routing=0.99,
                errors=0,
                latency=0.01,
                category_r_precision=0.99,
            )
            for filename, payload in (
                ("benchmark_semantic.json", first),
                ("benchmark_hybrid.json", second),
                ("benchmark_old.json", incompatible),
            ):
                (root / filename).write_text(json.dumps(payload), encoding="utf-8")
                (root / filename.replace(".json", ".md")).write_text(
                    "report", encoding="utf-8"
                )

            leaderboard = build_leaderboard(root)
            markdown = markdown_leaderboard(leaderboard)

        self.assertEqual(leaderboard["experiment_count"], 3)
        self.assertEqual(len(leaderboard["groups"]), 2)
        current = next(group for group in leaderboard["groups"] if group["current"])
        self.assertEqual(len(current["experiments"]), 2)
        self.assertEqual(
            current["best"]["r_precision"]["winner_names"],
            ["hybrid experiment"],
        )
        self.assertEqual(
            current["best"]["recall_at_5"]["winner_names"],
            ["semantic baseline"],
        )
        self.assertEqual(
            current["best"]["mean_retrieval_seconds"]["winner_names"],
            ["semantic baseline"],
        )
        self.assertIn("**0.700**", markdown)
        self.assertIn("[hybrid experiment](<benchmark_hybrid.md>)", markdown)
        self.assertIn("Older or incompatible experiment groups", markdown)
        self.assertIn("| Experiment | Mode | Created |", markdown)
        self.assertNotIn("P@5", markdown)
        self.assertNotIn("P@10", markdown)

    def test_configured_current_suite_does_not_promote_incompatible_history(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = report(
                name="old suite",
                created_at="2026-09-14T09:00:00-04:00",
                query_fingerprint="o" * 64,
                dream_fingerprint="d" * 64,
                r_precision=0.99,
                recall_at_5=0.99,
                recall_at_10=0.99,
                routing=0.99,
                errors=0,
                latency=0.01,
                category_r_precision=0.99,
            )
            (root / "benchmark_old.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            leaderboard = build_leaderboard(
                root,
                current_query_fingerprint="n" * 64,
                current_dream_fingerprint="d" * 64,
            )
            markdown = markdown_leaderboard(leaderboard)

        self.assertFalse(any(group["current"] for group in leaderboard["groups"]))
        self.assertIn("No experiments match the currently configured", markdown)
        self.assertIn("Older or incompatible experiment groups", markdown)

    def test_writer_rebuilds_json_and_markdown_and_skips_bad_reports(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = report(
                name="valid",
                created_at="2026-09-15T11:00:00-04:00",
                query_fingerprint="q" * 64,
                dream_fingerprint="d" * 64,
                r_precision=0.5,
                recall_at_5=0.5,
                recall_at_10=0.5,
                routing=None,
                errors=0,
                latency=0.2,
                category_r_precision=0.5,
            )
            (root / "benchmark_valid.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (root / "benchmark_broken.json").write_text("{", encoding="utf-8")

            json_path, markdown_path = write_leaderboard(root)
            leaderboard = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertTrue(markdown_path.is_file())
            self.assertEqual(leaderboard["experiment_count"], 1)
            self.assertEqual(
                leaderboard["skipped_reports"][0]["file"],
                "benchmark_broken.json",
            )


if __name__ == "__main__":
    unittest.main()
