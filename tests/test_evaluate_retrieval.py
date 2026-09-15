from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from cli import evaluate_retrieval
from dream_analysis.agent import AgentResponse, ToolExecution
from dream_analysis.bm25 import DreamBm25Index
from dream_analysis.models import Dream


class RetrievalMetricTests(unittest.TestCase):
    @staticmethod
    def preflight_args(root: Path) -> SimpleNamespace:
        chroma_path = root / "chroma"
        chroma_path.mkdir()
        paths = {
            "dreams_path": root / "dreams.jsonl",
            "structured_dreams_path": root / "structured.jsonl",
            "characters_path": root / "characters.json",
        }
        paths["dreams_path"].write_text("", encoding="utf-8")
        paths["structured_dreams_path"].write_text("", encoding="utf-8")
        paths["characters_path"].write_text("[]", encoding="utf-8")
        return SimpleNamespace(
            chroma_path=str(chroma_path),
            collection_name="dreams",
            embed_model="embed",
            **paths,
        )

    def test_preflight_validates_collection_and_tool_files(self) -> None:
        class Collection:
            name = "dreams"
            metadata = {"embedding_model": "embed"}

            @staticmethod
            def count():
                return 12

        class Client:
            @staticmethod
            def list_collections():
                return [Collection()]

            @staticmethod
            def get_collection(*, name):
                self.assertEqual(name, "dreams")
                return Collection()

        with TemporaryDirectory() as temporary_directory:
            args = self.preflight_args(Path(temporary_directory))
            args.structured_dreams_path.unlink()
            result = evaluate_retrieval.preflight(
                args,
                chroma_client=Client(),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["collection_count"], 12)
        self.assertEqual(
            result["data_files"]["structured dreams"]["status"], "unavailable"
        )
        self.assertEqual(
            result["data_files"]["structured dreams"]["effect"],
            "get_character_mentions disabled",
        )

    def test_preflight_reports_available_collection(self) -> None:
        class Collection:
            name = "dreams_with_underscores"

        class Client:
            @staticmethod
            def list_collections():
                return [Collection()]

            @staticmethod
            def get_collection(*, name):
                raise ValueError(f"Collection {name} does not exist")

        with TemporaryDirectory() as temporary_directory:
            args = self.preflight_args(Path(temporary_directory))
            args.structured_dreams_path.unlink()
            with self.assertRaises(evaluate_retrieval.EvaluationPreflightError) as raised:
                evaluate_retrieval.preflight(args, chroma_client=Client())

        message = str(raised.exception)
        self.assertIn("no queries were run", message)
        self.assertNotIn("structured dreams file does not exist", message)
        self.assertIn("Available collections: dreams_with_underscores", message)

    def test_embedding_preflight_does_not_require_agent_data_files(self) -> None:
        class Collection:
            name = "dreams"
            metadata = {"embedding_model": "embed"}

            @staticmethod
            def count():
                return 12

        class Client:
            @staticmethod
            def list_collections():
                return [Collection()]

            @staticmethod
            def get_collection(*, name):
                return Collection()

        with TemporaryDirectory() as temporary_directory:
            args = self.preflight_args(Path(temporary_directory))
            args.retrieval_mode = "embedding"
            args.dreams_path.unlink()
            args.structured_dreams_path.unlink()
            args.characters_path.unlink()
            result = evaluate_retrieval.preflight(args, chroma_client=Client())

        self.assertEqual(result["retrieval_mode"], "embedding")
        self.assertEqual(result["data_files"], {})

    def test_bm25_preflight_requires_only_parsed_dreams(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            args = self.preflight_args(Path(temporary_directory))
            args.retrieval_mode = "bm25"
            Path(args.chroma_path).rmdir()
            args.structured_dreams_path.unlink()
            args.characters_path.unlink()

            result = evaluate_retrieval.preflight(args)

        self.assertEqual(result["retrieval_mode"], "bm25")
        self.assertIsNone(result["collection_count"])
        self.assertEqual(result["available_collections"], [])
        self.assertEqual(result["data_files"]["parsed dreams"]["record_count"], 0)

    def test_query_loader_requires_a_known_expected_strategy(self) -> None:
        payload = {
            "num_queries": 1,
            "queries": [
                {
                    "query": "dreams about dogs",
                    "category": "direct",
                    "relevant_dream_ids": ["dog-dream"],
                }
            ],
        }
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "queries.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_strategy"):
                evaluate_retrieval.load_evaluation_queries(path)

            payload["queries"][0]["expected_strategy"] = "unknown"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one of"):
                evaluate_retrieval.load_evaluation_queries(path)

            payload["queries"][0]["expected_strategy"] = "semantic_or_hybrid"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = evaluate_retrieval.load_evaluation_queries(path)

        self.assertEqual(
            loaded["queries"][0]["expected_strategy"],
            "semantic_or_hybrid",
        )

    def test_strategy_classification_and_alternative_labels(self) -> None:
        self.assertEqual(
            evaluate_retrieval.infer_agent_strategy(["search_dreams"]),
            "semantic",
        )
        self.assertEqual(
            evaluate_retrieval.infer_agent_strategy(
                ["search_dreams", "search_dreams_by_keywords"]
            ),
            "hybrid",
        )
        self.assertEqual(
            evaluate_retrieval.infer_agent_strategy(["get_character_mentions"]),
            "reasoned_filter",
        )
        accepted = evaluate_retrieval.evaluate_agent_strategy(
            "semantic_or_hybrid",
            ["search_dreams", "search_dreams_by_keywords"],
        )
        rejected = evaluate_retrieval.evaluate_agent_strategy(
            "bm25",
            ["search_dreams"],
        )

        self.assertTrue(accepted["strategy_match"])
        self.assertEqual(accepted["actual_strategy"], "hybrid")
        self.assertFalse(rejected["strategy_match"])

    def test_metrics_at_cutoffs_and_r_precision(self) -> None:
        relevant = ["a", "c", "e"]
        retrieved = ["a", "x", "c", "y", "z", "e"]

        metrics = evaluate_retrieval.retrieval_metrics(retrieved, relevant)

        self.assertEqual(metrics["precision_at_5"], 2 / 5)
        self.assertEqual(metrics["recall_at_5"], 2 / 3)
        self.assertEqual(metrics["max_precision_at_5"], 3 / 5)
        self.assertEqual(metrics["precision_at_10"], 3 / 10)
        self.assertEqual(metrics["recall_at_10"], 1.0)
        self.assertEqual(metrics["max_precision_at_10"], 3 / 10)
        self.assertEqual(metrics["r_precision"], 2 / 3)

    def test_zero_relevant_query_has_defined_precision_only(self) -> None:
        metrics = evaluate_retrieval.retrieval_metrics(["a", "b"], [])

        self.assertEqual(metrics["precision_at_5"], 0.0)
        self.assertEqual(metrics["max_precision_at_5"], 0.0)
        self.assertIsNone(metrics["recall_at_5"])
        self.assertIsNone(metrics["r_precision"])

    def test_parser_accepts_all_retrieval_modes(self) -> None:
        parser = evaluate_retrieval.build_parser()

        agent_args = parser.parse_args([])

        self.assertEqual(agent_args.retrieval_mode, "agent")
        for mode in ("embedding", "bm25", "hybrid"):
            with self.subTest(mode=mode):
                args = parser.parse_args(["--retrieval-mode", mode])
                self.assertEqual(args.retrieval_mode, mode)

    def test_embedding_baseline_embeds_query_verbatim_and_retrieves_to_r(self) -> None:
        calls = []

        def retrieve(query, **kwargs):
            calls.append((query, kwargs))
            return [{"dream_id": "relevant"}]

        with redirect_stdout(StringIO()):
            rows = evaluate_retrieval.evaluate_embedding_queries(
                [
                    {
                        "query": "dreams about exact wording",
                        "category": "direct",
                        "relevant_dream_ids": [
                            "relevant",
                            *[f"other-{index}" for index in range(11)],
                        ],
                    }
                ],
                chroma_path="db",
                collection_name="dreams",
                embed_model="embed",
                retrieve=retrieve,
            )

        self.assertEqual(calls[0][0], "dreams about exact wording")
        self.assertEqual(calls[0][1]["top_k"], 12)
        self.assertEqual(rows[0]["retrieval_query"], "dreams about exact wording")
        self.assertEqual(rows[0]["retrieval_depth"], 12)
        self.assertEqual(rows[0]["tool_calls"], [])

    def test_bm25_baseline_searches_query_verbatim(self) -> None:
        index = DreamBm25Index(
            [
                Dream(
                    dream_id="relevant",
                    date="1/1/2024",
                    text="A green bicycle appeared.",
                ),
                Dream(
                    dream_id="irrelevant",
                    date="1/2/2024",
                    text="A familiar house appeared.",
                ),
            ]
        )

        with redirect_stdout(StringIO()):
            rows = evaluate_retrieval.evaluate_bm25_queries(
                [
                    {
                        "query": "green bicycles",
                        "category": "literal",
                        "relevant_dream_ids": ["relevant"],
                    }
                ],
                index=index,
            )

        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["retrieval_query"], "green bicycles")
        self.assertEqual(rows[0]["retrieval_depth"], 10)
        self.assertEqual(rows[0]["retrieved_dream_ids"], ["relevant"])
        self.assertEqual(rows[0]["r_precision"], 1.0)

    def test_fixed_hybrid_fuses_verbatim_semantic_and_bm25_rankings(self) -> None:
        semantic_calls = []

        def retrieve(query, **kwargs):
            semantic_calls.append((query, kwargs))
            return [
                {"dream_id": "semantic-only"},
                {"dream_id": "shared"},
            ]

        index = DreamBm25Index(
            [
                Dream(
                    dream_id="bm25-only",
                    date="1/1/2024",
                    text="bicycle bicycle bicycle",
                ),
                Dream(
                    dream_id="shared",
                    date="1/2/2024",
                    text="bicycle",
                ),
            ],
            b=0,
        )

        with redirect_stdout(StringIO()):
            rows = evaluate_retrieval.evaluate_hybrid_queries(
                [
                    {
                        "query": "bicycle",
                        "category": "mixed",
                        "relevant_dream_ids": ["shared"],
                    }
                ],
                index=index,
                chroma_path="db",
                collection_name="dreams",
                embed_model="embed",
                retrieve=retrieve,
            )

        self.assertEqual(semantic_calls[0][0], "bicycle")
        self.assertEqual(semantic_calls[0][1]["top_k"], 10)
        self.assertEqual(rows[0]["retrieved_dream_ids"][0], "shared")
        self.assertEqual(
            rows[0]["semantic_dream_ids"],
            ["semantic-only", "shared"],
        )
        self.assertEqual(rows[0]["bm25_dream_ids"], ["bm25-only", "shared"])
        self.assertEqual(rows[0]["r_precision"], 1.0)

    def test_evaluator_uses_agent_and_fuses_its_tool_results(self) -> None:
        class FakeAgent:
            def __init__(self):
                self.calls = []

            def answer(self, query, **kwargs):
                self.calls.append((query, kwargs))
                first = {
                    "ok": True,
                    "dreams": [
                        {"dream_id": "a", "distance": 0.1},
                        {"dream_id": "shared", "distance": 0.2},
                    ],
                }
                second = {
                    "ok": True,
                    "dreams": [
                        {"dream_id": "b", "distance": 0.1},
                        {"dream_id": "shared", "distance": 0.2},
                    ],
                }
                return AgentResponse(
                    answer="",
                    tool_executions=(
                        ToolExecution("search_dreams", {"query": "one"}, first),
                        ToolExecution("search_dreams", {"query": "two"}, second),
                    ),
                )

        agent = FakeAgent()

        output = StringIO()
        with redirect_stdout(output):
            rows = evaluate_retrieval.evaluate_queries(
                [
                    {
                        "query": "original query",
                        "category": "test",
                        "expected_strategy": "semantic",
                        "relevant_dream_ids": ["shared"],
                    },
                ],
                agent=agent,
                chat_model="chat",
                max_tool_calls=3,
                num_ctx=4096,
                num_predict=200,
                temperature=0,
            )

        self.assertTrue(agent.calls[0][1]["synthesize"] is False)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["retrieved_dream_ids"][0], "shared")
        self.assertEqual(rows[0]["r_precision"], 1.0)
        self.assertEqual(rows[0]["expected_strategy"], "semantic")
        self.assertEqual(rows[0]["actual_strategy"], "semantic")
        self.assertTrue(rows[0]["strategy_match"])
        self.assertEqual(rows[0]["tool_calls"][0]["arguments"], {"query": "one"})
        self.assertIn("[1/1] Starting: original query", output.getvalue())
        self.assertIn("[1/1] Finished in", output.getvalue())
        self.assertIn("2 tool call(s), 3 unique dream(s)", output.getvalue())

    def test_tool_failure_marks_query_error_and_excludes_its_metrics(self) -> None:
        class FailingAgent:
            @staticmethod
            def answer(query, **kwargs):
                return AgentResponse(
                    answer="",
                    tool_executions=(
                        ToolExecution(
                            "search_dreams",
                            {"query": query},
                            {"ok": False, "error": "collection missing"},
                        ),
                    ),
                )

        with redirect_stdout(StringIO()):
            rows = evaluate_retrieval.evaluate_queries(
                [
                    {
                        "query": "dogs",
                        "category": "direct",
                        "expected_strategy": "semantic",
                        "relevant_dream_ids": ["a"],
                    }
                ],
                agent=FailingAgent(),
                chat_model="chat",
                max_tool_calls=3,
                num_ctx=4096,
                num_predict=200,
                temperature=0,
            )

        summary = evaluate_retrieval.summarize(rows)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIsNone(rows[0]["precision_at_5"])
        self.assertEqual(summary["evaluated_query_count"], 0)
        self.assertEqual(summary["error_query_count"], 1)
        self.assertIsNone(summary["precision_at_5"])
        self.assertEqual(summary["strategy_evaluated_count"], 1)
        self.assertEqual(summary["strategy_accuracy"], 1.0)

    def test_strategy_summary_counts_only_observable_labeled_routes(self) -> None:
        metrics = evaluate_retrieval.retrieval_metrics([], [])
        rows = [
            {"status": "ok", **metrics, "strategy_match": True},
            {"status": "ok", **metrics, "strategy_match": False},
            {"status": "error", **metrics, "strategy_match": None},
        ]

        summary = evaluate_retrieval.summarize(rows)

        self.assertEqual(summary["strategy_evaluated_count"], 2)
        self.assertEqual(summary["strategy_correct_count"], 1)
        self.assertEqual(summary["strategy_accuracy"], 0.5)

    def test_markdown_reports_maximum_precision(self) -> None:
        row = {
            "query": "dogs",
            "category": "direct",
            "expected_strategy": "semantic",
            "actual_strategy": "semantic",
            "strategy_match": True,
            **evaluate_retrieval.retrieval_metrics(["a"], ["a", "b"]),
        }
        summary = evaluate_retrieval.summarize([row])
        report = {
            "created_at": "2026-09-11T12:00:00-04:00",
            "queries_path": "queries.json",
            "settings": {
                "retrieval_mode": "agent",
                "collection_name": "dreams",
                "embed_model": "embed",
                "chat_model": "chat",
                "max_tool_calls": 3,
                "results_per_semantic_search": 10,
            },
            "preflight": {"status": "ok"},
            "summary": summary,
            "category_summaries": {"direct": summary},
            "queries": [row],
        }
        row["status"] = "ok"
        row["error"] = None
        row["tool_calls"] = []
        row["unexecuted_tool_calls"] = []

        markdown = evaluate_retrieval.markdown_report(report)

        self.assertIn("max P@5", markdown)
        self.assertIn("max P@10", markdown)
        self.assertIn("R-precision", markdown)
        self.assertIn("| dogs | direct | ok | 2 |", markdown)
        self.assertIn("## Agent routing accuracy", markdown)
        self.assertIn("| all | 1 | 1 | 1.000 |", markdown)
        self.assertIn("| dogs | semantic | semantic | yes |", markdown)

    def test_markdown_describes_bm25_and_hybrid_baselines(self) -> None:
        summary = evaluate_retrieval.summarize([])

        def report(mode):
            return {
                "created_at": "2026-09-11T12:00:00-04:00",
                "queries_path": "queries.json",
                "settings": {
                    "retrieval_mode": mode,
                    "collection_name": "dreams",
                    "embed_model": "embed",
                },
                "preflight": {"status": "ok"},
                "summary": summary,
                "category_summaries": {},
                "queries": [],
            }

        bm25_markdown = evaluate_retrieval.markdown_report(report("bm25"))
        hybrid_markdown = evaluate_retrieval.markdown_report(report("hybrid"))

        self.assertIn("sent verbatim to the in-memory BM25 index", bm25_markdown)
        self.assertNotIn("Chroma collection", bm25_markdown)
        self.assertIn("sent verbatim to both Chroma and BM25", hybrid_markdown)
        self.assertIn("reciprocal-rank fusion", hybrid_markdown)


if __name__ == "__main__":
    unittest.main()
