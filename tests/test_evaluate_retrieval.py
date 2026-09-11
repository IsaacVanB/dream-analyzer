from __future__ import annotations

import unittest

from cli import evaluate_retrieval
from dream_analysis.agent import AgentResponse, ToolExecution


class RetrievalMetricTests(unittest.TestCase):
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

        rows = evaluate_retrieval.evaluate_queries(
            [
                {
                    "query": "original query",
                    "category": "test",
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
        self.assertEqual(rows[0]["retrieved_dream_ids"][0], "shared")
        self.assertEqual(rows[0]["r_precision"], 1.0)
        self.assertEqual(rows[0]["tool_calls"][0]["arguments"], {"query": "one"})

    def test_markdown_reports_maximum_precision(self) -> None:
        row = {
            "query": "dogs",
            "category": "direct",
            **evaluate_retrieval.retrieval_metrics(["a"], ["a", "b"]),
        }
        summary = evaluate_retrieval.summarize([row])
        report = {
            "created_at": "2026-09-11T12:00:00-04:00",
            "queries_path": "queries.json",
            "settings": {
                "collection_name": "dreams",
                "embed_model": "embed",
                "chat_model": "chat",
                "max_tool_calls": 3,
                "results_per_semantic_search": 10,
            },
            "summary": summary,
            "category_summaries": {"direct": summary},
            "queries": [row],
        }
        row["tool_calls"] = []
        row["unexecuted_tool_calls"] = []

        markdown = evaluate_retrieval.markdown_report(report)

        self.assertIn("max P@5", markdown)
        self.assertIn("max P@10", markdown)
        self.assertIn("R-precision", markdown)
        self.assertIn("| dogs | direct | 2 |", markdown)


if __name__ == "__main__":
    unittest.main()
