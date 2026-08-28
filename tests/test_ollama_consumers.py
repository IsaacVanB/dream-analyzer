from __future__ import annotations

import unittest
from unittest.mock import patch

from cli import cluster_dreams, evaluate_retrieval, structure_dreams


class FakeStructuredGateway:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.responses.pop(0)


class FakeChatGateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return {"message": {"content": self.content}}

    @staticmethod
    def message_content(response) -> str:
        return response["message"]["content"]


def valid_features() -> dict:
    features = {}
    for name, definition in structure_dreams.DREAM_FEATURE_SCHEMA[
        "properties"
    ].items():
        if definition["type"] == "array":
            features[name] = []
        elif definition["type"] == "boolean":
            features[name] = False
        elif "enum" in definition:
            features[name] = definition["enum"][0]
        else:
            features[name] = "unclear"
    features["lucidity_level"] = "none"
    return features


class OllamaConsumerMigrationTests(unittest.TestCase):
    def test_structured_dream_extraction_uses_gateway_schema(self) -> None:
        gateway = FakeStructuredGateway([valid_features()])

        features = structure_dreams.extract_features(
            {
                "dream_id": "dream-1",
                "date": "1/2/2024",
                "tags": ["house"],
                "text": "I found another room.",
            },
            model="feature-model",
            num_ctx=4096,
            gateway=gateway,
        )

        self.assertEqual(features["lucidity_level"], "none")
        call = gateway.calls[0]
        self.assertEqual(call["schema"], structure_dreams.DREAM_FEATURE_SCHEMA)
        self.assertEqual(call["model"], "feature-model")
        self.assertEqual(call["options"]["num_ctx"], 4096)

    def test_retrieval_focus_generation_uses_gateway_schema(self) -> None:
        gateway = FakeStructuredGateway([{"focus": "distinctive hidden room conflict"}])

        focus = evaluate_retrieval.generate_focus(
            "I found a room.",
            model="judge-model",
            num_ctx=8192,
            gateway=gateway,
        )

        self.assertEqual(focus, "distinctive hidden room conflict")
        self.assertEqual(gateway.calls[0]["schema"], evaluate_retrieval.FOCUS_SCHEMA)
        self.assertEqual(gateway.calls[0]["model"], "judge-model")

    def test_retrieval_evaluation_retains_domain_validation(self) -> None:
        gateway = FakeStructuredGateway(
            [
                {
                    "evaluations": [
                        {
                            "dream_id": "dream-1",
                            "relevance": 4,
                            "generic_overlap": 2,
                            "reason": "The hidden room is central.",
                        }
                    ]
                }
            ]
        )
        retrieved = [
            {
                "dream_id": "dream-1",
                "date": "1/2/2024",
                "distance": 0.2,
                "document": "A hidden room appeared.",
            }
        ]

        evaluations = evaluate_retrieval.evaluate_relevance(
            "hidden room",
            retrieved,
            judge_model="judge-model",
            gateway=gateway,
        )

        self.assertEqual(evaluations[0]["relevance"], 4)
        self.assertEqual(
            gateway.calls[0]["schema"],
            evaluate_retrieval.EVALUATION_SCHEMA,
        )

    def test_metric_parser_preserves_chroma_default_and_accepts_both(self) -> None:
        parser = evaluate_retrieval.build_parser()

        default_args = parser.parse_args(["hidden rooms"])
        comparison_args = parser.parse_args(
            ["hidden rooms", "--retrieval-metric", "both"]
        )

        self.assertEqual(default_args.retrieval_metric, "chroma")
        self.assertEqual(
            evaluate_retrieval.selected_metrics(comparison_args.retrieval_metric),
            ("chroma", "cosine"),
        )

    def test_retrieval_candidates_are_judged_once_across_batches(self) -> None:
        def response(*dream_ids):
            return {
                "evaluations": [
                    {
                        "dream_id": dream_id,
                        "relevance": 4,
                        "generic_overlap": 2,
                        "reason": "Relevant.",
                    }
                    for dream_id in dream_ids
                ]
            }

        gateway = FakeStructuredGateway(
            [response("dream-1", "dream-2"), response("dream-3")]
        )
        retrieved = [
            {
                "dream_id": f"dream-{index}",
                "date": f"1/{index}/2024",
                "document": f"Dream {index}",
            }
            for index in range(1, 4)
        ]

        evaluations = evaluate_retrieval.evaluate_in_batches(
            "hidden rooms",
            retrieved,
            batch_size=2,
            gateway=gateway,
        )

        self.assertEqual(
            [item["dream_id"] for item in evaluations],
            ["dream-1", "dream-2", "dream-3"],
        )
        self.assertEqual(len(gateway.calls), 2)

    def test_metric_retrieval_uses_existing_chroma_and_cosine_paths(self) -> None:
        chroma_results = [
            {
                "dream_id": "target",
                "date": "1/1/2024",
                "distance": 0.0,
                "document": "target text",
            },
            {
                "dream_id": "other",
                "date": "1/2/2024",
                "distance": 0.2,
                "document": "metadata\n--- DREAM TEXT ---\nother text",
            },
        ]
        cosine_results = [
            {
                "dream_id": "cosine-other",
                "date": "1/3/2024",
                "similarity": 0.8,
                "text": "cosine text",
            }
        ]

        with patch.object(
            evaluate_retrieval.basic_rag,
            "retrieve_dreams",
            return_value=chroma_results,
        ) as chroma_retrieve:
            chroma = evaluate_retrieval.retrieve_candidates(
                "target text",
                metric="chroma",
                top_k=1,
                target_dream_id="target",
                chroma_path="db",
                collection_name="dreams",
                embed_model="embed",
            )
        with patch.object(
            evaluate_retrieval.analyze_dream,
            "retrieve_related_dreams",
            return_value=cosine_results,
        ) as cosine_retrieve:
            cosine = evaluate_retrieval.retrieve_candidates(
                "target text",
                metric="cosine",
                top_k=1,
                target_dream_id="target",
                chroma_path="db",
                collection_name="dreams",
                embed_model="embed",
            )

        self.assertEqual(chroma[0]["dream_id"], "other")
        self.assertEqual(chroma[0]["document"], "other text")
        self.assertEqual(cosine[0]["similarity"], 0.8)
        self.assertEqual(chroma_retrieve.call_args.kwargs["top_k"], 2)
        self.assertEqual(
            cosine_retrieve.call_args.kwargs["similarity_threshold"],
            -1.0,
        )

    def test_metric_comparison_reports_overlap_and_relevance_delta(self) -> None:
        embed_model, collection_name = evaluate_retrieval.EMBEDDING_INDEXES[0]

        def retrieval_run(metric, ids, mean_relevance):
            return {
                "embed_model": embed_model,
                "collection_name": collection_name,
                "retrieval_metric": metric,
                "status": "ok",
                "summary": {"mean_relevance": mean_relevance},
                "results": [{"dream_id": dream_id} for dream_id in ids],
            }

        comparisons = evaluate_retrieval.compare_metric_results(
            [
                retrieval_run("chroma", ["a", "b"], 3.5),
                retrieval_run("cosine", ["b", "c"], 4.0),
            ]
        )

        self.assertEqual(comparisons[0]["shared_at_k"], 1)
        self.assertEqual(comparisons[0]["jaccard_overlap"], 0.333)
        self.assertEqual(comparisons[0]["chroma_only"], ["a"])
        self.assertEqual(comparisons[0]["cosine_only"], ["c"])
        self.assertEqual(
            comparisons[0]["mean_relevance_delta_cosine_minus_chroma"],
            0.5,
        )

    def test_metric_markdown_uses_each_native_score_name(self) -> None:
        def run(metric, score_key, score):
            return {
                "embed_model": "embed-model",
                "collection_name": "dreams",
                "retrieval_metric": metric,
                "score_key": score_key,
                "score_direction": (
                    "lower_is_better" if metric == "chroma" else "higher_is_better"
                ),
                "status": "ok",
                "retrieval_seconds": 0.1,
                "summary": {
                    "mean_relevance": 4.0,
                    "median_relevance": 4,
                    "relevant_at_4_or_5": 1,
                },
                "results": [
                    {
                        "rank": 1,
                        "dream_id": "dream-1",
                        "date": "1/1/2024",
                        score_key: score,
                        "relevance": 4,
                        "generic_overlap": 2,
                        "reason": "Relevant.",
                        "text": "Dream text.",
                    }
                ],
            }

        report = {
            "created_at": "2026-08-28T12:00:00-04:00",
            "target": {"retrieval_prompt": "hidden room"},
            "evaluation_focus": "hidden room",
            "focus_source": "prompt",
            "top_k": 1,
            "judge_model": "judge",
            "unique_candidates_judged": 1,
            "evaluation_seconds": 0.2,
            "metric_comparisons": [],
            "embedding_models": [
                run("chroma", "distance", 0.2),
                run("cosine", "similarity", 0.8),
            ],
        }

        markdown = evaluate_retrieval.markdown_report(report)

        self.assertIn("## embed-model — Chroma distance", markdown)
        self.assertIn("| rank | dream_id | date | distance |", markdown)
        self.assertIn("## embed-model — Cosine similarity", markdown)
        self.assertIn("| rank | dream_id | date | similarity |", markdown)

    def test_cluster_label_uses_gateway_and_cleans_first_line(self) -> None:
        gateway = FakeChatGateway('"Hidden Rooms and Doors"\nextra text')

        label = cluster_dreams.llm_label(
            model="label-model",
            terms=["hidden room"],
            tags=[("house", 2.0, 3)],
            texts=["A room appeared behind a door."],
            gateway=gateway,
        )

        self.assertEqual(label, "Hidden Rooms and Doors")
        self.assertEqual(gateway.calls[0]["model"], "label-model")

    def test_empty_cluster_label_falls_back_to_automatic_label(self) -> None:
        gateway = FakeChatGateway("  ")

        label = cluster_dreams.llm_label(
            model="label-model",
            terms=["hidden room"],
            tags=[("house", 2.0, 3)],
            texts=["Dream text"],
            gateway=gateway,
        )

        self.assertEqual(label, "house / hidden room")


if __name__ == "__main__":
    unittest.main()
