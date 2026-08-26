from __future__ import annotations

import unittest

import cluster_dreams
import evaluate_retrieval
import structure_dreams


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
