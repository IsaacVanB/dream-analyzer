from __future__ import annotations

import unittest

from dream_analysis.config import OllamaSettings
from dream_analysis.ollama_client import OllamaGateway, OllamaResponseError


class FakeOllamaClient:
    def __init__(self) -> None:
        self.embed_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        return {
            "embeddings": [
                [float(index), float(len(text))]
                for index, text in enumerate(kwargs["input"], start=1)
            ]
        }

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {"message": {"content": "answer"}}


class OllamaGatewayTests(unittest.TestCase):
    def test_model_can_be_overridden_without_changing_settings(self) -> None:
        client = FakeOllamaClient()
        gateway = OllamaGateway(
            OllamaSettings(embedding_model="default-embed"),
            client=client,
        )

        embeddings = gateway.embed_many(["one", "three"], model="test-embed")

        self.assertEqual(embeddings, [[1.0, 3.0], [2.0, 5.0]])
        self.assertEqual(client.embed_calls[0]["model"], "test-embed")

    def test_chat_forwards_tools_and_model_override(self) -> None:
        client = FakeOllamaClient()
        gateway = OllamaGateway(client=client)
        tool = {"type": "function", "function": {"name": "search"}}

        response = gateway.chat(
            [{"role": "user", "content": "Search"}],
            model="test-chat",
            tools=[tool],
            think=False,
        )

        self.assertEqual(response["message"]["content"], "answer")
        self.assertEqual(client.chat_calls[0]["model"], "test-chat")
        self.assertEqual(client.chat_calls[0]["tools"], [tool])

    def test_embedding_count_must_match_input_count(self) -> None:
        class ShortResponseClient(FakeOllamaClient):
            def embed(self, **kwargs):
                return {"embeddings": [[1.0]]}

        gateway = OllamaGateway(client=ShortResponseClient())
        with self.assertRaisesRegex(OllamaResponseError, "1 embeddings for 2"):
            gateway.embed_many(["one", "two"])


if __name__ == "__main__":
    unittest.main()

