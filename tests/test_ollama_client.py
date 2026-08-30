from __future__ import annotations

import unittest

from dream_analysis.config import OllamaSettings
from dream_analysis.ollama_client import OllamaGateway, OllamaResponseError


class FakeOllamaClient:
    def __init__(self, *, chat_content: str = "answer") -> None:
        self.embed_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.chat_content = chat_content

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
        return {"message": {"content": self.chat_content}}


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

    def test_chat_json_forwards_schema_and_decodes_content(self) -> None:
        client = FakeOllamaClient(chat_content='{"value": 3}')
        gateway = OllamaGateway(client=client)
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }

        result = gateway.chat_json(
            [{"role": "user", "content": "Return a value"}],
            schema=schema,
            model="structured-model",
            options={"temperature": 0},
        )

        self.assertEqual(result, {"value": 3})
        self.assertEqual(client.chat_calls[0]["format"], schema)
        self.assertEqual(client.chat_calls[0]["model"], "structured-model")

    def test_chat_json_rejects_invalid_json(self) -> None:
        gateway = OllamaGateway(client=FakeOllamaClient(chat_content="not json"))

        with self.assertRaisesRegex(OllamaResponseError, "invalid JSON"):
            gateway.chat_json(
                [{"role": "user", "content": "Return JSON"}],
                schema={"type": "object"},
            )

    def test_tool_calls_and_assistant_message_are_normalized(self) -> None:
        response = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": {"query": "hidden room"},
                        }
                    }
                ],
            }
        }

        calls = OllamaGateway.tool_calls(response)
        message = OllamaGateway.assistant_message(response)

        self.assertEqual(calls[0].name, "search_dreams")
        self.assertEqual(calls[0].arguments, {"query": "hidden room"})
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "")
        self.assertEqual(
            message["tool_calls"][0]["function"]["name"],
            "search_dreams",
        )

    def test_invalid_tool_call_arguments_are_rejected(self) -> None:
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": "not-an-object",
                        }
                    }
                ]
            }
        }

        with self.assertRaisesRegex(OllamaResponseError, "invalid arguments"):
            OllamaGateway.tool_calls(response)

    def test_response_diagnostics_normalize_generation_metadata(self) -> None:
        diagnostics = OllamaGateway.response_diagnostics(
            {
                "model": "qwen3:8b",
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 128,
                "eval_count": 64,
                "message": {
                    "content": "",
                    "thinking": "I should synthesize the searches.",
                },
            }
        )

        self.assertEqual(diagnostics["model"], "qwen3:8b")
        self.assertEqual(diagnostics["done_reason"], "length")
        self.assertEqual(diagnostics["eval_count"], 64)
        self.assertEqual(
            diagnostics["thinking"],
            "I should synthesize the searches.",
        )


if __name__ == "__main__":
    unittest.main()
