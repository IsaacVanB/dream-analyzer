"""Single adapter for interactions with the Ollama Python client."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import ollama

from dream_analysis.config import OllamaSettings


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns a response with an unexpected shape."""


@dataclass(frozen=True, slots=True)
class OllamaToolCall:
    """One normalized function call requested by a chat model."""

    name: str
    arguments: Mapping[str, Any]


class OllamaGateway:
    """Provide model-overridable chat and embedding operations.

    The optional client makes the boundary straightforward to fake in tests and
    keeps callers independent of the concrete Ollama SDK response classes.
    """

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or OllamaSettings()
        self._client = client or ollama.Client(
            host=self.settings.host,
            timeout=self.settings.request_timeout_seconds,
        )

    def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        """Embed one non-empty string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text to embed cannot be empty")
        return self.embed_many([text], model=model)[0]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed several strings in one Ollama request."""
        inputs = list(texts)
        if not inputs:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("texts to embed must all be non-empty strings")

        response = self._client.embed(
            model=model or self.settings.embedding_model,
            input=inputs,
        )
        raw_embeddings = self._response_value(response, "embeddings")
        if not isinstance(raw_embeddings, Sequence) or isinstance(
            raw_embeddings, (str, bytes)
        ):
            raise OllamaResponseError("Ollama returned no embeddings array")
        if len(raw_embeddings) != len(inputs):
            raise OllamaResponseError(
                f"Ollama returned {len(raw_embeddings)} embeddings for "
                f"{len(inputs)} inputs"
            )

        embeddings: list[list[float]] = []
        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, Sequence) or isinstance(
                raw_embedding, (str, bytes)
            ):
                raise OllamaResponseError("Ollama returned an invalid embedding")
            try:
                embedding = [float(value) for value in raw_embedding]
            except (TypeError, ValueError) as exc:
                raise OllamaResponseError(
                    "Ollama returned a non-numeric embedding"
                ) from exc
            if not embedding:
                raise OllamaResponseError("Ollama returned an empty embedding")
            embeddings.append(embedding)
        return embeddings

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        tools: Sequence[Any] | None = None,
        format: str | Mapping[str, Any] | None = None,
        think: bool | str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Send a non-streaming chat request, optionally with callable tools."""
        return self._client.chat(
            model=model or self.settings.chat_model,
            messages=list(messages),
            tools=list(tools) if tools else None,
            format=format,
            think=think,
            options=options,
        )

    def chat_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        schema: Mapping[str, Any],
        model: str | None = None,
        think: bool | str | None = False,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Request schema-constrained output and decode its JSON content.

        Domain-specific validation remains with the caller because JSON Schema
        generation constraints do not replace application-level checks.
        """
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("schema must be a non-empty mapping")
        response = self.chat(
            messages,
            model=model,
            format=schema,
            think=think,
            options=options,
        )
        content = self.message_content(response)
        if not content.strip():
            raise OllamaResponseError("Ollama returned empty structured content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                f"Ollama returned invalid JSON: {exc.msg}"
            ) from exc

    @classmethod
    def message_content(cls, response: Any) -> str:
        """Extract text content from a mapping or SDK chat response."""
        message = cls._response_value(response, "message")
        content = cls._response_value(message, "content")
        if content is None:
            raise OllamaResponseError("Ollama returned no chat message content")
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama returned non-text chat content")
        return content

    @classmethod
    def done_reason(cls, response: Any) -> str | None:
        """Return Ollama's completion reason when one was supplied."""
        value = cls._response_value(response, "done_reason")
        if value is None:
            return None
        return str(value)

    @classmethod
    def response_diagnostics(cls, response: Any) -> dict[str, Any]:
        """Return JSON-compatible response metadata useful for agent tracing."""
        diagnostics: dict[str, Any] = {}
        for key in (
            "model",
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            value = cls._response_value(response, key)
            if value is not None:
                diagnostics[key] = cls._json_safe(value)

        message = cls._response_value(response, "message")
        thinking = cls._response_value(message, "thinking")
        if thinking is not None:
            diagnostics["thinking"] = cls._json_safe(thinking)
        return diagnostics

    @classmethod
    def tool_calls(cls, response: Any) -> list[OllamaToolCall]:
        """Normalize tool calls from mapping or Ollama SDK responses."""
        message = cls._response_value(response, "message")
        if message is None:
            raise OllamaResponseError("Ollama returned no chat message")
        raw_calls = cls._response_value(message, "tool_calls")
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise OllamaResponseError("Ollama returned invalid tool calls")

        calls: list[OllamaToolCall] = []
        for raw_call in raw_calls:
            function = cls._response_value(raw_call, "function")
            name = cls._response_value(function, "name")
            arguments = cls._response_value(function, "arguments")
            if not isinstance(name, str) or not name.strip():
                raise OllamaResponseError("Ollama returned a tool call without a name")
            if not isinstance(arguments, Mapping):
                raise OllamaResponseError(
                    f"Ollama returned invalid arguments for tool {name!r}"
                )
            calls.append(OllamaToolCall(name=name, arguments=dict(arguments)))
        return calls

    @classmethod
    def assistant_message(cls, response: Any) -> dict[str, Any]:
        """Return a response message in the shape accepted by Ollama chat."""
        message = cls._response_value(response, "message")
        if message is None:
            raise OllamaResponseError("Ollama returned no chat message")
        content = cls._response_value(message, "content")
        if content is not None and not isinstance(content, str):
            raise OllamaResponseError("Ollama returned non-text chat content")

        normalized: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }
        calls = cls.tool_calls(response)
        if calls:
            normalized["tool_calls"] = [
                {
                    "function": {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                    }
                }
                for call in calls
            ]
        return normalized

    @staticmethod
    def _response_value(response: Any, key: str) -> Any:
        if isinstance(response, Mapping):
            return response.get(key)
        return getattr(response, key, None)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): OllamaGateway._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [OllamaGateway._json_safe(item) for item in value]
        return str(value)
