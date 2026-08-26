"""Single adapter for interactions with the Ollama Python client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import ollama

from dream_analysis.config import OllamaSettings


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns a response with an unexpected shape."""


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

    @staticmethod
    def _response_value(response: Any, key: str) -> Any:
        if isinstance(response, Mapping):
            return response.get(key)
        return getattr(response, key, None)
