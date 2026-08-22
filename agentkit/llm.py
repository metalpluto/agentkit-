"""
LLM provider abstraction.

The agent core never talks to a vendor SDK directly — it talks to an
LLMProvider. This keeps the framework swappable: implement `.generate()`
against any backend (Anthropic, OpenAI, a local model, etc.) and the
rest of the framework doesn't need to change.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Normalized response returned by every provider."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    stop_reason: str | None = None


class LLMProvider(abc.ABC):
    """Base class every model backend must implement."""

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send messages to the model and return a normalized response."""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Default provider backed by the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from e

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
            stop_reason=response.stop_reason,
        )
