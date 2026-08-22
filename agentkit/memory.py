"""
Minimal conversation memory.

Kept intentionally simple — a list of messages plus optional trimming.
Swap this out (or subclass it) for vector-store-backed or summarizing
memory in more advanced setups; the Agent only relies on `.messages`,
`.add()`, and `.clear()`.
"""

from __future__ import annotations

from typing import Any


class Memory:
    def __init__(self, max_messages: int | None = None):
        self.messages: list[dict[str, Any]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": content})
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            # Keep the most recent messages, dropping from the front.
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)
