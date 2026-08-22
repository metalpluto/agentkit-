"""
Core Agent: a loop that lets an LLM call tools until it produces a final answer.

Flow per `.run(user_input)` call:
  1. Add user input to memory.
  2. Ask the LLM to respond, offering it the registered tools.
  3. If the LLM asks to call a tool: run it, feed the result back, repeat.
  4. If the LLM returns plain text: that's the final answer, stop.

A `max_steps` guard prevents infinite tool-calling loops.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMProvider
from .memory import Memory
from .tools import Tool


class AgentError(Exception):
    """Raised when the agent can't complete a run (e.g. step limit hit)."""


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        tools: list[Tool] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        memory: Memory | None = None,
        max_steps: int = 8,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.system_prompt = system_prompt
        self.memory = memory if memory is not None else Memory()
        self.max_steps = max_steps
        self.verbose = verbose

    def add_tool(self, tool_instance: Tool) -> None:
        self.tools[tool_instance.name] = tool_instance

    def _log(self, *args: Any) -> None:
        if self.verbose:
            print("[agent]", *args)

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [t.to_llm_schema() for t in self.tools.values()]

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: no such tool '{name}'"
        try:
            self._log(f"calling tool '{name}' with {tool_input}")
            result = self.tools[name].run(**tool_input)
            self._log(f"tool '{name}' returned: {result}")
            return result
        except Exception as e:  # noqa: BLE001 - surface tool errors to the model
            return f"Error running tool '{name}': {e}"

    def run(self, user_input: str) -> str:
        """Run the agent to completion on a single user turn and return the final text answer."""
        self.memory.add("user", user_input)

        for step in range(self.max_steps):
            response = self.llm.generate(
                messages=self.memory.messages,
                system=self.system_prompt,
                tools=self._tool_schemas() if self.tools else None,
            )

            if not response.tool_calls:
                self.memory.add("assistant", response.text)
                return response.text

            # Record the assistant's tool-use turn, then run each requested tool.
            assistant_content = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for call in response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["input"],
                    }
                )
            self.memory.add("assistant", assistant_content)

            tool_results = []
            for call in response.tool_calls:
                result_text = self._execute_tool(call["name"], call["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": result_text,
                    }
                )
            self.memory.add("user", tool_results)

        raise AgentError(f"Agent did not finish within {self.max_steps} steps.")

    def to_json_transcript(self) -> str:
        """Handy for debugging/logging a full run."""
        return json.dumps(self.memory.messages, indent=2, default=str)
