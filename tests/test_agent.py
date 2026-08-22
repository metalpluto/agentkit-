"""
Tests run without any real API key by using a scripted FakeProvider
that mimics LLMProvider's interface.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit import Agent, Memory, tool
from agentkit.llm import LLMProvider, LLMResponse


class FakeProvider(LLMProvider):
    """Replays a scripted sequence of responses, one per .generate() call."""

    def __init__(self, script: list[LLMResponse]):
        self.script = script
        self.calls = 0

    def generate(self, messages, system="", tools=None, max_tokens=1024):
        response = self.script[self.calls]
        self.calls += 1
        return response


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def test_agent_returns_plain_text_with_no_tool_call():
    provider = FakeProvider([LLMResponse(text="Hello there!")])
    agent = Agent(llm=provider, tools=[add])
    result = agent.run("Hi")
    assert result == "Hello there!"
    assert provider.calls == 1


def test_agent_executes_tool_then_returns_final_answer():
    provider = FakeProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[{"id": "call_1", "name": "add", "input": {"a": 2, "b": 3}}],
            ),
            LLMResponse(text="2 + 3 is 5."),
        ]
    )
    agent = Agent(llm=provider, tools=[add])
    result = agent.run("What's 2 + 3?")
    assert result == "2 + 3 is 5."
    assert provider.calls == 2

    # tool result should have made it into memory
    tool_result_messages = [
        m for m in agent.memory.messages
        if isinstance(m["content"], list)
        and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
    ]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0]["content"][0]["content"] == "5"


def test_agent_raises_after_max_steps():
    # Every response asks for another tool call, so it never terminates.
    infinite_tool_call = LLMResponse(
        text="", tool_calls=[{"id": "x", "name": "add", "input": {"a": 1, "b": 1}}]
    )
    provider = FakeProvider([infinite_tool_call] * 10)
    agent = Agent(llm=provider, tools=[add], max_steps=3)
    try:
        agent.run("loop forever")
        assert False, "expected AgentError"
    except Exception as e:
        assert "did not finish" in str(e)


def test_memory_trims_to_max_messages():
    mem = Memory(max_messages=3)
    for i in range(5):
        mem.add("user", f"msg {i}")
    assert len(mem) == 3
    assert mem.messages[0]["content"] == "msg 2"
    assert mem.messages[-1]["content"] == "msg 4"


def test_tool_decorator_builds_schema():
    schema = add.to_llm_schema()
    assert schema["name"] == "add"
    assert schema["input_schema"]["required"] == ["a", "b"]
    assert schema["input_schema"]["properties"]["a"]["type"] == "integer"


if __name__ == "__main__":
    import inspect

    failures = 0
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    if failures:
        sys.exit(1)
    print(f"\n{len(tests)} tests passed.")
