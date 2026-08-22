"""
Example: defining a stateful tool via the Tool base class instead of
the @tool decorator. Useful when a tool needs setup (API clients, DB
connections, internal counters, etc).

Run with:
    export ANTHROPIC_API_KEY=sk-...
    python examples/custom_tool_class.py
"""

from agentkit import Agent, AnthropicProvider, Tool


class Notepad(Tool):
    """A tool with internal state: it remembers notes across calls."""

    name = "notepad"
    description = "Save a short note to memory, or list all saved notes."

    def __init__(self):
        self._notes: list[str] = []

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save", "list"]},
                "note": {"type": "string"},
            },
            "required": ["action"],
        }

    def run(self, action: str, note: str = "") -> str:
        if action == "save":
            self._notes.append(note)
            return f"Saved. You now have {len(self._notes)} note(s)."
        if action == "list":
            if not self._notes:
                return "No notes saved yet."
            return "\n".join(f"- {n}" for n in self._notes)
        return f"Unknown action: {action}"


def main():
    agent = Agent(
        llm=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[Notepad()],
        system_prompt="You help the user keep notes. Use the notepad tool as needed.",
        verbose=True,
    )

    print(agent.run("Save a note: 'Buy milk on the way home.'"))
    print(agent.run("Also note: 'Call the dentist tomorrow.'"))
    print(agent.run("What notes do I have saved?"))


if __name__ == "__main__":
    main()
