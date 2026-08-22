# agentkit

A minimal, extensible framework for building tool-using AI agents.

## Why this exists

Most agent frameworks bury the core loop under layers of abstraction. This
one keeps it small on purpose:

-  **250 lines of core code across 4 files** (`agent.py`, `llm.py`, `tools.py`, `memory.py`)
- **Provider-agnostic** — the agent talks to an `LLMProvider` interface, not a specific vendor SDK
- **Two ways to define tools** — a `@tool` decorator for quick functions, or a `Tool` base class for stateful ones
- **No required backend services** — memory is in-process by default; swap in your own store if you need persistence

## Install

```bash
git clone https://github.com/yourname/agentkit.git
cd agentkit
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

Or install as a package:

```bash
pip install -e .
```

## Quick start

```python
from agentkit import Agent, AnthropicProvider, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."

agent = Agent(
    llm=AnthropicProvider(model="claude-sonnet-4-6"),
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

print(agent.run("What's the weather in Lisbon?"))
```

Run the bundled examples:

```bash
python examples/simple_agent.py       # scripted demo (fixed questions)
python examples/custom_tool_class.py  # stateful tool example
python examples/chat.py               # interactive — type your own questions
```

## How it works

```
User input
    │
    ▼
Agent.run() ──► LLMProvider.generate() ──► model responds
    │                                          │
    │                              ┌───────────┴───────────┐
    │                              ▼                        ▼
    │                        wants a tool?             plain text?
    │                              │                        │
    │                              ▼                        ▼
    │                    execute Tool.run() ──►       return to caller
    │                    feed result back into
    │                    the conversation, loop
    ▼
(repeats until the model stops calling tools, or max_steps is hit)
```

## Project structure

```
agentkit/
├── agentkit/
│   ├── agent.py      # Core think-act loop
│   ├── llm.py         # LLMProvider interface + AnthropicProvider
│   ├── tools.py        # Tool base class + @tool decorator
│   └── memory.py        # Simple message history store
├── examples/
│   ├── simple_agent.py         # @tool decorator, two stateless tools
│   ├── custom_tool_class.py    # Tool subclass with internal state
│   └── chat.py                 # interactive loop — type your own questions
├── tests/
│   └── test_agent.py    # Runs against a fake LLM — no API key needed
└── requirements.txt
```

 ## Defining tools

**Quick, stateless tools** — use the decorator; the schema is inferred from type hints and the docstring:

```python
@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)
```

**Stateful or complex tools** — subclass `Tool`:

```python
from agentkit import Tool

class Notepad(Tool):
    name = "notepad"
    description = "Save or list notes."

    def __init__(self):
        self._notes = []

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save", "list"]},
                "note": {"type": "string"},
            },
            "required": ["action"],
        }

    def run(self, action, note=""):
        ...
```

## Using a different LLM backend

Implement `LLMProvider.generate()` against any API and pass it to `Agent`:

```python
from agentkit.llm import LLMProvider, LLMResponse

class MyProvider(LLMProvider):
    def generate(self, messages, system="", tools=None, max_tokens=1024):
        # call your backend, normalize its output
        return LLMResponse(text="...", tool_calls=[...])
```

## Testing

Tests run against a scripted fake provider, so no API key or network access
is required:

```bash
python tests/test_agent.py
```

## Roadmap ideas

- [ ] Streaming responses
- [ ] Multi-agent orchestration (agents calling agents)
- [ ] Persistent memory backends (SQLite, vector store)
- [ ] Async support
- [ ] Built in retry/backoff for rate limits

## License

MIT see [LICENSE](LICENSE)
