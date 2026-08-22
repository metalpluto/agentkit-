"""
Basic example: an agent with two tools (calculator, fake weather lookup).

Run with:
    export ANTHROPIC_API_KEY=sk-...
    python examples/simple_agent.py
"""

from agentkit import Agent, AnthropicProvider, tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 1)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    return str(eval(expression))  # noqa: S307 - restricted to arithmetic chars above


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. (Stubbed for this example.)"""
    fake_data = {
        "san francisco": "58F, foggy",
        "new york": "72F, clear",
        "tokyo": "81F, humid",
    }
    return fake_data.get(city.lower(), f"No weather data for {city}.")


def main():
    agent = Agent(
        llm=AnthropicProvider(model="claude-sonnet-4-6"),
        tools=[calculator, get_weather],
        system_prompt="You are a concise, helpful assistant. Use tools when they help.",
        verbose=True,
    )

    questions = [
        "What's 17 * 24, and what's the weather like in Tokyo?",
        "Thanks! One more: what's (150 - 30) / 4?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        answer = agent.run(q)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()
