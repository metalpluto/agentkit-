"""
Interactive chat with the agent — type your own questions instead of
running a fixed script.

Run with:
    export ANTHROPIC_API_KEY=sk-...      (macOS/Linux)
    $env:ANTHROPIC_API_KEY="sk-..."       (Windows PowerShell)
    python examples/chat.py

Type 'exit' or 'quit' to stop.
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
    """Get the current weather for a city. (Stubbed with fake data for this example.)"""
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
        verbose=True,  # set to False if you don't want to see the tool-call logs
    )

    print("Chat with the agent. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = agent.run(user_input)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
