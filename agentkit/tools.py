"""
Tools are how the agent affects/queries the world beyond generating text.

Two ways to define one:

1. Subclass `Tool` for anything stateful or complex.
2. Use the `@tool` decorator on a plain function for the common case.
"""

from __future__ import annotations

import abc
import inspect
from typing import Any, Callable, get_type_hints


_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool(abc.ABC):
    """Base class for a tool the agent can call."""

    name: str = ""
    description: str = ""

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result (or JSON-serializable text)."""
        raise NotImplementedError

    def parameters_schema(self) -> dict[str, Any]:
        """Override to declare a JSON-schema `input_schema` for the LLM.
        Default: no parameters."""
        return {"type": "object", "properties": {}, "required": []}

    def to_llm_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema(),
        }


class _FunctionTool(Tool):
    """Wraps a plain function as a Tool. Built by the @tool decorator."""

    def __init__(self, fn: Callable, name: str, description: str, schema: dict[str, Any]):
        self.fn = fn
        self.name = name
        self.description = description
        self.schema = schema

    def run(self, **kwargs: Any) -> str:
        result = self.fn(**kwargs)
        return str(result)

    def parameters_schema(self) -> dict[str, Any]:
        return self.schema


def tool(fn: Callable) -> Tool:
    """
    Decorator that turns a function into a Tool instance.

    The function's docstring becomes the tool description, and its type
    hints (str, int, float, bool) are used to build a parameter schema.
    Parameters without a default are marked required.

        @tool
        def get_weather(city: str) -> str:
            '''Get the current weather for a city.'''
            ...
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(py_type, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema = {"type": "object", "properties": properties, "required": required}
    description = (fn.__doc__ or "").strip()

    return _FunctionTool(fn=fn, name=fn.__name__, description=description, schema=schema)
