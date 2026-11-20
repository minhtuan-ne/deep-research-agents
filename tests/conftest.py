"""Shared fixtures: a scripted fake chat model so the graph runs offline, without API keys."""

from typing import Any, Callable

import pytest

from deep_research_agents import graph as graph_module


class FakeModel:
    """Mimics the slice of the LangChain chat-model API that graph.py uses.

    `handler(schema, tool_names, messages)` decides each response, so a test can
    script how the clarifier, supervisor, researchers, compressor, and writer behave.
    """

    def __init__(self, handler: Callable, schema: Any = None, tools: list | None = None):
        self.handler = handler
        self.schema = schema
        self.tools = tools or []
        self.calls: list[dict] = []

    def _clone(self, **kw) -> "FakeModel":
        clone = FakeModel(self.handler, kw.get("schema", self.schema), kw.get("tools", self.tools))
        clone.calls = self.calls  # share the call log across derived models
        return clone

    def with_structured_output(self, schema):
        return self._clone(schema=schema)

    def bind_tools(self, tools):
        return self._clone(tools=tools)

    def with_retry(self, **_):
        return self

    def with_config(self, *_args, **_kw):
        return self

    @property
    def tool_names(self) -> set[str]:
        return {getattr(t, "name", None) or t.__name__ for t in self.tools}

    async def ainvoke(self, messages, *_args, **_kw):
        self.calls.append({"schema": self.schema, "tools": self.tool_names, "messages": messages})
        return self.handler(self.schema, self.tool_names, messages)


@pytest.fixture
def fake_model(monkeypatch):
    """Install a FakeModel driven by the given handler in place of the real chat model."""

    def install(handler: Callable) -> FakeModel:
        model = FakeModel(handler)
        monkeypatch.setattr(graph_module, "configurable_model", model)
        return model

    return install


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Configuration reads env vars first; make sure a local .env can't leak into tests."""
    from deep_research_agents.configuration import Configuration

    for field in Configuration.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
