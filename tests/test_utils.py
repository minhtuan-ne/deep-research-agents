"""Unit tests for pure helpers, state reducers, and configuration."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agents.configuration import Configuration, SearchAPI
from deep_research_agents.state import override_reducer
from deep_research_agents.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_notes_from_tool_calls,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    web_search,
)


class TestOverrideReducer:
    def test_appends_by_default(self):
        assert override_reducer(["a"], ["b"]) == ["a", "b"]

    def test_override_replaces(self):
        assert override_reducer(["a", "b"], {"type": "override", "value": ["c"]}) == ["c"]

    def test_override_can_clear(self):
        assert override_reducer(["a"], {"type": "override", "value": []}) == []


class TestMessageHelpers:
    def test_remove_up_to_last_ai_message(self):
        msgs = [HumanMessage("q"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2")]
        assert remove_up_to_last_ai_message(msgs) == msgs[:3]

    def test_remove_without_ai_message_is_noop(self):
        msgs = [HumanMessage("q")]
        assert remove_up_to_last_ai_message(msgs) == msgs

    def test_notes_are_tool_message_contents(self):
        msgs = [
            AIMessage("thinking"),
            ToolMessage(content="finding 1", tool_call_id="1"),
            ToolMessage(content="finding 2", tool_call_id="2"),
        ]
        assert get_notes_from_tool_calls(msgs) == ["finding 1", "finding 2"]


class TestTokenLimitDetection:
    def test_detects_known_phrasings(self):
        assert is_token_limit_exceeded(Exception("This model's maximum context length is 128000"))
        assert is_token_limit_exceeded(Exception("prompt is too many tokens"))

    def test_ignores_unrelated_errors(self):
        assert not is_token_limit_exceeded(Exception("rate limit exceeded"))


class TestApiKeyLookup:
    def test_provider_prefix(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        assert get_api_key_for_model("anthropic:claude-sonnet-5-5", {}) == "sk-ant"

    def test_bare_model_defaults_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        assert get_api_key_for_model("gpt-4.1", {}) == "sk-oai"

    def test_unknown_provider(self):
        assert get_api_key_for_model("mystery:model", {}) is None


class TestConfiguration:
    def test_defaults(self):
        cfg = Configuration.from_runnable_config(None)
        assert cfg.search_api == SearchAPI.TAVILY
        assert cfg.max_concurrent_research_units == 3

    def test_runnable_config_values(self):
        cfg = Configuration.from_runnable_config({"configurable": {"max_react_tool_calls": 2}})
        assert cfg.max_react_tool_calls == 2

    def test_env_overrides_runnable_config(self, monkeypatch):
        monkeypatch.setenv("MAX_REACT_TOOL_CALLS", "9")
        cfg = Configuration.from_runnable_config({"configurable": {"max_react_tool_calls": 2}})
        assert cfg.max_react_tool_calls == 9


class TestTools:
    async def test_tavily_backend_includes_search(self):
        names = {t.name for t in await get_all_tools({})}
        assert names == {"think_tool", "ResearchComplete", "web_search"}

    async def test_no_search_backend(self):
        names = {t.name for t in await get_all_tools({"configurable": {"search_api": "none"}})}
        assert names == {"think_tool", "ResearchComplete"}

    async def test_web_search_without_key_returns_error_string(self):
        result = await web_search.ainvoke({"query": "anything"})
        assert "TAVILY_API_KEY" in result
