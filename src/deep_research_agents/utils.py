"""Search tools, the think_tool, and small helper functions."""

import os
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, tool

from deep_research_agents.configuration import Configuration, SearchAPI


def get_today_str() -> str:
    """Return today's date as a readable string, for grounding prompts."""
    return datetime.now().strftime("%A, %B %d, %Y")


@tool(description="Record a brief strategic reflection. Does not perform any external action.")
def think_tool(reflection: str) -> str:
    """A no-op tool used purely to force an explicit 'pause and plan' step."""
    return f"Reflection recorded: {reflection}"


@tool(description="Search the web for information on a query.")
async def web_search(
    query: str, config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> str:
    """Run a Tavily search and return a compact, readable summary of results.

    Uses the async client so parallel researchers don't block each other's
    event loop while waiting on network I/O.
    """
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        return "Error: tavily-python is not installed. `pip install tavily-python`."

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set."

    client = AsyncTavilyClient(api_key=api_key)
    try:
        response = await client.search(query, max_results=5, include_raw_content=False)
    except Exception as e:  # noqa: BLE001
        return f"Error: search failed: {e}"

    lines = []
    for result in response.get("results", []):
        lines.append(f"- {result['title']} ({result['url']}): {result['content'][:500]}")
    return "\n".join(lines) if lines else "No results found."


@tool(
    "ResearchComplete",
    description="Call this once you have gathered enough information on your assigned topic.",
)
def research_complete_tool() -> str:
    """Signals the researcher loop to stop and move on to compression."""
    return "Research marked complete."


async def get_all_tools(config: RunnableConfig) -> list[BaseTool]:
    """Assemble the tool list available to a researcher: search + think_tool + ResearchComplete."""
    configurable = Configuration.from_runnable_config(config)
    tools: list[BaseTool] = [think_tool, research_complete_tool]
    if configurable.search_api == SearchAPI.TAVILY:
        tools.append(web_search)
    return tools


def get_notes_from_tool_calls(messages: list) -> list[str]:
    """Pull the content of every ToolMessage (i.e. research results) out of a transcript."""
    return [str(m.content) for m in filter_messages(messages, include_types=["tool"])]


def get_api_key_for_model(model_name: str, config: RunnableConfig) -> str | None:
    """Resolve the right API key env var for a given 'provider:model' string."""
    provider = model_name.split(":")[0] if ":" in model_name else "openai"
    env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
    }.get(provider)
    return os.environ.get(env_var) if env_var else None


def is_token_limit_exceeded(exception: Exception) -> bool:
    """Best-effort, provider-agnostic detection of a context-length error."""
    message = str(exception).lower()
    return any(
        phrase in message
        for phrase in ("context length", "context_length", "maximum context", "too many tokens")
    )


def remove_up_to_last_ai_message(messages: list) -> list:
    """Trim a message list back to (and excluding) the most recent AIMessage.

    Used to shrink an over-budget transcript before retrying a call.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages
