"""The deep research graph: clarify -> brief -> supervisor(fan-out researchers) -> report."""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_research_agents.configuration import Configuration
from deep_research_agents.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from deep_research_agents.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from deep_research_agents.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

# One configurable model reused everywhere; each node specializes it via .with_config().
configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))


###################
# Top-level nodes
###################
async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Ask a clarifying question if needed, otherwise proceed straight to research."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    model = (
        configurable_model.with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(
            {
                "model": configurable.research_model,
                "max_tokens": configurable.research_model_max_tokens,
                "api_key": get_api_key_for_model(configurable.research_model, config),
                "tags": ["langsmith:nostream"],
            }
        )
    )
    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(state["messages"]), date=get_today_str()
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """Turn the conversation into a structured brief and seed the supervisor's context."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        configurable_model.with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(
            {
                "model": configurable.research_model,
                "max_tokens": configurable.research_model_max_tokens,
                "api_key": get_api_key_for_model(configurable.research_model, config),
                "tags": ["langsmith:nostream"],
            }
        )
    )
    prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = await model.ainvoke([HumanMessage(content=prompt)])

    supervisor_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_prompt),
                    HumanMessage(content=response.research_brief),
                ],
            },
        },
    )


###################
# Supervisor subgraph
###################
async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """Decide whether to think, delegate research, or declare completion."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        configurable_model.bind_tools([ConductResearch, ResearchComplete, think_tool])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(
            {
                "model": configurable.research_model,
                "max_tokens": configurable.research_model_max_tokens,
                "api_key": get_api_key_for_model(configurable.research_model, config),
                "tags": ["langsmith:nostream"],
            }
        )
    )
    response = await model.ainvoke(state["supervisor_messages"])
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Execute think_tool calls, fan out ConductResearch calls in parallel, or exit."""
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state["supervisor_messages"]
    most_recent = supervisor_messages[-1]

    exceeded_iterations = state.get("research_iterations", 0) > configurable.max_researcher_iterations
    no_tool_calls = not most_recent.tool_calls
    research_complete = any(tc["name"] == "ResearchComplete" for tc in most_recent.tool_calls)

    if exceeded_iterations or no_tool_calls or research_complete:
        return Command(
            goto=END,
            update={"notes": get_notes_from_tool_calls(supervisor_messages)},
        )

    tool_messages = []

    for tc in [c for c in most_recent.tool_calls if c["name"] == "think_tool"]:
        tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {tc['args']['reflection']}",
                name="think_tool",
                tool_call_id=tc["id"],
            )
        )

    conduct_calls = [c for c in most_recent.tool_calls if c["name"] == "ConductResearch"]
    if conduct_calls:
        allowed = conduct_calls[: configurable.max_concurrent_research_units]
        overflow = conduct_calls[configurable.max_concurrent_research_units :]

        tasks = [
            researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                    "research_topic": tc["args"]["research_topic"],
                },
                config,
            )
            for tc in allowed
        ]
        results = await asyncio.gather(*tasks)

        for result, tc in zip(results, allowed):
            tool_messages.append(
                ToolMessage(
                    content=result.get("compressed_research", "Error: research failed."),
                    name="ConductResearch",
                    tool_call_id=tc["id"],
                )
            )
        for tc in overflow:
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"Error: exceeded max_concurrent_research_units="
                        f"{configurable.max_concurrent_research_units}. Try again with fewer."
                    ),
                    name="ConductResearch",
                    tool_call_id=tc["id"],
                )
            )

    return Command(goto="supervisor", update={"supervisor_messages": tool_messages})


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


###################
# Researcher subgraph
###################
async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """Take one tool-calling step toward researching the assigned topic."""
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)
    if not tools:
        raise ValueError("No tools configured: set search_api or add tools to get_all_tools().")

    model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(
            {
                "model": configurable.research_model,
                "max_tokens": configurable.research_model_max_tokens,
                "api_key": get_api_key_for_model(configurable.research_model, config),
                "tags": ["langsmith:nostream"],
            }
        )
    )
    system_prompt = research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_prompt)] + state["researcher_messages"]
    response = await model.ainvoke(messages)

    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute the researcher's tool calls, or exit to compression."""
    configurable = Configuration.from_runnable_config(config)
    most_recent = state["researcher_messages"][-1]

    if not most_recent.tool_calls:
        return Command(goto="compress_research")

    tools = await get_all_tools(config)
    tools_by_name = {t.name: t for t in tools}

    async def run_one(tc):
        try:
            return await tools_by_name[tc["name"]].ainvoke(tc["args"], config)
        except Exception as e:  # noqa: BLE001
            return f"Error executing tool: {e}"

    observations = await asyncio.gather(*[run_one(tc) for tc in most_recent.tool_calls])
    tool_messages = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, most_recent.tool_calls)
    ]

    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    done = any(tc["name"] == "ResearchComplete" for tc in most_recent.tool_calls)

    goto = "compress_research" if (exceeded or done) else "researcher"
    return Command(goto=goto, update={"researcher_messages": tool_messages})


async def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Distill the researcher's transcript into a dense summary, with retry on overflow."""
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config(
        {
            "model": configurable.compression_model,
            "max_tokens": configurable.compression_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.compression_model, config),
            "tags": ["langsmith:nostream"],
        }
    )
    messages = list(state["researcher_messages"]) + [
        HumanMessage(content=compress_research_simple_human_message)
    ]

    for _ in range(3):
        try:
            system_prompt = compress_research_system_prompt.format(date=get_today_str())
            response = await model.ainvoke([SystemMessage(content=system_prompt)] + messages)
            return {"compressed_research": str(response.content)}
        except Exception as e:  # noqa: BLE001
            if is_token_limit_exceeded(e):
                messages = remove_up_to_last_ai_message(messages)
                continue
            break

    return {"compressed_research": "Error compressing research: max retries exceeded."}


researcher_builder = StateGraph(
    ResearcherState, output=ResearcherOutputState, config_schema=Configuration
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


###################
# Final report
###################
async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    """Write the final report from all accumulated notes."""
    configurable = Configuration.from_runnable_config(config)
    findings = "\n".join(state.get("notes", []))
    model = configurable_model.with_config(
        {
            "model": configurable.final_report_model,
            "max_tokens": configurable.final_report_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.final_report_model, config),
            "tags": ["langsmith:nostream"],
        }
    )
    prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
    )
    try:
        report = await model.ainvoke([HumanMessage(content=prompt)])
        return {
            "final_report": report.content,
            "messages": [report],
            "notes": {"type": "override", "value": []},
        }
    except Exception as e:  # noqa: BLE001
        return {
            "final_report": f"Error generating final report: {e}",
            "notes": {"type": "override", "value": []},
        }


###################
# Top-level graph
###################
builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("write_research_brief", write_research_brief)
builder.add_node("research_supervisor", supervisor_subgraph)
builder.add_node("final_report_generation", final_report_generation)
builder.add_edge(START, "clarify_with_user")
builder.add_edge("research_supervisor", "final_report_generation")
builder.add_edge("final_report_generation", END)

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

# Swap MemorySaver for SqliteSaver / AsyncPostgresSaver for cross-process persistence.
deep_researcher = builder.compile(checkpointer=MemorySaver())
