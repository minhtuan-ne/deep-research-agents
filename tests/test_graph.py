"""Offline tests of graph control flow: routing, fan-out, failure isolation, and retries."""

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from deep_research_agents import graph as graph_module
from deep_research_agents.graph import builder, final_report_generation, researcher_tools, supervisor_tools
from deep_research_agents.state import ClarifyWithUser, ResearchQuestion


def tool_call(name: str, args: dict | None = None, call_id: str = "1") -> dict:
    return {"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}


def scripted_pipeline(need_clarification: bool = False):
    """A handler that walks the whole graph: brief -> 2 parallel researchers -> report."""

    def handler(schema, tool_names, messages):
        if schema is ClarifyWithUser:
            return ClarifyWithUser(
                need_clarification=need_clarification,
                combined_question="Which aspect do you care about?" if need_clarification else "",
                verification="Starting research.",
            )
        if schema is ResearchQuestion:
            return ResearchQuestion(research_brief="Compare A and B.")

        has_results = any(isinstance(m, ToolMessage) for m in messages)

        if "ConductResearch" in tool_names:  # supervisor
            if has_results:
                return AIMessage(content="", tool_calls=[tool_call("ResearchComplete")])
            return AIMessage(
                content="",
                tool_calls=[
                    tool_call("ConductResearch", {"research_topic": "Topic A"}, "a"),
                    tool_call("ConductResearch", {"research_topic": "Topic B"}, "b"),
                ],
            )

        if "ResearchComplete" in tool_names:  # researcher
            if has_results:
                return AIMessage(content="", tool_calls=[tool_call("ResearchComplete", call_id="done")])
            return AIMessage(
                content="", tool_calls=[tool_call("think_tool", {"reflection": "plan"}, "t")]
            )

        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        if "Distill" in system:  # compressor: echo the topic it was given
            return AIMessage(content=f"Summary of {messages[1].content}")
        # final report writer: include the prompt so the test can inspect the findings
        return AIMessage(content="FINAL REPORT\n" + messages[0].content)

    return handler


async def test_end_to_end_offline(fake_model):
    model = fake_model(scripted_pipeline())
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Compare A and B"}]})

    report = result["final_report"]
    assert report.startswith("FINAL REPORT")
    assert "Summary of Topic A" in report and "Summary of Topic B" in report
    assert result["research_brief"] == "Compare A and B."
    researcher_calls = [c for c in model.calls if "ResearchComplete" in c["tools"] and "ConductResearch" not in c["tools"]]
    assert len(researcher_calls) == 4  # 2 researchers x (think step + complete step)


async def test_clarification_stops_before_research(fake_model):
    model = fake_model(scripted_pipeline(need_clarification=True))
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Tell me about agents"}]})

    assert not result.get("final_report")
    assert result["messages"][-1].content == "Which aspect do you care about?"
    assert len(model.calls) == 1


async def test_clarification_can_be_disabled(fake_model):
    fake_model(scripted_pipeline(need_clarification=True))
    graph = builder.compile()

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Tell me about agents"}]},
        {"configurable": {"allow_clarification": False}},
    )

    assert result["final_report"].startswith("FINAL REPORT")


async def test_supervisor_caps_concurrency_and_isolates_failures(monkeypatch):
    started: list[str] = []

    class FakeResearcher:
        async def ainvoke(self, state, _config):
            topic = state["research_topic"]
            started.append(topic)
            if topic == "boom":
                raise RuntimeError("search backend down")
            return {"compressed_research": f"notes on {topic}"}

    monkeypatch.setattr(graph_module, "researcher_subgraph", FakeResearcher())
    calls = [
        tool_call("ConductResearch", {"research_topic": t}, t)
        for t in ["x", "boom", "y", "overflow"]
    ]
    state = {"supervisor_messages": [AIMessage(content="", tool_calls=calls)], "research_iterations": 1}

    cmd = await supervisor_tools(state, {})

    assert cmd.goto == "supervisor"
    by_id = {m.tool_call_id: m.content for m in cmd.update["supervisor_messages"]}
    assert started == ["x", "boom", "y"]  # default max_concurrent_research_units=3
    assert by_id["x"] == "notes on x" and by_id["y"] == "notes on y"
    assert "search backend down" in by_id["boom"]
    assert "exceeded max_concurrent_research_units" in by_id["overflow"]


async def test_supervisor_exits_after_iteration_budget():
    state = {
        "supervisor_messages": [
            ToolMessage(content="earlier finding", tool_call_id="0"),
            AIMessage(content="", tool_calls=[tool_call("think_tool", {"reflection": "more?"})]),
        ],
        "research_iterations": 99,
    }
    cmd = await supervisor_tools(state, {})
    assert cmd.goto == "__end__"
    assert cmd.update["notes"] == ["earlier finding"]


async def test_researcher_research_complete_routes_to_compression():
    state = {
        "researcher_messages": [AIMessage(content="", tool_calls=[tool_call("ResearchComplete")])],
        "tool_call_iterations": 1,
    }
    cmd = await researcher_tools(state, {"configurable": {"search_api": "none"}})

    assert cmd.goto == "compress_research"
    assert cmd.update["researcher_messages"][0].content == "Research marked complete."


async def test_researcher_unknown_tool_is_reported_not_raised():
    state = {
        "researcher_messages": [AIMessage(content="", tool_calls=[tool_call("does_not_exist")])],
        "tool_call_iterations": 1,
    }
    cmd = await researcher_tools(state, {"configurable": {"search_api": "none"}})

    assert cmd.goto == "researcher"
    assert cmd.update["researcher_messages"][0].content.startswith("Error executing tool")


async def test_final_report_truncates_findings_on_context_overflow(fake_model):
    prompt_lengths: list[int] = []

    def handler(_schema, _tools, messages):
        prompt_lengths.append(len(messages[0].content))
        if len(prompt_lengths) == 1:
            raise ValueError("This model's maximum context length is 8192 tokens")
        return AIMessage(content="short report")

    fake_model(handler)
    state = {"research_brief": "brief", "notes": ["x" * 10_000]}

    result = await final_report_generation(state, {})

    assert result["final_report"] == "short report"
    assert prompt_lengths[1] < prompt_lengths[0]


async def test_final_report_non_token_error_is_not_retried(fake_model):
    attempts = []

    def handler(*_):
        attempts.append(1)
        raise ValueError("invalid api key")

    fake_model(handler)
    result = await final_report_generation({"research_brief": "b", "notes": ["n"]}, {})

    assert len(attempts) == 1
    assert "invalid api key" in result["final_report"]
