"""Eval: Research brief quality.

Checks that the research brief generated from a conversation is:
  1. Faithful — captures the user's intent without distortion
  2. Specific — actionable for an autonomous research agent
  3. Scoped — doesn't over-expand or under-specify

Uses LLM-as-judge. Only runs the graph up to the brief-writing step,
so this is faster than a full research eval.

Run:
    python evals/eval_research_brief.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import the builder directly so we can compile a stripped graph
from deep_research_agents.graph import builder  # noqa: E402
from deep_research_agents.state import AgentInputState  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

# Compile a graph that stops after write_research_brief
brief_only_graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["write_research_brief"],
)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
CASES = [
    {
        "messages": [
            HumanMessage(content="I want a report on memory approaches in LLM agents — specifically in-context, external vector stores, and episodic memory. Focus on recent techniques from 2023 onwards."),
        ],
        "expected_elements": ["memory", "in-context", "vector", "episodic", "2023"],
    },
    {
        "messages": [
            HumanMessage(content="Research the impact of climate change on coral reefs, focusing on bleaching events and ocean acidification."),
        ],
        "expected_elements": ["coral", "bleaching", "ocean acidification", "climate"],
    },
]

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
class BriefScore(BaseModel):
    faithful: bool = Field(description="Does the brief accurately reflect the user's request?")
    specific: bool = Field(description="Is the brief specific enough to guide autonomous research?")
    scoped: bool = Field(description="Does the brief avoid over- or under-expanding the original request?")
    contains_expected: bool = Field(description="Does the brief contain the expected key elements?")
    score: int = Field(description="Overall quality 1-5.")
    reasoning: str


judge_model = init_chat_model("openai:gpt-4.1-mini").with_structured_output(BriefScore)


async def judge_brief(original_messages: list, brief: str, expected: list[str]) -> BriefScore:
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in original_messages
    )
    prompt = f"""Evaluate this research brief generated from a user conversation.

Original conversation:
{conversation}

Generated research brief:
{brief}

Expected key elements that should appear: {expected}

Score the brief on faithfulness, specificity, and scope."""
    return await judge_model.ainvoke([HumanMessage(content=prompt)])


async def run_case(case: dict, idx: int) -> dict:
    print(f"\n[Case {idx + 1}] {case['messages'][0].content[:80]}...")
    thread = {"configurable": {"thread_id": f"eval-brief-{idx}", "allow_clarification": False}}

    await brief_only_graph.ainvoke({"messages": case["messages"]}, config=thread)

    state = brief_only_graph.get_state(thread)
    brief = state.values.get("research_brief", "")

    if not brief:
        print("  SKIP — no brief generated")
        return {"case": idx, "skipped": True}

    print(f"  Brief (truncated): {brief[:200]}...")
    score = await judge_brief(case["messages"], brief, case["expected_elements"])

    passed = score.score >= 3 and score.faithful and score.contains_expected
    print(f"  Score:            {score.score}/5")
    print(f"  Faithful:         {score.faithful}")
    print(f"  Specific:         {score.specific}")
    print(f"  Scoped:           {score.scoped}")
    print(f"  Contains expected:{score.contains_expected}")
    print(f"  Reasoning:        {score.reasoning}")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return {"case": idx, "skipped": False, "passed": passed, "score": score.score}


async def main() -> None:
    print("=== Eval: Research Brief Quality ===")
    results = []
    for i, case in enumerate(CASES):
        results.append(await run_case(case, i))

    evaluated = [r for r in results if not r.get("skipped")]
    if evaluated:
        passed = sum(1 for r in evaluated if r["passed"])
        avg = sum(r["score"] for r in evaluated) / len(evaluated)
        print(f"\n{'='*40}")
        print(f"Passed: {passed}/{len(evaluated)}")
        print(f"Average score: {avg:.1f}/5")


if __name__ == "__main__":
    asyncio.run(main())
