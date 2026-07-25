"""Eval: Clarification behavior.

Tests that the agent asks a clarifying question when the topic is ambiguous,
and skips clarification when the topic is specific enough.

Does NOT require a full research run — checks after the first graph turn only,
so it's fast and cheap.

Run:
    python evals/eval_clarification.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_research_agents import deep_researcher  # noqa: E402

# ---------------------------------------------------------------------------
# Test cases: (topic, should_clarify)
# Ambiguous topics should trigger a question; specific ones should not.
# ---------------------------------------------------------------------------
CASES = [
    # Should ask — too vague
    {"topic": "Tell me about agents", "should_clarify": True},
    {"topic": "Research AI", "should_clarify": True},
    # Should NOT ask — specific enough to proceed
    {
        "topic": "What are the main approaches to memory management in LLM agents, including in-context, external vector stores, and episodic memory?",
        "should_clarify": False,
    },
    {
        "topic": "Explain how the transformer self-attention mechanism works, focusing on the query-key-value computation and softmax scaling.",
        "should_clarify": False,
    },
]


async def run_case(case: dict, idx: int) -> dict:
    topic = case["topic"]
    should_clarify = case["should_clarify"]
    print(f"\n[Case {idx + 1}] {'Ambiguous' if should_clarify else 'Specific'}: {topic[:80]}...")

    thread = {"configurable": {"thread_id": f"eval-clarify-{idx}"}}

    # Use stream to capture just the first turn without running full research
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=topic)]},
        config={**thread, "configurable": {**thread["configurable"], "allow_clarification": True}},
    )

    last_msg = result["messages"][-1]
    asked_question = isinstance(last_msg, AIMessage) and not result.get("final_report")
    went_to_research = bool(result.get("final_report") or result.get("research_brief"))

    if should_clarify:
        passed = asked_question
        status = "asked clarifying question" if asked_question else "skipped clarification (unexpected)"
    else:
        passed = not asked_question
        status = "proceeded to research" if went_to_research else "asked clarifying question (unexpected)"

    print(f"  Expected clarify: {should_clarify}")
    print(f"  Asked question:   {asked_question}")
    print(f"  Status:           {status}")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return {"case": idx, "passed": passed, "asked": asked_question, "expected": should_clarify}


async def main() -> None:
    print("=== Eval: Clarification Behavior ===")
    print("Note: specific-topic cases will run a full research loop.\n")

    results = []
    for i, case in enumerate(CASES):
        results.append(await run_case(case, i))

    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*40}")
    print(f"Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
