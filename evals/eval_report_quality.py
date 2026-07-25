"""Eval: Final report quality.

Judges whether the final report:
  1. Answers the research brief
  2. Is well-structured (has sections, headings)
  3. Does not fabricate — only claims what findings support

Uses an LLM-as-judge with a structured score + reasoning output.

Run:
    python evals/eval_report_quality.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deep_research_agents import deep_researcher  # noqa: E402

# ---------------------------------------------------------------------------
# Test cases: (topic, what a good report must address)
# ---------------------------------------------------------------------------
CASES = [
    {
        "topic": "What are the main approaches to memory in LLM agents?",
        "must_cover": [
            "in-context memory",
            "external memory or vector store",
            "episodic or long-term memory",
        ],
    },
    {
        "topic": "How do transformer attention mechanisms work?",
        "must_cover": [
            "query key value",
            "softmax",
            "self-attention",
        ],
    },
]

# ---------------------------------------------------------------------------
# Judge schema
# ---------------------------------------------------------------------------
class ReportScore(BaseModel):
    answers_brief: bool = Field(description="Does the report address the research topic?")
    well_structured: bool = Field(description="Does the report have clear sections or headings?")
    no_fabrication: bool = Field(description="Does the report avoid claims unsupported by the findings?")
    covers_required_topics: bool = Field(description="Does the report cover the required topics?")
    score: int = Field(description="Overall quality score from 1 (poor) to 5 (excellent).")
    reasoning: str = Field(description="Brief explanation of the scores.")


judge_model = init_chat_model("openai:gpt-4.1-mini").with_structured_output(ReportScore)


async def judge_report(report: str, brief: str, must_cover: list[str]) -> ReportScore:
    prompt = f"""You are evaluating a research report.

Research Brief:
{brief}

Report:
{report}

Required topics that should be covered (check if they appear): {must_cover}

Score the report on the four criteria and provide an overall score (1-5)."""
    return await judge_model.ainvoke([HumanMessage(content=prompt)])


async def run_case(case: dict, idx: int) -> dict:
    print(f"\n[Case {idx + 1}] Topic: {case['topic']}")
    thread = {"configurable": {"thread_id": f"eval-quality-{idx}"}}

    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=case["topic"])]},
        config=thread,
    )

    report = result.get("final_report", "")
    if not report:
        print("  SKIP — no final report returned (may have asked clarifying question)")
        return {"case": idx, "skipped": True}

    score = await judge_report(report, case["topic"], case["must_cover"])
    passed = score.score >= 3 and score.answers_brief

    print(f"  Score:            {score.score}/5")
    print(f"  Answers brief:    {score.answers_brief}")
    print(f"  Well structured:  {score.well_structured}")
    print(f"  No fabrication:   {score.no_fabrication}")
    print(f"  Covers topics:    {score.covers_required_topics}")
    print(f"  Reasoning:        {score.reasoning}")
    print(f"  Result:           {'PASS' if passed else 'FAIL'}")

    return {"case": idx, "skipped": False, "passed": passed, "score": score.score}


async def main() -> None:
    print("=== Eval: Report Quality ===")
    results = []
    for i, case in enumerate(CASES):
        results.append(await run_case(case, i))

    evaluated = [r for r in results if not r.get("skipped")]
    if evaluated:
        passed = sum(1 for r in evaluated if r["passed"])
        avg_score = sum(r["score"] for r in evaluated) / len(evaluated)
        print(f"\n{'='*40}")
        print(f"Passed: {passed}/{len(evaluated)}")
        print(f"Average score: {avg_score:.1f}/5")


if __name__ == "__main__":
    asyncio.run(main())
