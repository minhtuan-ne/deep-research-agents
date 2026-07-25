"""Run the agent from the command line, e.g.: python run.py "your research question"."""

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from deep_research_agents import deep_researcher  # noqa: E402


async def main() -> None:
    """Invoke the graph with a question from argv and print the final report."""
    question = " ".join(sys.argv[1:]) or "What are the main approaches to LLM agent memory?"
    result = await deep_researcher.ainvoke({"messages": [HumanMessage(content=question)]})

    if result.get("final_report"):
        print(result["final_report"])
    else:
        # The graph ended early, e.g. to ask a clarifying question.
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
