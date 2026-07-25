"""Run the agent from the command line, e.g.: python run.py "your research question".

Uses a fixed thread ID so clarifying questions work across multiple invocations:
the graph pauses, asks a question, and resumes when you run the script again with your answer.
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from deep_research_agents import deep_researcher  # noqa: E402

# Fixed thread ID ties all CLI invocations into one conversation thread.
THREAD_CONFIG = {"configurable": {"thread_id": "cli-session"}}


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What are the main approaches to LLM agent memory?"
    result = await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=THREAD_CONFIG,
    )

    if result.get("final_report"):
        print(result["final_report"])
    else:
        # Graph paused to ask a clarifying question — run again with your answer.
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
