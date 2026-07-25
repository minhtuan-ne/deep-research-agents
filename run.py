"""Run the deep research agent from the command line.

Single-turn (default):
    python run.py "Tell me about SAP"
    python run.py "my follow-up answer"   # continues the same thread via SQLite

Chat loop (interactive):
    python run.py --chat

    Type your message and press Enter. The agent will ask clarifying questions
    if needed, then run research and print the final report — all in one session.
    Type 'exit' or 'quit' to stop. Type 'new' to start a fresh conversation.
"""

import asyncio
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

from deep_research_agents.graph import builder  # noqa: E402

# Persistent thread for single-turn mode (survives across process restarts).
CLI_THREAD = {"configurable": {"thread_id": "cli-session"}}


async def run_single_turn(graph, question: str, thread: dict) -> None:
    """Send one message and print the response."""
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=thread,
    )
    if result.get("final_report"):
        print("\n" + result["final_report"])
    else:
        print("\nAgent: " + result["messages"][-1].content)


async def chat_loop(graph) -> None:
    """Interactive multi-turn chat loop."""
    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id}}

    print("Deep Research Agent — chat mode")
    print("Commands: 'new' = fresh conversation, 'exit' / 'quit' = stop\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            thread = {"configurable": {"thread_id": thread_id}}
            print("--- new conversation started ---\n")
            continue

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread,
        )

        if result.get("final_report"):
            print("\nAgent:\n" + result["final_report"] + "\n")
            # Research is done — next message starts a new topic on a fresh thread.
            thread_id = str(uuid.uuid4())
            thread = {"configurable": {"thread_id": thread_id}}
            print("--- ready for a new question ---\n")
        else:
            print("\nAgent: " + result["messages"][-1].content + "\n")


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        if "--chat" in sys.argv:
            await chat_loop(graph)
        else:
            question = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
            if not question:
                print("Usage: python run.py \"your question\"  or  python run.py --chat")
                return
            await run_single_turn(graph, question, CLI_THREAD)


if __name__ == "__main__":
    asyncio.run(main())
