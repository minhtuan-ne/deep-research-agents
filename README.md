# Deep Research Agents

[![tests](https://github.com/minhtuan-ne/deep-research-agents/actions/workflows/tests.yml/badge.svg)](https://github.com/minhtuan-ne/deep-research-agents/actions/workflows/tests.yml)

A supervisor/researcher "deep research" agent built on
LangGraph. Same three-level nested-graph architecture as the reference project it's
derived from, trimmed down to one search backend and no auth/MCP layer, so you can read
the whole thing in one sitting and extend it from there.

```
clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation
                                                    |
                                        (subgraph) supervisor <-> supervisor_tools
                                                    |  fans out via asyncio.gather
                                                    v
                                    (subgraph, x N parallel) researcher -> researcher_tools -> compress_research
```

## Design highlights

- **Parallel fan-out**: the supervisor delegates up to `max_concurrent_research_units`
  topics at once; each runs in its own researcher subgraph via `asyncio.gather`, and
  search uses Tavily's async client so researchers never block each other.
- **Failure isolation**: a researcher that raises returns an error message for its topic
  instead of aborting its siblings or the whole run.
- **Context-window resilience**: compression and final-report generation detect
  context-length errors and retry with a trimmed transcript / findings.
- **Cited reports**: researchers keep source URLs through compression, and the final
  report uses numbered inline citations with a `Sources` section.
- **Bounded loops**: supervisor iterations and researcher tool calls are both capped
  by configuration, so a run always terminates.
- **Persistent threads**: conversation state is checkpointed to SQLite, so clarifying
  questions can be answered across separate CLI invocations.

## Layout

```
.
├── src/deep_research_agents/
│   ├── graph.py           # the three nested StateGraphs (this is the core)
│   ├── configuration.py   # Pydantic Configuration, env-var aware
│   ├── state.py           # graph states + structured-output schemas
│   ├── prompts.py         # every prompt template, in one place
│   └── utils.py           # search tool, think_tool, small helpers
├── tests/                 # offline unit + graph tests (fake model, no API keys)
├── evals/                 # LLM-as-judge evals (real models, needs API keys)
├── .github/workflows/     # CI: runs the test suite on Python 3.10-3.12
├── run.py                 # CLI entry point for local testing
├── langgraph.json         # LangGraph Platform / Studio deployment config
├── pyproject.toml
└── .env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                # then fill in your API keys
```

You need at minimum `OPENAI_API_KEY` (or swap `research_model` / `final_report_model` in
`configuration.py` to an `anthropic:...` model and set `ANTHROPIC_API_KEY` instead) and
`TAVILY_API_KEY` for web search.

## Run from the command line

Interactive chat mode — handles clarifying questions and multi-turn conversation in one session:

```bash
python run.py --chat
```

Single-turn mode — state is persisted to `checkpoints.db` so you can reply across separate invocations:

```bash
python run.py "What are the main approaches to LLM agent memory?"
python run.py "your follow-up answer"
```

## Tests

Fast, offline tests. A scripted fake chat model drives the real graph end to end
(clarify -> brief -> parallel researchers -> report), so no API keys or network are needed.
They cover routing, the concurrency cap, failure isolation, and context-overflow retries.

```bash
pip install -e ".[dev]"
pytest
```

## Evals

LLM-as-judge evals covering clarification behavior, research brief quality, and final report quality.

```bash
# Run all evals
python evals/run_all.py

# Or run individually
python evals/eval_clarification.py   # checks ask-vs-skip clarification logic
python evals/eval_research_brief.py  # checks brief faithfulness and specificity
python evals/eval_report_quality.py  # checks final report quality (full research run)
```

## Run with LangGraph Studio (visual debugger)

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This opens a local API server plus a Studio UI where you can step through each node,
inspect state at every point, and tweak `Configuration` fields live.

## Where to extend

- **New search backend**: add a case to `SearchAPI` in `configuration.py` and a branch in
  `get_all_tools()` in `utils.py` (plus its API key in `.env`). Prefer an async client so
  parallel researchers don't block each other.
- **MCP tools**: append additional `BaseTool` instances into the list returned by
  `get_all_tools()`. It is called on every researcher step, so create the MCP client once
  and cache it rather than reconnecting inside the function.
- **Different clarification / brief / report behavior**: prompts live in `prompts.py`.
  Keep the existing `{placeholders}` (they are filled via `.format()`; escape literal
  braces as `{{ }}`) and don't rename the tool or field names they mention. To skip
  clarification entirely, set `allow_clarification=False`.
- **Auth for multi-tenant deployment** (LangGraph server only, not `run.py`): add
  `src/security/auth.py` exposing `auth = langgraph_sdk.Auth()` and set
  `"auth": {"path": "./src/security/auth.py:auth"}` in `langgraph.json`.
