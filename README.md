# Deep Research Agents

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

## Layout

```
.
├── src/deep_research_agents/
│   ├── graph.py           # the three nested StateGraphs (this is the core)
│   ├── configuration.py   # Pydantic Configuration, env-var aware
│   ├── state.py           # graph states + structured-output schemas
│   ├── prompts.py         # every prompt template, in one place
│   └── utils.py           # search tool, think_tool, small helpers
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

```bash
python run.py "What are the main approaches to LLM agent memory?"
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
  `get_all_tools()` in `utils.py`.
- **MCP tools**: append additional `BaseTool` instances into the list returned by
  `get_all_tools()`.
- **Different clarification / brief / report behavior**: everything is just prompt text
  in `prompts.py` -- edit freely, no code changes needed.
- **Auth for multi-tenant deployment**: add a `src/security/auth.py` exposing an `auth`
  object (see `langgraph_sdk.Auth`) and point `"auth": {"path": "..."}` at it from
  `langgraph.json`.
