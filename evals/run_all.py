"""Run all evals and print a summary.

Run:
    python evals/run_all.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root and src are importable regardless of cwd
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evals.eval_clarification as eval_clarification
import evals.eval_research_brief as eval_research_brief
import evals.eval_report_quality as eval_report_quality

EVALS = [
    ("eval_clarification",  eval_clarification),
    ("eval_research_brief", eval_research_brief),
    ("eval_report_quality", eval_report_quality),
]


async def main() -> None:
    print("=" * 50)
    print("Running all evals")
    print("=" * 50)

    for name, mod in EVALS:
        print(f"\n>>> {name}")
        await mod.main()

    print("\n" + "=" * 50)
    print("All evals complete.")


if __name__ == "__main__":
    asyncio.run(main())
