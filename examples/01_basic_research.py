#!/usr/bin/env python3
"""Example: Run a basic research project."""

import asyncio
import sys

sys.path.insert(0, ".")

from src.ai.llm import LLM
from src.db.database import Database
from src.research.engine import ResearchEngine


async def main():
    db = Database()
    await db.initialize()
    llm = LLM()
    engine = ResearchEngine(db, llm)

    print(f"\n  LLM: {llm.provider}/{llm.model}\n")

    # Run research
    print("  Researching: 'What is FastAPI and how does it compare to Django?'\n")
    result = await engine.research("What is FastAPI and how does it compare to Django?")

    if result["status"] == "completed":
        print(f"  Sources: {result['sources']}")
        print(f"  Findings: {result['findings']}")
        print(f"\n  Report:\n{result['report'][:500]}...")
    else:
        print(f"  Error: {result.get('error')}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
