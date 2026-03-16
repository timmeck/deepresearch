"""Tests for research engine (unit tests, no LLM/network needed)."""
import pytest, pytest_asyncio, tempfile
from pathlib import Path
from src.db.database import Database
from src.research.engine import ResearchEngine


@pytest_asyncio.fixture
async def eng():
    with tempfile.TemporaryDirectory() as t:
        db = Database(db_path=Path(t) / "test.db")
        await db.initialize()
        engine = ResearchEngine(db, None)  # No LLM
        yield engine, db
        await db.close()


@pytest.mark.asyncio
async def test_follow_up_no_project(eng):
    engine, db = eng
    result = await engine.follow_up(999, "question?")
    assert "error" in result


@pytest.mark.asyncio
async def test_follow_up_no_llm(eng):
    engine, db = eng
    p = await db.create_project("Test", "query")
    result = await engine.follow_up(p["id"], "question?")
    # Should fail gracefully without LLM
    assert "answer" in result or "error" in result
