"""Tests for DeepResearch database."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from src.db.database import Database


@pytest_asyncio.fixture
async def db():
    with tempfile.TemporaryDirectory() as t:
        db = Database(db_path=Path(t) / "test.db")
        await db.initialize()
        yield db
        await db.close()


@pytest.mark.asyncio
async def test_create_project(db):
    p = await db.create_project("Test Research", "What is FastAPI?")
    assert p["title"] == "Test Research"
    assert p["status"] == "pending"


@pytest.mark.asyncio
async def test_list_projects(db):
    await db.create_project("P1", "q1")
    await db.create_project("P2", "q2")
    projects = await db.list_projects()
    assert len(projects) == 2


@pytest.mark.asyncio
async def test_update_project(db):
    p = await db.create_project("Update", "q")
    await db.update_project(p["id"], status="completed", report="Done")
    updated = await db.get_project(p["id"])
    assert updated["status"] == "completed"
    assert updated["report"] == "Done"


@pytest.mark.asyncio
async def test_delete_project(db):
    p = await db.create_project("Delete", "q")
    ok = await db.delete_project(p["id"])
    assert ok is True
    assert await db.get_project(p["id"]) is None


@pytest.mark.asyncio
async def test_add_source(db):
    p = await db.create_project("Sources", "q")
    s = await db.add_source(p["id"], "https://example.com", title="Example", content="Hello")
    assert s["url"] == "https://example.com"
    sources = await db.get_sources(p["id"])
    assert len(sources) == 1


@pytest.mark.asyncio
async def test_source_increments_count(db):
    p = await db.create_project("Count", "q")
    await db.add_source(p["id"], "https://a.com")
    await db.add_source(p["id"], "https://b.com")
    updated = await db.get_project(p["id"])
    assert updated["sources_count"] == 2


@pytest.mark.asyncio
async def test_chunks_and_search(db):
    p = await db.create_project("Chunks", "q")
    s = await db.add_source(p["id"], "https://example.com")
    await db.add_chunk(s["id"], 0, "Python is a programming language")
    await db.add_chunk(s["id"], 1, "JavaScript runs in browsers")
    results = await db.search_chunks("Python")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_add_finding(db):
    p = await db.create_project("Findings", "q")
    f = await db.add_finding(p["id"], "FastAPI is fast", category="fact", confidence=0.9)
    assert f["content"] == "FastAPI is fast"
    findings = await db.get_findings(p["id"])
    assert len(findings) == 1


@pytest.mark.asyncio
async def test_follow_up(db):
    p = await db.create_project("Follow", "q")
    fu = await db.add_follow_up(p["id"], "What about X?", answer="X is great")
    assert fu["question"] == "What about X?"
    follow_ups = await db.get_follow_ups(p["id"])
    assert len(follow_ups) == 1


@pytest.mark.asyncio
async def test_activity_log(db):
    await db.log_event("test", "Test event", data={"key": "val"})
    events = await db.get_activity(limit=1)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_stats(db):
    p = await db.create_project("Stats", "q")
    s = await db.add_source(p["id"], "https://x.com")
    await db.add_chunk(s["id"], 0, "content")
    await db.add_finding(p["id"], "finding")
    stats = await db.get_stats()
    assert stats["projects"] == 1
    assert stats["sources"] == 1
    assert stats["findings"] == 1
    assert stats["chunks"] == 1
