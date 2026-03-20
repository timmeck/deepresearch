"""Tests for research engine (unit tests, no LLM/network needed)."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from src.db.database import Database
from src.research.engine import ResearchEngine, compute_source_score, RESEARCH_TEMPLATES


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
    engine, _db = eng
    result = await engine.follow_up(999, "question?")
    assert "error" in result


@pytest.mark.asyncio
async def test_follow_up_no_llm(eng):
    engine, db = eng
    p = await db.create_project("Test", "query")
    result = await engine.follow_up(p["id"], "question?")
    # Should fail gracefully without LLM
    assert "answer" in result or "error" in result


# ── Source Quality Scoring Tests ──────────────────────────────────

def test_source_score_trusted_domain():
    score = compute_source_score("https://github.com/user/repo", "A " * 3000)
    assert score >= 0.35  # Trusted domain bonus


def test_source_score_edu_domain():
    score = compute_source_score("https://cs.stanford.edu/paper", "A " * 3000)
    assert score >= 0.30  # .edu bonus


def test_source_score_gov_domain():
    score = compute_source_score("https://data.gov/dataset", "A " * 3000)
    assert score >= 0.30  # .gov bonus


def test_source_score_docs_prefix():
    score = compute_source_score("https://docs.python.org/3/", "A " * 3000)
    assert score >= 0.25  # docs.* prefix bonus


def test_source_score_unknown_domain():
    score = compute_source_score("https://randomsite.xyz/page", "Short text")
    assert score < 0.5  # Unknown domain, short text


def test_source_score_long_content_higher():
    short = compute_source_score("https://example.com", "Short")
    long_ = compute_source_score("https://example.com", "Word " * 2000)
    assert long_ > short


def test_source_score_with_citations():
    text_with_refs = "Some text [1] more text [2] even more [3] https://example.com https://other.com " * 10
    text_no_refs = "Plain text without any references or links. " * 10
    score_refs = compute_source_score("https://example.com", text_with_refs)
    score_plain = compute_source_score("https://example.com", text_no_refs)
    assert score_refs > score_plain


def test_source_score_freshness():
    text_recent = "Published in 2025, this study shows important results about AI."
    text_old = "Published in 2010, this study shows important results about AI."
    score_recent = compute_source_score("https://example.com", text_recent + " word" * 500)
    score_old = compute_source_score("https://example.com", text_old + " word" * 500)
    assert score_recent >= score_old


def test_source_score_range():
    """Score should always be between 0 and 1."""
    # Best case: trusted domain + long content + citations + fresh
    best = compute_source_score(
        "https://github.com/repo",
        ("Reference [1] [2] [3] https://a.com https://b.com 2025 " * 500),
    )
    assert 0.0 <= best <= 1.0

    # Worst case
    worst = compute_source_score("https://x.xyz", "hi")
    assert 0.0 <= worst <= 1.0


# ── Research Templates Tests ──────────────────────────────────────

def test_templates_exist():
    assert "technical_comparison" in RESEARCH_TEMPLATES
    assert "market_analysis" in RESEARCH_TEMPLATES
    assert "security_audit" in RESEARCH_TEMPLATES


def test_template_structure():
    for name, tmpl in RESEARCH_TEMPLATES.items():
        assert "system_prompt" in tmpl, f"{name} missing system_prompt"
        assert "report_format" in tmpl, f"{name} missing report_format"
        assert len(tmpl["system_prompt"]) > 20, f"{name} system_prompt too short"
        assert len(tmpl["report_format"]) > 20, f"{name} report_format too short"


def test_template_technical_comparison_content():
    tmpl = RESEARCH_TEMPLATES["technical_comparison"]
    assert "table" in tmpl["report_format"].lower() or "comparison" in tmpl["report_format"].lower()


def test_template_security_audit_content():
    tmpl = RESEARCH_TEMPLATES["security_audit"]
    assert "vulnerabilit" in tmpl["system_prompt"].lower() or "security" in tmpl["system_prompt"].lower()


def test_template_market_analysis_content():
    tmpl = RESEARCH_TEMPLATES["market_analysis"]
    assert "market" in tmpl["system_prompt"].lower()
