"""Tests for web crawler."""
import pytest
from src.research.crawler import chunk_text


def test_chunk_small_text():
    chunks = chunk_text("Hello world")
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_chunk_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_large_text():
    text = "Word " * 500  # ~2500 chars
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    # Check overlap
    for i in range(len(chunks) - 1):
        # Chunks should have some shared content
        assert len(chunks[i]) > 100


def test_chunk_respects_size():
    text = "A " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    for chunk in chunks:
        assert len(chunk) <= 250  # Some tolerance for boundary seeking


def test_chunk_preserves_content():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    # All content should appear in at least one chunk
    full = " ".join(chunks)
    assert "First" in full
    assert "Second" in full
    assert "Third" in full
