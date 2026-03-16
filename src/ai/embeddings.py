"""Ollama Embeddings -- Semantic search via qwen2.5 (3584 dims).

Shared module: embed text, store in SQLite as BLOB, cosine similarity search.
Hybrid search: 0.6 * semantic + 0.4 * FTS5 rank (when available).
"""

import struct
import math
import httpx
from src.config import OLLAMA_URL
from src.utils.logger import get_logger

log = get_logger("embeddings")

EMBEDDING_MODEL = "qwen2.5"
EMBEDDING_DIMS = 3584


async def ensure_table(conn):
    """Create the embeddings table if it doesn't exist."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    await conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_emb_source
        ON embeddings(source_table, source_id)
    """)
    await conn.commit()


async def embed_text(text: str) -> list[float] | None:
    """Get embedding vector from Ollama. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": text[:8000]},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings[0]) > 0:
                return embeddings[0]
    except Exception as e:
        log.debug(f"Embedding failed: {e}")
    return None


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed multiple texts. Returns list of vectors (None for failures)."""
    results = []
    for text in texts:
        results.append(await embed_text(text))
    return results


def to_blob(vector: list[float]) -> bytes:
    """Pack float list into binary blob for SQLite storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def from_blob(blob: bytes) -> list[float]:
    """Unpack binary blob back to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def store_embedding(conn, source_table: str, source_id: int, vector: list[float]):
    """Store an embedding vector in the database."""
    blob = to_blob(vector)
    await conn.execute("""
        INSERT OR REPLACE INTO embeddings (source_table, source_id, embedding)
        VALUES (?, ?, ?)
    """, (source_table, source_id, blob))
    await conn.commit()


async def search_similar(conn, query_vector: list[float], source_table: str,
                         limit: int = 10, source_ids: set = None) -> list[dict]:
    """Find most similar entries by cosine similarity.

    Returns list of {source_id, similarity} sorted by similarity DESC.
    """
    sql = "SELECT source_id, embedding FROM embeddings WHERE source_table = ?"
    params = [source_table]
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()

    scored = []
    for row in rows:
        sid = row[0] if isinstance(row, (tuple, list)) else row["source_id"]
        blob = row[1] if isinstance(row, (tuple, list)) else row["embedding"]
        if source_ids and sid not in source_ids:
            continue
        vec = from_blob(blob)
        sim = cosine_similarity(query_vector, vec)
        scored.append({"source_id": sid, "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]
