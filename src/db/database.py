"""Database for DeepResearch."""
import aiosqlite, json
from datetime import datetime, timezone
from pathlib import Path
from src.config import DB_PATH
from src.utils.logger import get_logger

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    query TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    report TEXT,
    sources_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    summary TEXT,
    relevance REAL DEFAULT 0.5,
    crawled_at TEXT,
    FOREIGN KEY (project_id) REFERENCES research_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'insight',
    confidence REAL DEFAULT 0.5,
    source_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES research_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS follow_ups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    sources TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES research_projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_project ON sources(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON source_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_findings_project ON findings(project_id);
CREATE INDEX IF NOT EXISTS idx_followups_project ON follow_ups(project_id);
CREATE INDEX IF NOT EXISTS idx_activity ON activity_log(created_at DESC);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, content='source_chunks', content_rowid='id'
);
"""


class Database:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.conn: aiosqlite.Connection | None = None

    async def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(str(self.db_path))
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA foreign_keys=ON")
        await self.conn.executescript(SCHEMA)
        for s in FTS_SCHEMA.strip().split(";"):
            s = s.strip()
            if s:
                try: await self.conn.execute(s)
                except: pass
        await self.conn.commit()
        log.info(f"Database initialized: {self.db_path}")

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Projects ───────────────────────────────────────────────────

    async def create_project(self, title: str, query: str) -> dict:
        now = self._now()
        c = await self.conn.execute(
            "INSERT INTO research_projects (title, query, status, created_at) VALUES (?, ?, 'pending', ?)",
            (title, query, now))
        await self.conn.commit()
        return await self.get_project(c.lastrowid)

    async def get_project(self, project_id: int) -> dict | None:
        c = await self.conn.execute("SELECT * FROM research_projects WHERE id = ?", (project_id,))
        r = await c.fetchone()
        return dict(r) if r else None

    async def list_projects(self, limit: int = 50) -> list[dict]:
        c = await self.conn.execute("SELECT * FROM research_projects ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in await c.fetchall()]

    async def update_project(self, project_id: int, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [project_id]
        await self.conn.execute(f"UPDATE research_projects SET {sets} WHERE id = ?", vals)
        await self.conn.commit()

    async def delete_project(self, project_id: int) -> bool:
        for tbl in ["sources", "findings", "follow_ups", "source_chunks"]:
            if tbl == "source_chunks":
                await self.conn.execute(f"DELETE FROM source_chunks WHERE source_id IN (SELECT id FROM sources WHERE project_id = ?)", (project_id,))
            else:
                await self.conn.execute(f"DELETE FROM {tbl} WHERE project_id = ?", (project_id,))
        cur = await self.conn.execute("DELETE FROM research_projects WHERE id = ?", (project_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    # ── Sources ────────────────────────────────────────────────────

    async def add_source(self, project_id: int, url: str, title: str = None,
                         content: str = None, summary: str = None, relevance: float = 0.5) -> dict:
        c = await self.conn.execute(
            "INSERT INTO sources (project_id, url, title, content, summary, relevance, crawled_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, url, title, content, summary, relevance, self._now()))
        await self.conn.execute("UPDATE research_projects SET sources_count = sources_count + 1 WHERE id = ?", (project_id,))
        await self.conn.commit()
        return {"id": c.lastrowid, "url": url, "title": title}

    async def get_sources(self, project_id: int) -> list[dict]:
        c = await self.conn.execute("SELECT * FROM sources WHERE project_id = ? ORDER BY relevance DESC", (project_id,))
        return [dict(r) for r in await c.fetchall()]

    async def add_chunk(self, source_id: int, chunk_index: int, content: str) -> int:
        c = await self.conn.execute(
            "INSERT INTO source_chunks (source_id, chunk_index, content) VALUES (?, ?, ?)",
            (source_id, chunk_index, content))
        cid = c.lastrowid
        try:
            await self.conn.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", (cid, content))
        except: pass
        await self.conn.commit()
        return cid

    async def search_chunks(self, query: str, project_id: int = None, limit: int = 10) -> list[dict]:
        try:
            sql = """SELECT sc.*, s.url, s.title as source_title, s.project_id
                     FROM chunks_fts fts
                     JOIN source_chunks sc ON sc.id = fts.rowid
                     JOIN sources s ON s.id = sc.source_id
                     WHERE chunks_fts MATCH ?"""
            params = [query]
            if project_id:
                sql += " AND s.project_id = ?"
                params.append(project_id)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            c = await self.conn.execute(sql, params)
            return [dict(r) for r in await c.fetchall()]
        except:
            sql = "SELECT sc.*, s.url, s.title as source_title FROM source_chunks sc JOIN sources s ON s.id = sc.source_id WHERE sc.content LIKE ? LIMIT ?"
            c = await self.conn.execute(sql, (f"%{query}%", limit))
            return [dict(r) for r in await c.fetchall()]

    # ── Findings ───────────────────────────────────────────────────

    async def add_finding(self, project_id: int, content: str, category: str = "insight",
                          confidence: float = 0.5, source_ids: list = None) -> dict:
        c = await self.conn.execute(
            "INSERT INTO findings (project_id, content, category, confidence, source_ids, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, content, category, confidence, json.dumps(source_ids or []), self._now()))
        await self.conn.commit()
        return {"id": c.lastrowid, "content": content, "category": category, "confidence": confidence}

    async def get_findings(self, project_id: int) -> list[dict]:
        c = await self.conn.execute("SELECT * FROM findings WHERE project_id = ? ORDER BY confidence DESC", (project_id,))
        results = []
        for r in await c.fetchall():
            d = dict(r)
            try: d["source_ids"] = json.loads(d["source_ids"])
            except: pass
            results.append(d)
        return results

    # ── Follow-ups ─────────────────────────────────────────────────

    async def add_follow_up(self, project_id: int, question: str, answer: str = None,
                            sources: list = None) -> dict:
        c = await self.conn.execute(
            "INSERT INTO follow_ups (project_id, question, answer, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, question, answer, json.dumps(sources or []), self._now()))
        await self.conn.commit()
        return {"id": c.lastrowid, "question": question}

    async def get_follow_ups(self, project_id: int) -> list[dict]:
        c = await self.conn.execute("SELECT * FROM follow_ups WHERE project_id = ? ORDER BY id", (project_id,))
        return [dict(r) for r in await c.fetchall()]

    # ── Activity ───────────────────────────────────────────────────

    async def log_event(self, event_type: str, message: str, project_id: int = None, data: dict = None):
        await self.conn.execute(
            "INSERT INTO activity_log (project_id, event_type, message, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, event_type, message, json.dumps(data, default=str) if data else None, self._now()))
        await self.conn.commit()

    async def get_activity(self, project_id: int = None, limit: int = 50) -> list[dict]:
        if project_id:
            c = await self.conn.execute("SELECT * FROM activity_log WHERE project_id = ? ORDER BY created_at DESC LIMIT ?", (project_id, limit))
        else:
            c = await self.conn.execute("SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in await c.fetchall()]

    # ── Stats ──────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        p = await self.conn.execute("SELECT COUNT(*) as c FROM research_projects")
        s = await self.conn.execute("SELECT COUNT(*) as c FROM sources")
        f = await self.conn.execute("SELECT COUNT(*) as c FROM findings")
        ch = await self.conn.execute("SELECT COUNT(*) as c FROM source_chunks")
        return {
            "projects": (await p.fetchone())["c"],
            "sources": (await s.fetchone())["c"],
            "findings": (await f.fetchone())["c"],
            "chunks": (await ch.fetchone())["c"],
        }
