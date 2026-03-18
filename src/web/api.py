"""FastAPI for DeepResearch."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.ai.llm import LLM
from src.config import DEEPRESEARCH_PORT, NEXUS_URL
from src.db.database import Database
from src.research.engine import ResearchEngine
from src.utils.logger import get_logger
from src.web.auth import AuthMiddleware

log = get_logger("web")

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"

db = Database()
llm = LLM()
engine = ResearchEngine(db, llm)

_subs: list[asyncio.Queue] = []


async def broadcast(event_type, data):
    payload = json.dumps({"type": event_type, **data})
    dead = []
    for q in _subs:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subs.remove(q)


engine.on_event = broadcast


@asynccontextmanager
async def lifespan(app):
    await db.initialize()
    log.info(f"DeepResearch started (LLM: {llm.provider}/{llm.model})")
    yield
    await db.close()


app = FastAPI(title="DeepResearch", version="1.0.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)

from src.nexus_sdk import NexusAdapter  # noqa: E402

nexus = NexusAdapter(
    app=app,
    agent_name="deep-research",
    nexus_url=NEXUS_URL,
    endpoint=f"http://localhost:{DEEPRESEARCH_PORT}",
    description="AI Research Assistant — automated web research with citations",
    capabilities=[
        {
            "name": "deep_research",
            "description": "Conduct automated research on any topic",
            "languages": ["en"],
            "price_per_request": 0.05,
        },
        {
            "name": "fact_checking",
            "description": "Verify claims with source-backed research",
            "languages": ["en"],
            "price_per_request": 0.05,
        },
    ],
    tags=["research", "web", "citations", "facts"],
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/api/status")
async def api_status():
    stats = await db.get_stats()
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "llm_healthy": llm.is_healthy,
        **stats,
    }


@nexus.handle("deep_research")
async def handle_research(query: str, params: dict) -> dict:
    result = await engine.research(query)
    return {
        "result": result.get("report", result.get("summary", str(result))),
        "confidence": result.get("confidence", 0.80),
        "cost": 0.05,
        "sources": result.get("sources", []),
    }


@nexus.handle("fact_checking")
async def handle_fact_check(query: str, params: dict) -> dict:
    result = await engine.research(f"Fact check: {query}")
    return {
        "result": result.get("report", str(result)),
        "confidence": result.get("confidence", 0.70),
        "cost": 0.05,
        "sources": result.get("sources", []),
    }


# ── Research ───────────────────────────────────────────────────────


@app.post("/api/research")
async def api_start_research(request: Request, bg: BackgroundTasks):
    body = await request.json()
    topic = body.get("topic", "")
    if not topic:
        return JSONResponse({"error": "topic required"}, 400)

    result_holder = {}

    async def _run():
        result_holder["result"] = await engine.research(topic)

    bg.add_task(_run)
    return {"status": "started", "topic": topic}


# ── Projects ───────────────────────────────────────────────────────


@app.get("/api/projects")
async def api_list_projects(limit: int = 50):
    return {"projects": await db.list_projects(limit=limit)}


@app.get("/api/projects/{pid}")
async def api_get_project(pid: int):
    p = await db.get_project(pid)
    if not p:
        return JSONResponse({"error": "Not found"}, 404)
    sources = await db.get_sources(pid)
    findings = await db.get_findings(pid)
    follow_ups = await db.get_follow_ups(pid)
    return {"project": p, "sources": sources, "findings": findings, "follow_ups": follow_ups}


@app.delete("/api/projects/{pid}")
async def api_delete_project(pid: int):
    ok = await db.delete_project(pid)
    return {"status": "deleted"} if ok else JSONResponse({"error": "Not found"}, 404)


# ── Follow-up ──────────────────────────────────────────────────────


@app.post("/api/projects/{pid}/ask")
async def api_follow_up(pid: int, request: Request):
    body = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "question required"}, 400)
    result = await engine.follow_up(pid, question)
    return result


# ── Search ─────────────────────────────────────────────────────────


@app.get("/api/search")
async def api_search(q: str, project_id: int | None = None, limit: int = 10):
    return {"results": await db.search_chunks(q, project_id=project_id, limit=limit), "query": q}


# ── Activity ───────────────────────────────────────────────────────


@app.get("/api/activity")
async def api_activity(project_id: int | None = None, limit: int = 50):
    return {"events": await db.get_activity(project_id=project_id, limit=limit)}


# ── SSE ────────────────────────────────────────────────────────────


@app.get("/api/events/stream")
async def sse_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subs.append(queue)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    p = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {p}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in _subs:
                _subs.remove(queue)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Dashboard ──────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = await db.get_stats()
    projects = await db.list_projects(limit=20)
    activity = await db.get_activity(limit=20)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "stats": stats,
            "projects": projects,
            "activity": activity,
            "llm_healthy": llm.is_healthy,
            "llm_provider": llm.provider,
            "llm_model": llm.model,
        },
    )
