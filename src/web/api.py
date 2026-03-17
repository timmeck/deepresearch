"""FastAPI for DeepResearch."""
import asyncio, json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.db.database import Database
from src.ai.llm import LLM
from src.research.engine import ResearchEngine
from src.web.auth import AuthMiddleware
from src.config import DEEPRESEARCH_PORT, REPORTS_DIR
from src.utils.logger import get_logger

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
        try: q.put_nowait(payload)
        except asyncio.QueueFull: dead.append(q)
    for q in dead: _subs.remove(q)

engine.on_event = broadcast

@asynccontextmanager
async def lifespan(app):
    await db.initialize()
    log.info(f"DeepResearch started (LLM: {llm.provider}/{llm.model})")
    yield
    await db.close()

app = FastAPI(title="DeepResearch", version="1.0.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/api/status")
async def api_status():
    stats = await db.get_stats()
    return {"status": "ok", "llm_provider": llm.provider, "llm_model": llm.model,
            "llm_healthy": llm.is_healthy, **stats}


# ── Nexus Protocol Endpoint ────────────────────────────────────────

@app.post("/nexus/handle")
async def nexus_handle(request: Request):
    """Handle incoming NexusRequest from the Nexus protocol layer."""
    import time, uuid
    body = await request.json()
    start = time.perf_counter_ns()
    capability = body.get("capability", "")
    query = body.get("query", "")
    req_id = body.get("request_id", "")
    from_agent = body.get("from_agent", "")

    try:
        if capability == "deep_research":
            result = await engine.research(query)
            answer = result.get("report", result.get("summary", str(result)))
            confidence = result.get("confidence", 0.80)
            sources = result.get("sources", [])
        elif capability == "fact_checking":
            result = await engine.research(f"Fact check: {query}")
            answer = result.get("report", str(result))
            confidence = result.get("confidence", 0.70)
            sources = result.get("sources", [])
        else:
            elapsed = (time.perf_counter_ns() - start) // 1_000_000
            return {"response_id": uuid.uuid4().hex, "request_id": req_id,
                    "from_agent": "deep-research", "to_agent": from_agent,
                    "status": "failed", "answer": "", "confidence": 0.0,
                    "error": f"Unsupported capability: {capability}",
                    "processing_ms": elapsed, "cost": 0.0, "sources": [], "meta": {}}

        elapsed = (time.perf_counter_ns() - start) // 1_000_000
        return {"response_id": uuid.uuid4().hex, "request_id": req_id,
                "from_agent": "deep-research", "to_agent": from_agent,
                "status": "completed", "answer": answer, "confidence": confidence,
                "processing_ms": elapsed, "cost": 0.05, "sources": sources, "meta": {"capability": capability}}
    except Exception as e:
        elapsed = (time.perf_counter_ns() - start) // 1_000_000
        return {"response_id": uuid.uuid4().hex, "request_id": req_id,
                "from_agent": "deep-research", "to_agent": from_agent,
                "status": "failed", "answer": "", "confidence": 0.0,
                "error": str(e), "processing_ms": elapsed, "cost": 0.0, "sources": [], "meta": {}}


# ── Research ───────────────────────────────────────────────────────

@app.post("/api/research")
async def api_start_research(request: Request, bg: BackgroundTasks):
    body = await request.json()
    topic = body.get("topic", "")
    if not topic: return JSONResponse({"error": "topic required"}, 400)

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
    if not p: return JSONResponse({"error": "Not found"}, 404)
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
    if not question: return JSONResponse({"error": "question required"}, 400)
    result = await engine.follow_up(pid, question)
    return result

# ── Search ─────────────────────────────────────────────────────────

@app.get("/api/search")
async def api_search(q: str, project_id: int = None, limit: int = 10):
    return {"results": await db.search_chunks(q, project_id=project_id, limit=limit), "query": q}

# ── Activity ───────────────────────────────────────────────────────

@app.get("/api/activity")
async def api_activity(project_id: int = None, limit: int = 50):
    return {"events": await db.get_activity(project_id=project_id, limit=limit)}

# ── SSE ────────────────────────────────────────────────────────────

@app.get("/api/events/stream")
async def sse_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subs.append(queue)
    async def gen():
        try:
            while True:
                if await request.is_disconnected(): break
                try:
                    p = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {p}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in _subs: _subs.remove(queue)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Dashboard ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = await db.get_stats()
    projects = await db.list_projects(limit=20)
    activity = await db.get_activity(limit=20)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "stats": stats, "projects": projects, "activity": activity,
        "llm_healthy": llm.is_healthy, "llm_provider": llm.provider, "llm_model": llm.model,
    })
