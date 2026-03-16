"""DeepResearch CLI -- AI Research Assistant."""
import asyncio, json, sys
import click
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

def _run(coro): return asyncio.run(coro)

async def _services():
    from src.db.database import Database
    from src.ai.llm import LLM
    from src.research.engine import ResearchEngine
    db = Database()
    await db.initialize()
    llm = LLM()
    engine = ResearchEngine(db, llm)
    return db, llm, engine

@click.group()
def cli():
    """DeepResearch -- AI Research Assistant."""
    pass

@cli.command()
def status():
    """Show system status."""
    async def _():
        db, llm, _ = await _services()
        stats = await db.get_stats()
        click.echo()
        click.secho("  +===================================+", fg="magenta")
        click.secho("  |     DEEPRESEARCH STATUS           |", fg="magenta")
        click.secho("  +===================================+", fg="magenta")
        click.echo()
        click.echo(f"  LLM:       {llm.provider}/{llm.model}")
        h = click.style("HEALTHY", fg="green") if llm.is_healthy else click.style("OFFLINE", fg="red")
        click.echo(f"  Health:    {h}")
        click.echo(f"  Projects:  {stats['projects']}")
        click.echo(f"  Sources:   {stats['sources']}")
        click.echo(f"  Findings:  {stats['findings']}")
        click.echo(f"  Chunks:    {stats['chunks']}")
        click.echo()
        await db.close()
    _run(_())

@cli.command()
@click.argument("topic", nargs=-1, required=True)
def research(topic):
    """Start a deep research on a topic."""
    topic_str = " ".join(topic)
    async def _():
        db, llm, engine = await _services()
        click.secho(f"\n  Researching: {topic_str}\n", fg="cyan")

        async def on_event(t, d):
            if t == "step":
                click.secho(f"  >> {d.get('step', '')}", fg="yellow")

        engine.on_event = on_event
        result = await engine.research(topic_str)

        if result.get("status") == "completed":
            click.echo(f"\n{result['report']}\n")
            click.secho(f"  Sources: {result['sources']} | Findings: {result['findings']}", fg="white")
        else:
            click.secho(f"  Error: {result.get('error')}", fg="red")
        click.echo()
        await db.close()
    _run(_())

@cli.command()
@click.argument("project_id", type=int)
@click.argument("question", nargs=-1, required=True)
def ask(project_id, question):
    """Ask a follow-up question about a research project."""
    q = " ".join(question)
    async def _():
        db, llm, engine = await _services()
        click.secho(f"\n  Q: {q}\n", fg="cyan")
        result = await engine.follow_up(project_id, q)
        if "error" in result:
            click.secho(f"  Error: {result['error']}", fg="red")
        else:
            click.echo(f"  A: {result['answer']}\n")
            if result.get("sources"):
                click.secho(f"  Sources:", fg="white")
                for s in result["sources"]:
                    click.echo(f"    - {s.get('title', s.get('url', '?'))}")
        click.echo()
        await db.close()
    _run(_())

@cli.command("projects")
def list_projects():
    """List research projects."""
    async def _():
        db, *_ = await _services()
        projects = await db.list_projects()
        if not projects:
            click.echo("  No projects yet. Start one: python run.py research \"your topic\"")
            await db.close()
            return
        click.echo()
        click.echo(f"  {'ID':<5} {'STATUS':<12} {'SOURCES':<8} {'TOPIC':<50} {'DATE'}")
        click.echo(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*50} {'-'*19}")
        for p in projects:
            s_color = {"completed": "green", "failed": "red", "researching": "yellow"}.get(p["status"], "white")
            status = click.style(p["status"], fg=s_color)
            click.echo(f"  {p['id']:<5} {status:<21} {p['sources_count']:<8} {p['title'][:49]:<50} {p['created_at'][:19]}")
        click.echo()
        await db.close()
    _run(_())

@cli.command()
@click.argument("project_id", type=int)
def show(project_id):
    """Show a research project with findings and report."""
    async def _():
        db, *_ = await _services()
        p = await db.get_project(project_id)
        if not p:
            click.secho(f"  Project {project_id} not found.", fg="red")
            await db.close()
            return
        findings = await db.get_findings(project_id)
        sources = await db.get_sources(project_id)
        click.echo()
        click.echo(f"  Topic:    {p['title']}")
        click.echo(f"  Status:   {p['status']}")
        click.echo(f"  Sources:  {len(sources)}")
        click.echo(f"  Findings: {len(findings)}")
        if findings:
            click.echo(f"\n  Key Findings:")
            for f in findings:
                conf = click.style(f"[{f['confidence']:.1f}]", fg="yellow")
                click.echo(f"    {conf} [{f['category']}] {f['content'][:100]}")
        if sources:
            click.echo(f"\n  Sources:")
            for s in sources:
                click.echo(f"    - {s.get('title', s['url'])[:60]} ({s['url'][:50]})")
        if p.get("report"):
            click.echo(f"\n  Report:\n{p['report']}")
        click.echo()
        await db.close()
    _run(_())

@cli.command()
@click.argument("query")
@click.option("--limit", default=5)
def search(query, limit):
    """Search across all research content."""
    async def _():
        db, *_ = await _services()
        results = await db.search_chunks(query, limit=limit)
        if not results:
            click.echo(f"  No results for '{query}'")
            await db.close()
            return
        click.echo(f"\n  Found {len(results)} results:\n")
        for r in results:
            click.secho(f"  --- {r.get('source_title', '?')} ---", fg="cyan")
            click.echo(f"  {r['content'][:250]}\n")
        await db.close()
    _run(_())

@cli.command()
@click.option("--port", default=None, type=int)
def serve(port):
    """Start the web dashboard."""
    import uvicorn
    from src.config import DEEPRESEARCH_PORT
    p = port or DEEPRESEARCH_PORT
    click.secho(f"\n  Starting DeepResearch on http://localhost:{p}\n", fg="magenta")
    uvicorn.run("src.web.api:app", host="0.0.0.0", port=p, reload=False, log_level="info")

if __name__ == "__main__":
    cli()
