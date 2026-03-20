"""Research Engine -- The core: plan, crawl, analyze, synthesize.

Pipeline:
1. Generate search queries from research topic
2. Crawl web pages for each query
3. Extract and chunk content
4. Analyze chunks for key findings
5. Synthesize findings into a structured report
6. Support follow-up questions with context
"""

from src.ai.llm import LLM
from src.config import MAX_SEARCH_RESULTS
from src.db.database import Database
from src.research.crawler import chunk_text, fetch_url, search_web, search_web_rich
from src.utils.logger import get_logger

log = get_logger("engine")


class ResearchEngine:
    def __init__(self, db: Database, llm: LLM):
        self.db = db
        self.llm = llm
        self.on_event = None  # SSE callback

    async def research(self, topic: str, depth: int = 1) -> dict:
        """Execute a full research cycle on a topic.

        Args:
            topic: The research topic/question.
            depth: Number of research passes (1=standard, 2+=iterative deepening).
                   After the first pass, the LLM identifies gaps and follow-up
                   questions, then runs additional search+crawl+analyze cycles.
        """
        # Create project
        project = await self.db.create_project(title=topic[:100], query=topic)
        pid = project["id"]

        await self._emit("research_started", {"project_id": pid, "topic": topic, "depth": depth})
        await self.db.log_event("research_started", f"Research (depth={depth}): {topic[:80]}", project_id=pid)
        await self.db.update_project(pid, status="researching")

        try:
            # === Pass 1: Initial research ===
            await self._emit("step", {"project_id": pid, "step": "Pass 1: Generating search queries..."})
            urls = await self._generate_search_urls(topic)

            await self._emit("step", {"project_id": pid, "step": f"Pass 1: Crawling {len(urls)} sources..."})
            for url in urls[:MAX_SEARCH_RESULTS]:
                await self._crawl_and_index(pid, url)

            await self._emit("step", {"project_id": pid, "step": "Pass 1: Analyzing content..."})
            await self._extract_findings(pid, topic)

            # === Passes 2..depth: Iterative deepening ===
            for current_depth in range(2, depth + 1):
                await self._emit("step", {
                    "project_id": pid,
                    "step": f"Pass {current_depth}/{depth}: Identifying gaps..."
                })
                gap_queries = await self._identify_gaps(pid, topic)

                if not gap_queries:
                    log.info(f"Pass {current_depth}: No gaps identified, stopping early")
                    break

                log.info(f"Pass {current_depth}: Found {len(gap_queries)} gap queries")

                for gq in gap_queries:
                    await self._emit("step", {
                        "project_id": pid,
                        "step": f"Pass {current_depth}/{depth}: Searching '{gq[:40]}...'"
                    })
                    gap_urls = await search_web(gq, num_results=5)

                    for url in gap_urls[:5]:
                        await self._crawl_and_index(pid, url)

                await self._emit("step", {
                    "project_id": pid,
                    "step": f"Pass {current_depth}/{depth}: Analyzing new content..."
                })
                await self._extract_findings(pid, topic)

            # === Final: Generate report ===
            await self._emit("step", {"project_id": pid, "step": "Generating report..."})
            report = await self._generate_report(pid, topic)

            # Complete
            await self.db.update_project(pid, status="completed", report=report, completed_at=self.db._now())
            await self.db.log_event("research_completed", f"Completed: {topic[:80]}", project_id=pid)
            await self._emit("research_completed", {"project_id": pid})

            sources = await self.db.get_sources(pid)
            findings = await self.db.get_findings(pid)

            return {
                "project_id": pid,
                "status": "completed",
                "topic": topic,
                "report": report,
                "sources": len(sources),
                "findings": len(findings),
                "depth": depth,
            }

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            await self.db.update_project(pid, status="failed")
            await self.db.log_event("research_failed", error, project_id=pid)
            await self._emit("research_failed", {"project_id": pid, "error": error})
            log.error(f"Research failed: {e}", exc_info=True)
            return {"project_id": pid, "status": "failed", "error": error}

    async def follow_up(self, project_id: int, question: str) -> dict:
        """Ask a follow-up question about a research project."""
        project = await self.db.get_project(project_id)
        if not project:
            return {"error": "Project not found"}

        # Search existing chunks for context
        chunks = await self.db.search_chunks(question, project_id=project_id, limit=5)

        context = ""
        source_refs = []
        if chunks:
            for c in chunks:
                context += f"\n[{c.get('source_title', c.get('url', '?'))}]:\n{c['content']}\n"
                source_refs.append({"url": c.get("url", ""), "title": c.get("source_title", "")})

        # Also include the report if available
        if project.get("report"):
            context += f"\n\n[Previous Report]:\n{project['report'][:3000]}\n"

        prompt = (
            f"Based on the following research context, answer this follow-up question.\n"
            f"Cite your sources.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )

        if not self.llm or not self.llm.is_healthy:
            answer = "LLM not available."
        else:
            answer = await self.llm.query(
                prompt,
                system="You answer research follow-up questions based on gathered sources. Cite sources. Be thorough.",
                max_tokens=2000,
            )

        if not answer:
            answer = "Failed to generate an answer."

        await self.db.add_follow_up(project_id, question, answer, source_refs)
        return {"answer": answer, "sources": source_refs}

    async def _generate_search_urls(self, topic: str) -> list[str]:
        """Use DuckDuckGo search as primary source, LLM-generated URLs as supplement."""
        # PRIMARY: real web search via DuckDuckGo
        search_urls = await search_web(topic, num_results=8)
        log.info(f"DuckDuckGo returned {len(search_urls)} URLs for '{topic[:50]}'")

        # SUPPLEMENT: LLM-generated URLs for known authoritative sources
        llm_urls = []
        result = await self.llm.query(
            f"Generate 3-5 specific URLs that would have valuable information about this research topic.\n"
            f"Include: Wikipedia, official docs, GitHub repos, reputable tech blogs.\n"
            f"Return ONLY URLs, one per line, no other text.\n\n"
            f"Topic: {topic}",
            system="You generate research URLs. Return only URLs, one per line.",
            max_tokens=300,
        )

        if result:
            for line in result.strip().split("\n"):
                line = line.strip().strip("-").strip("*").strip("0123456789.").strip()
                if line.startswith("http"):
                    llm_urls.append(line)

        # Merge: search URLs first, then LLM URLs (deduplicated)
        seen = set()
        merged = []
        for url in search_urls + llm_urls:
            if url not in seen:
                seen.add(url)
                merged.append(url)

        if not merged:
            # Last resort fallback
            merged = [
                f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                f"https://github.com/topics/{topic.lower().replace(' ', '-')}",
            ]

        log.info(f"Total URLs to crawl: {len(merged)} ({len(search_urls)} search + {len(llm_urls)} LLM)")
        return merged

    async def _crawl_and_index(self, project_id: int, url: str):
        """Crawl a URL, extract text, chunk and index it."""
        result = await fetch_url(url)
        if result.get("error") or not result.get("text"):
            return

        text = result["text"]
        title = result.get("title", url)

        # Store source
        source = await self.db.add_source(project_id, url, title=title, content=text[:10000], relevance=0.5)

        # Chunk and index
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            await self.db.add_chunk(source["id"], i, chunk)

        log.info(f"Crawled: {title[:50]} ({len(chunks)} chunks)")

    async def _extract_findings(self, project_id: int, topic: str):
        """Analyze crawled content and extract key findings."""
        sources = await self.db.get_sources(project_id)
        if not sources:
            return

        # Build content summary from all sources
        content_summary = ""
        for s in sources[:5]:  # Top 5 sources
            content = s.get("content", "")[:2000]
            content_summary += f"\n--- {s.get('title', s['url'])} ---\n{content}\n"

        result = await self.llm.query(
            f"Extract 5-10 key findings from these sources about: {topic}\n\n"
            f"For each finding, provide:\n"
            f"- The finding (one sentence)\n"
            f"- Confidence: high/medium/low\n"
            f"- Category: fact/insight/trend/comparison/warning\n\n"
            f"Format each as: [CATEGORY|CONFIDENCE] Finding text\n\n"
            f"SOURCES:\n{content_summary}",
            system="You extract research findings. Format: [category|confidence] finding",
            max_tokens=1500,
        )

        if not result:
            return

        for line in result.strip().split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Parse [CATEGORY|CONFIDENCE] format
            category = "insight"
            confidence = 0.5
            content = line

            if line.startswith("["):
                bracket_end = line.find("]")
                if bracket_end > 0:
                    meta = line[1:bracket_end]
                    content = line[bracket_end + 1 :].strip()
                    parts = meta.split("|")
                    if len(parts) >= 1:
                        category = parts[0].strip().lower()
                    if len(parts) >= 2:
                        conf_str = parts[1].strip().lower()
                        confidence = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf_str, 0.5)

            if content and len(content) > 10:
                await self.db.add_finding(project_id, content, category, confidence)

    async def _generate_report(self, project_id: int, topic: str) -> str:
        """Synthesize findings into a structured research report."""
        findings = await self.db.get_findings(project_id)
        sources = await self.db.get_sources(project_id)

        findings_text = "\n".join(f"- [{f['category']}] {f['content']}" for f in findings)
        sources_text = "\n".join(f"- {s.get('title', s['url'])} ({s['url']})" for s in sources)

        prompt = (
            f"Write a comprehensive research report on: {topic}\n\n"
            f"Based on these findings:\n{findings_text}\n\n"
            f"Sources consulted:\n{sources_text}\n\n"
            f"Structure the report with:\n"
            f"1. Executive Summary (2-3 sentences)\n"
            f"2. Key Findings (bullet points)\n"
            f"3. Analysis (detailed discussion)\n"
            f"4. Conclusions\n"
            f"5. Sources\n\n"
            f"Write in clear, professional prose. Cite sources where relevant."
        )

        report = await self.llm.query(
            prompt,
            system="You write thorough, well-structured research reports. Cite sources. Be comprehensive but concise.",
            max_tokens=3000,
        )

        return report or "Report generation failed."

    async def _identify_gaps(self, project_id: int, topic: str) -> list[str]:
        """Ask the LLM to identify 2-3 gaps/follow-up questions from current findings.

        Returns a list of search queries to fill the gaps.
        """
        findings = await self.db.get_findings(project_id)
        sources = await self.db.get_sources(project_id)

        if not findings:
            return []

        findings_text = "\n".join(f"- [{f['category']}] {f['content']}" for f in findings)
        sources_text = "\n".join(f"- {s.get('title', s['url'])}" for s in sources[:10])

        if not self.llm or not self.llm.is_healthy:
            return []

        result = await self.llm.query(
            f"You are analyzing research findings about: {topic}\n\n"
            f"CURRENT FINDINGS:\n{findings_text}\n\n"
            f"SOURCES ALREADY CONSULTED:\n{sources_text}\n\n"
            f"Identify 2-3 important gaps, missing perspectives, or follow-up questions "
            f"that would make this research more complete.\n\n"
            f"For each gap, write a specific web search query that would help fill it.\n"
            f"Return ONLY the search queries, one per line, no numbering or other text.",
            system="You identify research gaps and generate targeted search queries. Return only queries, one per line.",
            max_tokens=300,
        )

        if not result:
            return []

        queries = []
        for line in result.strip().split("\n"):
            line = line.strip().strip("-").strip("*").strip("0123456789.").strip()
            if line and len(line) > 5 and not line.startswith("#"):
                queries.append(line)

        return queries[:3]  # Cap at 3 follow-up queries

    async def _emit(self, event_type: str, data: dict):
        if self.on_event:
            await self.on_event(event_type, data)
