"""Web Crawler -- Fetch and extract text from URLs."""

import random
import httpx
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

log = get_logger("crawler")

MAX_PAGE_SIZE = 512 * 1024  # 512KB

# Realistic browser User-Agent strings for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _random_ua() -> str:
    return random.choice(USER_AGENTS)


def _browser_headers(ua: str = None) -> dict:
    """Return realistic browser-like headers."""
    return {
        "User-Agent": ua or _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


async def fetch_url(url: str) -> dict:
    """Fetch a URL and extract its text content.

    Handles 403 by retrying with a different User-Agent.
    Handles 404 gracefully (skip, don't retry).
    """
    last_error = None

    # Try up to 3 times with different User-Agents
    for attempt in range(3):
        try:
            ua = _random_ua()
            async with httpx.AsyncClient(
                timeout=20, follow_redirects=True,
                headers=_browser_headers(ua),
            ) as client:
                resp = await client.get(url)

                # 404: skip immediately, don't retry
                if resp.status_code == 404:
                    log.info(f"404 Not Found, skipping: {url}")
                    return {"url": url, "title": "", "text": "", "error": "404 Not Found"}

                # 403: retry with a different User-Agent
                if resp.status_code == 403:
                    log.info(f"403 Forbidden (attempt {attempt+1}/3): {url}")
                    last_error = "403 Forbidden"
                    if attempt < 2:
                        continue
                    # On last attempt, try robots.txt check as fallback
                    accessible = await _check_robots_txt(url)
                    if not accessible:
                        return {"url": url, "title": "", "text": "", "error": "403 Forbidden (site blocks bots)"}
                    continue

                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return {"url": url, "title": "", "text": "", "error": f"Not HTML: {content_type}"}

                html = resp.text
                if len(html) > MAX_PAGE_SIZE:
                    html = html[:MAX_PAGE_SIZE]

                return _parse_html(url, html)

        except Exception as e:
            last_error = str(e)
            log.warning(f"Failed to fetch {url} (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                continue

    return {"url": url, "title": "", "text": "", "error": last_error or "Failed after retries"}


async def _check_robots_txt(url: str) -> bool:
    """Check if a site is accessible at all by fetching robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        async with httpx.AsyncClient(timeout=10, follow_redirects=True,
                                      headers=_browser_headers()) as client:
            resp = await client.get(robots_url)
            return resp.status_code < 400
    except Exception:
        return False


def _parse_html(url: str, html: str) -> dict:
    """Parse HTML and extract clean text content."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)

    return {"url": url, "title": title, "text": text[:50000]}  # Cap at 50K chars


async def search_web(query: str, num_results: int = 8) -> list[str]:
    """Search DuckDuckGo HTML and return result URLs.

    Uses DuckDuckGo's HTML-only search page to avoid API keys.
    Returns a list of URLs from the search results.
    """
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    urls = []

    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            headers=_browser_headers(),
        ) as client:
            resp = await client.get(search_url)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # DuckDuckGo HTML results are in <a class="result__a"> tags
            for a_tag in soup.select("a.result__a"):
                href = a_tag.get("href", "")
                if href.startswith("http"):
                    urls.append(href)
                elif href.startswith("//"):
                    urls.append(f"https:{href}")

            # Also try result__url class as fallback
            if not urls:
                for a_tag in soup.select("a.result__url"):
                    href = a_tag.get("href", "")
                    if href.startswith("http"):
                        urls.append(href)

            # Dedupe while preserving order
            seen = set()
            unique = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    unique.append(u)
            urls = unique

    except Exception as e:
        log.warning(f"DuckDuckGo search failed for '{query}': {e}")

    log.info(f"DuckDuckGo search '{query[:50]}' returned {len(urls)} URLs")
    return urls[:num_results]


async def extract_links(url: str, html: str = None) -> list[str]:
    """Extract links from a page."""
    try:
        if not html:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                          headers=_browser_headers()) as client:
                resp = await client.get(url)
                html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        links = []
        base = url.rsplit("/", 1)[0]

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http"):
                links.append(href)
            elif href.startswith("/"):
                parsed = urlparse(url)
                links.append(f"{parsed.scheme}://{parsed.netloc}{href}")

        return list(set(links))[:50]  # Dedupe, cap at 50
    except Exception:
        return []


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks
