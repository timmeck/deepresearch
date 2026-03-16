"""Web Crawler -- Fetch and extract text from URLs."""

import httpx
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

log = get_logger("crawler")

MAX_PAGE_SIZE = 512 * 1024  # 512KB


async def fetch_url(url: str) -> dict:
    """Fetch a URL and extract its text content."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                      headers={"User-Agent": "DeepResearch/1.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {"url": url, "title": "", "text": "", "error": f"Not HTML: {content_type}"}

            html = resp.text
            if len(html) > MAX_PAGE_SIZE:
                html = html[:MAX_PAGE_SIZE]

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

    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return {"url": url, "title": "", "text": "", "error": str(e)}


async def extract_links(url: str, html: str = None) -> list[str]:
    """Extract links from a page."""
    try:
        if not html:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
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
                from urllib.parse import urlparse
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
