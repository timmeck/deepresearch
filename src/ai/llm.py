"""LLM for DeepResearch."""

import asyncio

import httpx

from src.config import (
    ANTHROPIC_API_KEY,
    DEFAULT_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_RETRY_DELAY,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from src.utils.logger import get_logger

log = get_logger("llm")


class LLM:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = DEFAULT_MODEL
        self._failures = 0
        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                self.provider = "ollama"
                self.model = OLLAMA_MODEL
                self.client = None
            else:
                from anthropic import AsyncAnthropic

                self.client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        else:
            self.provider = "ollama"
            self.model = OLLAMA_MODEL
            self.client = None
            log.info(f"Using Ollama ({OLLAMA_URL}) with model {self.model}")

    @property
    def is_healthy(self):
        return (self.client is not None or self.provider == "ollama") and self._failures < 5

    async def query(self, prompt: str, system: str = "", max_tokens: int = 2000) -> str | None:
        for attempt in range(LLM_MAX_RETRIES):
            try:
                if self.provider == "anthropic":
                    kw = {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if system:
                        kw["system"] = system
                    r = await self.client.messages.create(**kw)
                    self._failures = 0
                    return r.content[0].text
                else:
                    msgs = []
                    if system:
                        msgs.append({"role": "system", "content": system})
                    msgs.append({"role": "user", "content": prompt})
                    async with httpx.AsyncClient(timeout=120) as c:
                        resp = await c.post(
                            f"{OLLAMA_URL}/api/chat", json={"model": self.model, "messages": msgs, "stream": False}
                        )
                        resp.raise_for_status()
                    self._failures = 0
                    return resp.json().get("message", {}).get("content", "")
            except (TimeoutError, httpx.ConnectError, httpx.ReadError) as e:
                self._failures += 1
                log.warning(f"LLM network error (attempt {attempt + 1}): {type(e).__name__}: {e}")
            except httpx.HTTPStatusError as e:
                self._failures += 1
                log.error(f"LLM HTTP {e.response.status_code} (attempt {attempt + 1}): {e}")
            except Exception as e:
                self._failures += 1
                log.error(f"LLM unexpected error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                if attempt < LLM_MAX_RETRIES - 1:
                    await asyncio.sleep(LLM_RETRY_DELAY * (2**attempt))
        return None
