"""Configuration for DeepResearch."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
_data_vol = Path(os.getenv("DATA_VOLUME", ""))
DB_PATH = _data_vol / "deepresearch.db" if _data_vol.is_dir() else ROOT_DIR / "deepresearch.db"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic" if ANTHROPIC_API_KEY else "ollama")
DEFAULT_MODEL = os.getenv("DEEPRESEARCH_MODEL", "claude-sonnet-4-6" if LLM_PROVIDER == "anthropic" else OLLAMA_MODEL)

DEEPRESEARCH_PORT = int(os.getenv("DEEPRESEARCH_PORT", "8400"))
NEXUS_URL = os.getenv("NEXUS_URL", "http://localhost:9500")
DEEPRESEARCH_API_KEY = os.getenv("DEEPRESEARCH_API_KEY", "")

MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
MAX_CRAWL_DEPTH = int(os.getenv("MAX_CRAWL_DEPTH", "3"))

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
