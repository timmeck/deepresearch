# DeepResearch — AI Research Assistant

[![CI](https://github.com/timmeck/deepresearch/actions/workflows/ci.yml/badge.svg)](https://github.com/timmeck/deepresearch/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

```
 ██████╗ ███████╗███████╗██████╗
 ██╔══██╗██╔════╝██╔════╝██╔══██╗
 ██║  ██║█████╗  █████╗  ██████╔╝
 ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝
 ██████╔╝███████╗███████╗██║
 ╚═════╝ ╚══════╝╚══════╝╚═╝
 ██████╗ ███████╗███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗
 ██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
 ██████╔╝█████╗  ███████╗█████╗  ███████║██████╔╝██║     ███████║
 ██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║
 ██║  ██║███████╗███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
 ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

Give it a topic — it crawls the web, analyzes sources, extracts findings, and generates a structured research report. All local with Ollama. Ask follow-up questions and it remembers everything.

**Pure Python + SQLite. 18 tests. No API keys required.**

![DeepResearch Dashboard](docs/dashboard.png)

## What It Does

```
Topic → Generate URLs → Crawl → Chunk → Index → Analyze → Extract Findings → Report
                                                                    ↓
                                                          Follow-up Questions
```

1. **Plan**: LLM generates relevant URLs to research
2. **Crawl**: Fetches web pages, extracts clean text
3. **Chunk & Index**: Splits content into FTS5-searchable segments
4. **Analyze**: LLM extracts key findings with confidence scores
5. **Report**: Synthesizes everything into a structured report with citations
6. **Follow-up**: Ask questions — it searches its gathered sources for answers

## Features

| Feature | Description |
|---|---|
| **Deep Research** | Automated web crawling + analysis + report generation |
| **Follow-up Questions** | Ask questions about completed research, with source citations |
| **Source Tracking** | Every finding linked to its source URL |
| **Confidence Scoring** | Findings rated high/medium/low confidence |
| **Finding Categories** | fact, insight, trend, comparison, warning |
| **Full-Text Search** | FTS5 search across all crawled content |
| **Research Projects** | Persistent projects with sources, findings, reports |
| **Dual LLM** | Ollama (free, local) or Anthropic Claude |
| **Web Dashboard** | 4-tab UI: Overview, Research, Findings, Search |
| **CLI** | Research, ask, search, show from command line |
| **Live Progress** | SSE updates during research (step-by-step) |
| **Auth** | Optional API key protection |
| **Docker** | One-command deployment |
| **CI** | GitHub Actions for Python 3.11-3.13 |

## Quick Start

```bash
git clone https://github.com/timmeck/deepresearch.git
cd deepresearch
pip install -r requirements.txt

# Make sure Ollama is running
ollama pull qwen3:14b

# Start a research project
python run.py research "What is WebAssembly and why does it matter?"

# List projects
python run.py projects

# Show full report
python run.py show 1

# Ask a follow-up question
python run.py ask 1 "How does it compare to JavaScript?"

# Search all gathered content
python run.py search "performance"

# Start the dashboard
python run.py serve
# → http://localhost:8400
```

## CLI Reference

```bash
python run.py status                           # System status
python run.py research "topic"                 # Start deep research
python run.py projects                         # List all projects
python run.py show PROJECT_ID                  # Show report + findings + sources
python run.py ask PROJECT_ID "question"        # Follow-up question
python run.py search "query"                   # Search all content
python run.py serve [--port 8400]              # Web dashboard
```

## Architecture

```
src/
├── config.py              # Configuration
├── db/
│   └── database.py        # SQLite (projects, sources, findings, chunks, FTS5)
├── research/
│   ├── crawler.py         # Web crawling, text extraction, chunking
│   └── engine.py          # THE CORE: plan → crawl → analyze → report
├── ai/
│   └── llm.py             # Ollama + Anthropic with retry
├── web/
│   ├── api.py             # FastAPI + SSE
│   ├── auth.py            # Auth middleware
│   └── templates/         # Dashboard
└── utils/
    └── logger.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System status |
| POST | `/api/research` | Start research on a topic |
| GET | `/api/projects` | List projects |
| GET | `/api/projects/{id}` | Project details + sources + findings + follow-ups |
| DELETE | `/api/projects/{id}` | Delete project |
| POST | `/api/projects/{id}/ask` | Follow-up question |
| GET | `/api/search?q=` | Search all content |
| GET | `/api/activity` | Activity timeline |
| GET | `/api/events/stream` | SSE live events |
| GET | `/` | Web dashboard |

## Configuration

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
DEEPRESEARCH_PORT=8400
# DEEPRESEARCH_API_KEY=secret
MAX_SEARCH_RESULTS=10
```

## Nexus Protocol

DeepResearch integrates with [Nexus](https://github.com/timmeck/nexus) via the NexusAdapter SDK.

| Capability | Description | Price |
|-----------|-------------|-------|
| `deep_research` | Conduct automated research on any topic | 0.05 |
| `fact_checking` | Verify claims with source-backed research | 0.05 |

**Features**: HMAC signature verification, automatic heartbeats (30s), auto-registration with Nexus on startup.

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
# 18 passed in 0.40s
```

## Docker

```bash
docker compose up -d
```

## Support

[![Star this repo](https://img.shields.io/github/stars/timmeck/deepresearch?style=social)](https://github.com/timmeck/deepresearch)
[![PayPal](https://img.shields.io/badge/Donate-PayPal-blue)](https://paypal.me/tmeck86)

---

Built by [Tim Mecklenburg](https://github.com/timmeck)
