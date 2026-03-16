# Contributing to DeepResearch

## Setup
```bash
git clone https://github.com/timmeck/deepresearch.git
cd deepresearch
pip install -r requirements.txt
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Structure
```
src/
├── config.py              # Configuration
├── db/database.py         # SQLite (projects, sources, findings, chunks)
├── research/
│   ├── crawler.py         # Web crawling + text extraction
│   └── engine.py          # Research pipeline (plan, crawl, analyze, report)
├── ai/llm.py              # Ollama + Anthropic
└── web/api.py             # FastAPI
```
