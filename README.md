# lanh_dao - Vietnamese Leadership Search System

## Overview
A natural-language search system for Vietnamese leadership personnel data. The system uses Elasticsearch for hybrid search (BM25 + vector), Vietnamese embeddings, and DeepSeek LLM for answer generation.

## Features
- **Natural Language Queries**: Ask questions like "ai là thủ tướng", "danh sách lãnh đạo hà nội"
- **Hybrid Search**: Combines BM25 lexical search with Vietnamese vector embeddings
- **Intent Classification**: Automatically detects if user wants person info, list, or recent news
- **Internet Enrichment**: Fetches recent news from official Vietnamese sources when relevant
- **Confidence Scoring**: Returns confidence levels to help users assess answer reliability
- **Relationship Graph**: Visualize connections between leaders, roles, and organizations

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- API keys: DeepSeek (required), Serper (optional for internet search)

### 1. Setup Environment Variables
Create a `.env` file in the project root:
```bash
# Required: DeepSeek API key for LLM features
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Serper API key for internet/news search
SERPER_API_KEY=your_serper_api_key_here

# Elasticsearch credentials (change from defaults in production!)
ES_PASS=your_secure_password_here

# Optional: Override model names
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_MODEL_FAST=deepseek-v4-flash
```

### 2. Start Elasticsearch
```bash
docker-compose up -d elasticsearch kibana
```

Wait ~30 seconds for ES to be ready, then verify:
```bash
curl http://localhost:9200/_cluster/health
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Option A: CLI Mode (for testing)**
```bash
python app.py
```

**Option B: API Server (for production)**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open `index.html` in your browser or access API at `http://localhost:8000/docs`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Main search endpoint |
| `/lookup` | GET | Direct candidate lookup |
| `/tree` | GET | Organization tree structure |
| `/graph` | GET | Relationship graph data |
| `/health` | GET | Health check |

### Example Search Request
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "ai là thủ tướng việt nam"}'
```

## Architecture

```
User Query → Intent Analysis → DB Retrieval → Score Filtering → [Optional: Internet Search] → Answer Generation
                ↓                    ↓              ↓                    ↓                        ↓
           DeepSeek LLM        ES Hybrid      Role Matching       Serper API              DeepSeek LLM
                              Search         (if needed)          (optional)              (if needed)
```

### Key Components
- **`core.py`**: Main query pipeline (5 stages)
- **`es.py`**: Elasticsearch wrapper with hybrid search
- **`ai_service.py`**: Internet/news search via Serper
- **`api.py`**: FastAPI server
- **`config.py`**: Centralized configuration

## Configuration

Key parameters in `config.py`:
- `MIN_SCORE_THRESHOLD`: Minimum ES score to consider a match (default: 5.0)
- `SINGLE_SEARCH_LIMIT`: Max results for single-person queries (default: 12)
- `LIST_SEARCH_LIMIT`: Max results for list queries (default: 20)
- `OFFICIAL_NEWS_DOMAINS`: Whitelist of trusted news sources
- `BLOCKED_NEWS_DOMAINS`: Blacklist of unreliable sources

## Testing
```bash
python -m pytest tests/ -v
```

## Production Considerations

### Security
- ✅ Set strong ES password (change `ES_PASS` from default)
- ✅ Enable HTTPS for Elasticsearch (`verify_certs=True` in `api.py`)
- ✅ Restrict CORS origins in `api.py`
- ✅ Use secrets management (not .env files)
- ⚠️ Add authentication middleware

### Performance
- Add Redis caching for frequent queries
- Implement async queue for LLM calls
- Use connection pooling for ES
- Add rate limiting (already present, tune limits)

### Monitoring
- Add structured logging (JSON format)
- Integrate OpenTelemetry for tracing
- Set up metrics collection (latency, error rates)
- Create health dashboards

## Known Limitations
- No real-time data sync (CSV import only)
- Single-node ES setup (scale horizontally for production)
- No query analytics/logging persistence
- Limited error recovery for external API failures

## License
Internal use only.