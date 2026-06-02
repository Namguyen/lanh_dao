import functools
import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core import process_query
from core.text_utils import normalize_text, to_ascii_text, tokenize_normalized
from es import AICandidateDB
import config

log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

@functools.lru_cache(maxsize=256)
def _cached_process_query(question: str):
    global db
    return process_query(question, db)

db: Optional[AICandidateDB] = None
_DB_CONNECT_ATTEMPTS = 12
_DB_CONNECT_RETRY_SECONDS = 3

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    # Validate configuration before starting
    if not config.validate_config():
        log.error("Configuration validation failed. Starting with limited functionality.")

    for attempt in range(1, _DB_CONNECT_ATTEMPTS + 1):
        try:
            db = AICandidateDB(verify_certs=False)
            log.info("Database connection established.")
            break
        except Exception as exc:
            db = None
            if attempt == _DB_CONNECT_ATTEMPTS:
                log.error("Failed to connect to database after %d attempts: %s", attempt, exc)
                break
            log.warning(
                "Database unavailable on startup (attempt %d/%d): %s",
                attempt,
                _DB_CONNECT_ATTEMPTS,
                exc,
            )
            await asyncio.sleep(_DB_CONNECT_RETRY_SECONDS)
    yield
    db = None

app = FastAPI(
    title="Leadership Search API",
    description="Natural-language search system for Vietnamese leadership personnel data.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Thủ tướng Việt Nam là ai?"],
        description="Natural-language question in Vietnamese.",
    )


class QueryResponse(BaseModel):
    answer: str


class SlimResponse(BaseModel):
    answer: str
    metadata: dict | list


class QueryVerboseResponse(BaseModel):
    answer: str
    intent: str
    search_mode: str
    metadata: dict | list
    answer_mode: str
    confidence: float
    evidence: dict
    latency_ms: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


class LookupCandidate(BaseModel):
    name: str
    nam_sinh: int
    chuc_vu: str
    score: float


class LookupResponse(BaseModel):
    query: str
    total: int
    items: list[LookupCandidate]


def _normalize_lookup_text(text: str) -> str:
    """Normalize Vietnamese text for deterministic token matching in lookup."""
    return normalize_text(text)


def _tokenize_lookup_query(text: str) -> list[str]:
    return tokenize_normalized(text, min_len=2)


@app.get("/health")
def health():
    """Quick liveness check -- returns db status so callers know if ES is reachable."""
    return {"status": "ok", "db": "connected" if db is not None else "unavailable"}


@app.post(
    "/search",
    response_model=SlimResponse | QueryVerboseResponse,
    responses={500: {"model": ErrorResponse}},
)
@limiter.limit("30/minute")
def search(request: Request, body: QueryRequest, verbose: bool = True):
    """Accept a natural-language question.

    Example request body:
    ```json
    {"question": "Bộ trưởng Bộ Quốc phòng là ai?"}
    ```

    By default, only `answer` is returned.
    Set `verbose=true` to include metadata and diagnostics.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    start = time.time() if verbose else None
    try:
        result = _cached_process_query(body.question)
    except Exception as exc:
        log.error("Query processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not verbose:
        return SlimResponse(answer=result["answer"], metadata=result["metadata"])

    elapsed_ms = int((time.time() - start) * 1000)

    return QueryVerboseResponse(
        answer=result["answer"],
        intent=result["intent"],
        search_mode=result["search_mode"],
        metadata=result["metadata"],
        answer_mode=result["answer_mode"],
        confidence=result["confidence"],
        evidence=result["evidence"],
        latency_ms=elapsed_ms,
    )


@app.get("/lookup", response_model=LookupResponse)
@limiter.limit("60/minute")
def lookup(
    request: Request,
    q: str,
    limit: int = 20,
    role_keyword: str | None = None,
    min_score: float = 0.0,
):
    """Return deterministic ES candidates for lightweight profile exploration."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    normalized_query = q.strip()
    if not normalized_query:
        return LookupResponse(query=q, total=0, items=[])

    safe_limit = max(1, min(limit, 50))

    try:
        hits = db.search(normalized_query, limit=safe_limit)
    except Exception as exc:
        log.error("Lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not hits:
        return LookupResponse(query=normalized_query, total=0, items=[])

    top_score = float(hits[0][3])
    adaptive_min_score = max(min_score, top_score * 0.72)
    query_tokens = _tokenize_lookup_query(normalized_query)
    keyword = (role_keyword or "").strip().lower()

    rows = []
    for name, nam_sinh, chuc_vu, score in hits:
        if score < adaptive_min_score:
            continue
        if keyword and keyword not in chuc_vu.lower():
            continue

        # Reject semantic-only noise by requiring at least one query token
        # to appear in the candidate name/position after normalization.
        combined = _normalize_lookup_text(f"{name} {chuc_vu}")
        if query_tokens and not any(token in combined for token in query_tokens):
            continue

        rows.append(
            LookupCandidate(
                name=name,
                nam_sinh=nam_sinh,
                chuc_vu=chuc_vu,
                score=round(float(score), 2),
            )
        )

    return LookupResponse(query=normalized_query, total=len(rows), items=rows)

