import csv
import functools
import logging
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core import process_query
from es import AICandidateDB
import config

log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

@functools.lru_cache(maxsize=256)
def _cached_process_query(question: str):
    global db
    return process_query(question, db)

db: Optional[AICandidateDB] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    try:
        # Validate configuration before starting
        if not config.validate_config():
            log.error("Configuration validation failed. Starting with limited functionality.")
        
        db = AICandidateDB(verify_certs=False)  # Set to True in production with valid certs
        log.info("Database connection established.")
    except Exception as exc:
        log.error("Failed to connect to database on startup: %s", exc)
        db = None
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
    allow_origins=["*"],  # Restrict in production
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
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def _tokenize_lookup_query(text: str) -> list[str]:
    return [t for t in _normalize_lookup_text(text).split() if len(t) >= 2]


def _split_roles(position: str) -> list[str]:
    return [part.strip() for part in position.split(";") if part.strip()]


def _to_ascii(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.replace("đ", "d").split())


def _branch_of_role(role_text: str) -> str:
    role_norm = _to_ascii(role_text)
    if "bo chinh tri" in role_norm:
        return "Bộ Chính trị"
    if "chinh phu" in role_norm or "thu tuong" in role_norm:
        return "Chính phủ"
    if "quoc hoi" in role_norm:
        return "Quốc hội"
    if "dang" in role_norm or "trung uong" in role_norm or "bi thu" in role_norm:
        return "Đảng"
    if "tu lenh" in role_norm or "quan doi" in role_norm or "tong cuc" in role_norm:
        return "Quân đội"
    if "cong an" in role_norm:
        return "Công an"
    if "tinh uy" in role_norm or "thanh uy" in role_norm or "ubnd" in role_norm:
        return "Địa phương"
    if any(token in role_norm for token in ("chu tich", "pho chu tich")):
        return "Cơ quan Nhà nước"
    return "Khác"


def _org_bucket(role_text: str) -> str:
    role_norm = _to_ascii(role_text)
    if "ha noi" in role_norm:
        return "Hà Nội"
    if "ho chi minh" in role_norm or "tphcm" in role_norm:
        return "TP.HCM"
    if "chinh phu" in role_norm:
        return "Chính phủ"
    if "quoc hoi" in role_norm:
        return "Quốc hội"
    if "cong an" in role_norm:
        return "Bộ Công an"
    if "quoc phong" in role_norm or "quan doi" in role_norm:
        return "Bộ Quốc phòng"
    if "dang" in role_norm or "trung uong" in role_norm:
        return "Trung ương Đảng"
    return "Khác"


def _role_group(role_text: str) -> str:
    role_norm = _to_ascii(role_text)
    if "pho thu tuong" in role_norm:
        return "Nhóm Phó Thủ tướng"
    if "thu tuong" in role_norm:
        return "Nhóm Thủ tướng"
    if "tong bi thu" in role_norm:
        return "Nhóm Tổng Bí thư"
    if "bi thu" in role_norm:
        return "Nhóm Bí thư"
    if "chu tich" in role_norm and "pho" not in role_norm:
        return "Nhóm Chủ tịch"
    if "bo truong" in role_norm:
        return "Nhóm Bộ trưởng"
    if "thu truong" in role_norm:
        return "Nhóm Thứ trưởng"
    return "Nhóm Khác"


@functools.lru_cache(maxsize=1)
def _load_people_rows() -> list[dict[str, Any]]:
    csv_path = os.path.join(os.path.dirname(__file__), "data - danh_sach.csv")
    rows: list[dict[str, Any]] = []
    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, mode="r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Ten") or "").strip()
            position = (row.get("Chuc_Vu") or "").strip()
            if not name or not position:
                continue
            try:
                year = int((row.get("Nam_Sinh") or "0").strip() or "0")
            except ValueError:
                year = 0
            rows.append({"name": name, "nam_sinh": year, "chuc_vu": position})
    return rows


@functools.lru_cache(maxsize=1)
def _build_tree_data() -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    tree: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in _load_people_rows():
        roles = _split_roles(row["chuc_vu"])
        main_role = roles[0] if roles else row["chuc_vu"]
        branch = _branch_of_role(main_role)
        l1 = _org_bucket(main_role)
        l2 = _role_group(main_role)
        tree[branch][l1][l2].append(
            {
                "name": row["name"],
                "nam_sinh": row["nam_sinh"],
                "chuc_vu": row["chuc_vu"],
            }
        )

    out: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for branch, l1_map in tree.items():
        out[branch] = {}
        for l1, l2_map in l1_map.items():
            out[branch][l1] = {}
            for l2, people in l2_map.items():
                out[branch][l1][l2] = sorted(people, key=lambda p: p["name"])
    return out


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
def search(request: Request, body: QueryRequest, verbose: bool = False):
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


@app.get("/tree")
def tree() -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return organization tree used by the frontend tree demo."""
    return _build_tree_data()