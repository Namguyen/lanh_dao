"""Pipeline orchestrator for Vietnamese politician search.

This module coordinates all pipeline stages:
1. Intent analysis
2. Database retrieval
3. Score filtering
4. Internet enrichment (optional)
5. Answer generation

Returns structured response with metadata, confidence, and evidence.
"""

import logging
import time

from .intent_classifier import analyze_query_intent, is_ambiguous_leadership_query
from .retriever import retrieve_candidates, retrieve_per_entity
from .role_filter import (
    rerank_by_generic_role_rules,
    filter_single_role_candidates,
    filter_by_name_overlap,
    _is_role_like_query,
    entity_matches_position,
)
from .text_utils import normalize_role_text
from .internet_search import (
    should_search_internet,
    extract_web_sources,
    generate_evidence_first_news_answer,
)
from .llm_engine import format_direct_answer, format_multi_person_answer, generate_answer
import constants

log = logging.getLogger(__name__)


def _normalise_text(text: str) -> str:
    """Normalize Vietnamese text for robust keyword rule checks."""
    return normalize_role_text(text)


def _filter_and_build(
    results: list, search_mode: str, user_input: str, entity_only: str = ""
) -> tuple:
    """Apply score thresholds and build metadata + context string.

    Returns (best_person, db_context, metadata, strictly_valid, highest_score).
    - best_person: top-scoring tuple or None
    - db_context: formatted string for LLM prompt
    - metadata: dict (SINGLE) or list of dicts (LIST)
    """
    empty_single = {"name": None, "nam_sinh": None, "chuc_vu": None}
    no_match_msg = "Không có thông tin phù hợp trong Database nội bộ."

    if not results:
        return None, no_match_msg, [] if search_mode == "LIST" else empty_single, [], 0.0

    results = rerank_by_generic_role_rules(results, user_input)

    highest_score = results[0][3]
    if highest_score <= constants.MIN_SCORE_THRESHOLD:
        log.info("Top score %.2f below threshold; returning empty metadata", highest_score)
        return None, no_match_msg, [] if search_mode == "LIST" else empty_single, [], highest_score

    if search_mode == "LIST":
        # LIST queries are intended for counting/listing groups, so avoid
        # aggressive top-ratio filtering that can hide valid entries.
        from .role_filter import apply_list_query_filters

        strictly_valid = [r for r in results if r[3] >= constants.MIN_SCORE_THRESHOLD]
        strictly_valid = apply_list_query_filters(strictly_valid, user_input, entity_only)
        if not strictly_valid:
            strictly_valid = [
                r for r in results if r[3] >= highest_score * constants.SCORE_RELEVANCE_RATIO
            ]
    else:
        normalized_query = _normalise_text(user_input)
        strictly_valid = [
            r for r in results if r[3] >= highest_score * constants.SCORE_RELEVANCE_RATIO
        ]
        strictly_valid = filter_single_role_candidates(strictly_valid, normalized_query)

        # Phrase-level filter: use entity_only when available, else fall back to
        # normalized_query. This catches false positives where two roles share the
        # same modifier (e.g. "Phó Chủ tịch Thường trực Hội đồng Dân tộc" vs
        # "Phó Chủ tịch Thường trực Quốc hội").
        _phrase_source = _normalise_text(entity_only) if entity_only else normalized_query
        if _is_role_like_query(_phrase_source):
            phrase_filtered = [
                r for r in strictly_valid
                if entity_matches_position(_phrase_source, _normalise_text(r[2]))
            ]
            if phrase_filtered:
                strictly_valid = phrase_filtered

    if not strictly_valid:
        return None, no_match_msg, [] if search_mode == "LIST" else empty_single, [], highest_score

    best_person = strictly_valid[0]
    db_context = "\n".join(f"- {r[0]} ({r[1]}): {r[2]}" for r in strictly_valid)

    # FIX: build metadata according to actual search_mode to avoid type mismatch
    if search_mode == "LIST":
        metadata = [
            {"name": r[0], "nam_sinh": r[1], "chuc_vu": r[2]}
            for r in strictly_valid[: constants.LIST_METADATA_CAP]
        ]
    else:
        metadata = {
            "name": best_person[0],
            "nam_sinh": best_person[1],
            "chuc_vu": best_person[2],
        }

    return best_person, db_context, metadata, strictly_valid, highest_score


def _build_db_candidates(candidates: list) -> list[dict]:
    """Build compact evidence rows for top database candidates."""
    rows = []
    for row in candidates[:5]:
        rows.append(
            {
                "name": row[0],
                "nam_sinh": row[1],
                "chuc_vu": row[2],
                "score": round(float(row[3]), 2),
            }
        )
    return rows


def _estimate_confidence(
    highest_score: float, answer_mode: str, has_web_sources: bool
) -> float:
    """Estimate confidence from retrieval score and evidence availability."""
    if answer_mode == "no_match":
        return 0.0

    base = min(1.0, highest_score / max(constants.MIN_SCORE_THRESHOLD * 2, 1))

    if answer_mode == "database_only":
        base *= 0.9

    if answer_mode == "db_plus_web" and not has_web_sources:
        base *= 0.75

    return round(max(0.05, min(base, 1.0)), 2)


def process_query(user_input: str, db) -> dict:
    """Execute the full query pipeline shared by both CLI and API.

    Stages: intent analysis -> retrieval -> filtering -> internet -> answer.

    Args:
        user_input: Raw user query string
        db: CandidateDB instance with search method

    Returns:
        Dictionary with keys:
        - intent: "DATABASE" or "INTERNET"
        - search_mode: "SINGLE" or "LIST"
        - metadata: dict or list of candidate info
        - answer_mode: "database_only", "db_plus_web", "no_match", "needs_clarification"
        - confidence: 0.0-1.0 score
        - evidence: dict with db_candidates, web_sources, retrieval_trace
        - answer: Final answer string
    """
    timings_ms: dict[str, int] = {
        "intent": 0,
        "retrieval": 0,
        "filtering": 0,
        "internet": 0,
        "answer": 0,
    }

    # Stage 1: Intent analysis
    t0 = time.perf_counter()
    intent, _rewritten, search_mode, entity_only = analyze_query_intent(user_input)
    timings_ms["intent"] = int((time.perf_counter() - t0) * 1000)
    log.info("Intent=%s  mode=%s  entity='%s'", intent, search_mode, entity_only)

    if is_ambiguous_leadership_query(user_input, entity_only):
        clarify_answer = (
            "Câu hỏi còn quá chung nên chưa thể xác định một lãnh đạo cụ thể. "
            "Bạn vui lòng nêu rõ theo tên người hoặc chức danh/cơ quan, ví dụ: "
            "Bộ trưởng Bộ Công an, Bộ trưởng Bộ Quốc phòng, Thủ tướng, hoặc Chủ tịch Quốc hội."
        )
        empty_meta = {"name": None, "nam_sinh": None, "chuc_vu": None}
        return {
            "intent": intent,
            "search_mode": search_mode,
            "metadata": [] if search_mode == "LIST" else empty_meta,
            "answer_mode": "needs_clarification",
            "confidence": 0.1,
            "evidence": {
                "db_candidates": [],
                "web_sources": [],
                "retrieval_trace": [],
                "timings_ms": timings_ms,
            },
            "answer": clarify_answer,
        }

    if intent == "OUT_OF_SCOPE":
        empty_meta = {"name": None, "nam_sinh": None, "chuc_vu": None}
        return {
            "intent": intent,
            "search_mode": search_mode,
            "metadata": [] if search_mode == "LIST" else empty_meta,
            "answer_mode": "no_match",
            "confidence": 0.0,
            "evidence": {
                "db_candidates": [],
                "web_sources": [],
                "retrieval_trace": [],
                "timings_ms": timings_ms,
            },
            "answer": "Không có thông tin về nhân sự.",
        }

    # Stage 2: Database retrieval
    # ── MULTI mode: look up each person individually ──────────────────────────
    if search_mode == "MULTI":
        t0 = time.perf_counter()
        entity_names = [e.strip() for e in entity_only.split(",") if e.strip()]
        per_entity, retrieval_trace = retrieve_per_entity(db, entity_names)
        timings_ms["retrieval"] = int((time.perf_counter() - t0) * 1000)

        # Build metadata list and collect all found candidates for evidence
        metadata = []
        all_found = []
        for name in entity_names:
            hit = per_entity.get(name)
            if hit and hit[3] >= constants.MIN_SCORE_THRESHOLD:
                metadata.append({"name": hit[0], "nam_sinh": hit[1], "chuc_vu": hit[2]})
                all_found.append(hit)
            else:
                # Below threshold — treat as not found
                per_entity[name] = None

        t0 = time.perf_counter()
        answer = format_multi_person_answer(entity_names, per_entity)
        timings_ms["answer"] = int((time.perf_counter() - t0) * 1000)
        highest_score = max((h[3] for h in all_found), default=0.0)
        confidence = _estimate_confidence(highest_score, "database_only", False) if all_found else 0.1

        return {
            "intent": intent,
            "search_mode": search_mode,
            "metadata": metadata,
            "answer_mode": "database_only" if all_found else "no_match",
            "confidence": confidence,
            "evidence": {
                "db_candidates": _build_db_candidates(all_found),
                "web_sources": [],
                "retrieval_trace": retrieval_trace,
                "timings_ms": timings_ms,
            },
            "answer": answer,
        }

    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    entities = [e.strip() for e in entity_only.split(",")]
    all_results, retrieval_trace = retrieve_candidates(db, entities, search_mode)
    timings_ms["retrieval"] = int((time.perf_counter() - t0) * 1000)

    # Stage 3: Score filtering and metadata
    t0 = time.perf_counter()
    (
        best_person,
        db_context,
        metadata,
        strict_candidates,
        highest_score,
    ) = _filter_and_build(all_results, search_mode, user_input, entity_only)
    timings_ms["filtering"] = int((time.perf_counter() - t0) * 1000)

    # Stage 3b: Name-overlap guard (person-name queries only).
    if search_mode == "SINGLE" and best_person is not None and not _is_role_like_query(_normalise_text(entity_only)):
        name_matched = filter_by_name_overlap(strict_candidates, entity_only)
        if not name_matched:
            log.info("Name-overlap guard: no match for '%s'", entity_only)
            best_person = None
            strict_candidates = []
            metadata = {"name": None, "nam_sinh": None, "chuc_vu": None}

    # Ambiguity gate: SINGLE mode with multiple matched candidates.
    if search_mode == "SINGLE" and len(strict_candidates) > 1:
        top_score = strict_candidates[0][3]
        normalized_entity = _normalise_text(entity_only)
        if _is_role_like_query(normalized_entity):
            # Role query in SINGLE mode: return all matched holders of that role.
            rows_list = "\n".join(
                f"- {r[0]} (sinh {r[1]}): {r[2]}" for r in strict_candidates
            )
            return {
                "intent": intent,
                "search_mode": search_mode,
                "metadata": [
                    {"name": r[0], "nam_sinh": r[1], "chuc_vu": r[2]}
                    for r in strict_candidates
                ],
                "answer_mode": "database_only",
                "confidence": _estimate_confidence(top_score, "database_only", False),
                "evidence": {
                    "db_candidates": _build_db_candidates(strict_candidates),
                    "web_sources": [],
                    "retrieval_trace": retrieval_trace,
                    "timings_ms": timings_ms,
                },
                "answer": (
                    f"Có {len(strict_candidates)} người đang giữ chức vụ \"{entity_only}\":\n{rows_list}"
                ),
            }

        near_top = [
            r for r in strict_candidates
            if (top_score - r[3]) / max(top_score, 1) < 0.01
        ]
        if len(near_top) > 1:
            rows_list = "\n".join(
                f"- {r[0]} (sinh {r[1]}): {r[2]}" for r in near_top
            )
            return {
                "intent": intent,
                "search_mode": search_mode,
                "metadata": [
                    {"name": r[0], "nam_sinh": r[1], "chuc_vu": r[2]} for r in near_top
                ],
                "answer_mode": "needs_clarification",
                "confidence": 0.1,
                "evidence": {
                    "db_candidates": _build_db_candidates(near_top),
                    "web_sources": [],
                    "retrieval_trace": retrieval_trace,
                    "timings_ms": timings_ms,
                },
                "answer": (
                    f"Có {len(near_top)} người có tên \"{entity_only}\" trong dữ liệu. "
                    f"Bạn vui lòng cho biết thêm thông tin (năm sinh, chức vụ, cơ quan):\n{rows_list}"
                ),
            }

    # Hard safety gate: never answer from internet/LLM when internal DB has no match.
    # This prevents out-of-domain responses for unrelated public figures.
    if best_person is None:
        if search_mode == "LIST":
            answer = "Không tìm thấy nhân sự phù hợp trong dữ liệu nội bộ."
        else:
            answer = "Không có thông tin nhân sự phù hợp trong dữ liệu nội bộ."

        confidence = _estimate_confidence(highest_score, "no_match", False)
        return {
            "intent": intent,
            "search_mode": search_mode,
            "metadata": metadata,
            "answer_mode": "no_match",
            "confidence": confidence,
            "evidence": {
                "db_candidates": [],
                "web_sources": [],
                "retrieval_trace": retrieval_trace,
                "timings_ms": timings_ms,
            },
            "answer": answer,
        }

    # Stage 4: Optional internet enrichment
    # LIST queries (counting/listing groups) don't benefit from per-person news digests
    web_context = ""
    t0 = time.perf_counter()
    if search_mode != "LIST" and should_search_internet(intent, user_input):
        from ai_service import get_internet_info

        search_query = best_person[0]
        log.info("Internet search: %s", search_query)
        web_context = get_internet_info(search_query, person_name=best_person[0])
    timings_ms["internet"] = int((time.perf_counter() - t0) * 1000)

    web_sources = extract_web_sources(web_context)
    answer_mode = "db_plus_web" if web_sources else "database_only"
    confidence = _estimate_confidence(highest_score, answer_mode, bool(web_sources))

    # Stage 5: Answer generation
    is_news_query = any(kw in user_input.lower() for kw in constants.NEWS_KEYWORDS)

    t0 = time.perf_counter()
    if is_news_query:
        position = "; ".join(s.strip() for s in best_person[2].split(";") if s.strip())
        answer = generate_evidence_first_news_answer(best_person[0], position, web_context)
    elif answer_mode == "database_only" and confidence >= 0.7:
        # High-confidence pure DB result → skip LLM, format directly (~1-3s saved)
        answer = format_direct_answer(user_input, strict_candidates, search_mode)
    else:
        answer = generate_answer(user_input, db_context, web_context)
    timings_ms["answer"] = int((time.perf_counter() - t0) * 1000)

    return {
        "intent": intent,
        "search_mode": search_mode,
        "metadata": metadata,
        "answer_mode": answer_mode,
        "confidence": confidence,
        "evidence": {
            "db_candidates": _build_db_candidates(strict_candidates),
            "web_sources": web_sources,
            "retrieval_trace": retrieval_trace,
            "timings_ms": timings_ms,
        },
        "answer": answer,
    }
