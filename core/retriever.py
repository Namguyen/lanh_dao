"""Stage 2: Database retrieval.

This module handles searching the Elasticsearch database for candidates
based on extracted entities from the intent classification stage.
"""

import logging

import config

log = logging.getLogger(__name__)


def retrieve_candidates(db, entities: list[str], search_mode: str) -> tuple[list, list]:
    """Search the database for each entity and return merged, score-sorted results.

    Results are sorted by score descending so the global best is always at index 0,
    regardless of which entity produced it.

    Args:
        db: CandidateDB instance with search method
        entities: List of entity names to search for
        search_mode: "SINGLE" or "LIST" to determine result limit

    Returns:
        Tuple of (all_results, retrieval_trace)
        - all_results: List of candidate tuples sorted by score
        - retrieval_trace: List of trace info for debugging
    """
    limit = config.LIST_SEARCH_LIMIT if search_mode == "LIST" else config.SINGLE_SEARCH_LIMIT

    all_results = []
    retrieval_trace = []
    for entity in entities:
        hits = db.search(entity, limit=limit)

        if not hits:
            continue

        all_results.extend(hits)
        log.info("Entity '%s': %d hits, top_score=%.2f", entity, len(hits), hits[0][3])

    # Sort globally so the highest-scoring candidate is always first
    all_results.sort(key=lambda r: r[3], reverse=True)

    # Deduplicate by (name, birth year) — multiple entity terms can match the same person
    unique: list = []
    seen: set = set()
    for r in all_results:
        key = (r[0], r[1])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    all_results = unique

    return all_results, retrieval_trace
