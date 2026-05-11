"""Stage 2: Database retrieval.

This module handles searching the Elasticsearch database for candidates
based on extracted entities from the intent classification stage.
"""

import logging

import constants

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
    limit = constants.LIST_SEARCH_LIMIT if search_mode == "LIST" else constants.SINGLE_SEARCH_LIMIT

    all_results = []
    retrieval_trace = []
    for entity in entities:
        hits = db.search(entity, limit=limit)

        # For LIST/group queries, merge in pure full-text matches to improve
        # recall without relying on manual score tuning or prefix rules.
        if search_mode == "LIST" and hasattr(db, "search_text"):
            try:
                text_hits = db.search_text(entity, limit=max(limit * 2, limit))
            except Exception:
                text_hits = []

            if text_hits:
                merged = {}
                for row in hits + text_hits:
                    key = (row[0], row[1], row[2])
                    prev = merged.get(key)
                    if prev is None or float(row[3]) > float(prev[3]):
                        merged[key] = row
                hits = list(merged.values())
                hits.sort(key=lambda r: r[3], reverse=True)
                hits = hits[: max(limit * 2, limit)]

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


def retrieve_per_entity(db, entities: list[str]) -> tuple[dict, list]:
    """For MULTI mode: search each entity individually and return best hit per name.

    Args:
        db: CandidateDB instance with search method
        entities: List of specific person names to look up

    Returns:
        Tuple of (per_entity, retrieval_trace)
        - per_entity: dict mapping entity name -> best candidate tuple (or None if not found)
        - retrieval_trace: List of trace info for debugging
    """
    per_entity: dict = {}
    retrieval_trace = []

    for entity in entities:
        entity = entity.strip()
        if not entity:
            continue
        hits = db.search(entity, limit=constants.SINGLE_SEARCH_LIMIT)
        if hits:
            per_entity[entity] = hits[0]
            log.info("MULTI entity '%s': best_score=%.2f → %s", entity, hits[0][3], hits[0][0])
        else:
            per_entity[entity] = None
            log.info("MULTI entity '%s': no hits", entity)

    return per_entity, retrieval_trace
