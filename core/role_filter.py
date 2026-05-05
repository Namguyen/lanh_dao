"""Stage 3: Role-based filtering and reranking.

This module applies generic role-alignment rules to filter and rerank candidates
based on query modifiers, core token overlap, and role specificity.
"""

import unicodedata

import config


def _normalise_text(text: str) -> str:
    """Normalize Vietnamese text for robust keyword rule checks."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def entity_matches_position(normalized_entity: str, normalized_position: str) -> bool:
    """Return True when a role entity matches a position segment.

    Match policy is anchored subsequence on semicolon-separated segments:
    - First token of entity must match first token of a segment
    - Remaining entity tokens must appear in order within that segment

    This allows inserts like "thuong truc" in
    "pho thu tuong thuong truc chinh phu" for entity "pho thu tuong chinh phu",
    while rejecting prefix-subset false matches like
    entity "chu tich quoc hoi" vs segment "pho chu tich quoc hoi".
    """
    entity_tokens = normalized_entity.split()
    if not entity_tokens:
        return False

    for segment in normalized_position.split(";"):
        seg_tokens = segment.strip().split()
        if not seg_tokens:
            continue
        if seg_tokens[0] != entity_tokens[0]:
            continue

        ei = 1
        si = 1
        while ei < len(entity_tokens) and si < len(seg_tokens):
            if seg_tokens[si] == entity_tokens[ei]:
                ei += 1
            si += 1

        if ei == len(entity_tokens):
            return True

    return False


def _extract_modifier_phrases(normalized_text: str) -> set[str]:
    """Return configured modifier phrases appearing in the given normalized text."""
    text_tokens = normalized_text.split()
    found: set[str] = set()

    for phrase in config.ROLE_MODIFIER_PHRASES:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        if not n:
            continue

        for i in range(len(text_tokens) - n + 1):
            if text_tokens[i : i + n] == phrase_tokens:
                found.add(phrase)
                break

    return found


def _extract_core_query_tokens(normalized_query: str) -> list[str]:
    """Extract role-defining tokens by removing filler and modifier tokens."""
    modifier_tokens = {
        token
        for phrase in config.ROLE_MODIFIER_PHRASES
        for token in phrase.split()
    }
    return [
        t
        for t in normalized_query.split()
        if t not in config.ROLE_QUERY_FILLER_TOKENS and t not in modifier_tokens and len(t) > 1
    ]


def _is_role_like_query(normalized_query: str) -> bool:
    """Return True when query appears to target a role/title rather than a person name."""
    return any(hint in normalized_query for hint in config.SPECIFIC_ROLE_HINTS)


def _candidate_role_features(candidate: tuple, normalized_query: str) -> dict:
    """Compute role-alignment features for one candidate tuple."""
    position_norm = _normalise_text(candidate[2])
    query_modifiers = _extract_modifier_phrases(normalized_query)
    candidate_modifiers = _extract_modifier_phrases(position_norm)
    core_tokens = _extract_core_query_tokens(normalized_query)

    overlap = sum(1 for token in core_tokens if token in position_norm)
    core_overlap_ratio = overlap / max(len(core_tokens), 1)

    missing_required = len(query_modifiers - candidate_modifiers)
    # If query does not ask for a modifier, prefer cleaner primary roles.
    unexpected_modifiers = len(candidate_modifiers - query_modifiers)

    return {
        "position_norm": position_norm,
        "query_modifiers": query_modifiers,
        "candidate_modifiers": candidate_modifiers,
        "core_overlap_ratio": core_overlap_ratio,
        "missing_required": missing_required,
        "unexpected_modifiers": unexpected_modifiers,
    }


def rerank_by_generic_role_rules(results: list, user_input: str) -> list:
    """Apply conservative query-aware reranking rules.

    This keeps the existing ES scoring intact and only adjusts ordering via
    generic role modifiers (e.g. pho/thuong truc/van phong), not hardcoded titles.
    """
    return _rerank_by_generic_role_rules_impl(results, user_input)


def _rerank_by_generic_role_rules_impl(results: list, user_input: str) -> list:
    """Rerank by generic role rules without title-specific hardcoding."""
    if not results:
        return results

    normalized_query = _normalise_text(user_input)
    if not _is_role_like_query(normalized_query):
        return results

    scored = []
    for row in results:
        raw_score = float(row[3])
        features = _candidate_role_features(row, normalized_query)

        # Weighted adjustment: missing required modifiers is strongest penalty,
        # then unexpected modifiers, while preserving original retrieval score.
        adjusted = (
            raw_score
            + features["core_overlap_ratio"] * config.ROLE_CORE_OVERLAP_WEIGHT
            - features["missing_required"] * config.ROLE_MISSING_MODIFIER_PENALTY
            - features["unexpected_modifiers"] * config.ROLE_EXTRA_MODIFIER_PENALTY
        )
        scored.append((adjusted, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored]


def filter_single_role_candidates(candidates: list, normalized_query: str) -> list:
    """Keep best role-aligned candidates for SINGLE role-like queries."""
    if not candidates or not _is_role_like_query(normalized_query):
        return candidates

    decorated = []
    for row in candidates:
        features = _candidate_role_features(row, normalized_query)
        decorated.append((row, features))

    eligible = [
        (row, feat)
        for row, feat in decorated
        if feat["missing_required"] == 0 and feat["core_overlap_ratio"] >= config.ROLE_MIN_CORE_OVERLAP
    ]
    pool = eligible if eligible else decorated

    min_unexpected = min(feat["unexpected_modifiers"] for _, feat in pool)
    filtered = [row for row, feat in pool if feat["unexpected_modifiers"] == min_unexpected]

    return filtered if filtered else candidates


def apply_list_query_filters(results: list, user_input: str, entity_only: str = "") -> list:
    """Apply precise filters for list/count queries to avoid undercounting.

    Use generic role alignment so LIST mode does not rely on title-specific rules.
    """
    if not results:
        return results

    normalized_query = _normalise_text(user_input)
    filtered = results

    if _is_role_like_query(normalized_query):
        decorated = []
        for row in filtered:
            features = _candidate_role_features(row, normalized_query)
            decorated.append((row, features))

        role_aligned = [
            (row, feat)
            for row, feat in decorated
            if feat["missing_required"] == 0 and feat["core_overlap_ratio"] >= config.ROLE_MIN_CORE_OVERLAP
        ]
        pool = role_aligned if role_aligned else decorated

        query_modifiers = _extract_modifier_phrases(normalized_query)
        if query_modifiers:
            # For explicit modifier queries (e.g. "pho ..."), keep all aligned rows.
            filtered = [row for row, _ in pool]
        else:
            # For principal-role queries, minimize unexpected modifiers.
            min_unexpected = min(feat["unexpected_modifiers"] for _, feat in pool)
            filtered = [row for row, feat in pool if feat["unexpected_modifiers"] == min_unexpected]

    # If query targets a specific ministry, keep only candidates whose
    # positions strongly overlap with that ministry phrase.
    query_tokens = [
        t for t in normalized_query.split()
        if t not in {"lanh", "dao", "va", "cac", "nhung", "co", "bao", "nhieu", "la", "ai"}
    ]
    if "bo" in query_tokens:
        target_tokens = [t for t in query_tokens if len(t) > 1]
        org_filtered = []
        for row in filtered:
            pos_tokens = set(_normalise_text(row[2]).split())
            overlap = sum(1 for token in target_tokens if token in pos_tokens)
            overlap_ratio = overlap / max(len(target_tokens), 1)

            # Require both the "bo" token and strong overlap to avoid pulling
            # unrelated ministries into metadata.
            if "bo" in pos_tokens and overlap_ratio >= 0.6:
                org_filtered.append(row)

        if org_filtered:
            filtered = org_filtered

    # Phrase-level filtering for role entities.
    if entity_only:
        normalized_entity = _normalise_text(entity_only)
        if _is_role_like_query(normalized_entity):
            phrase_filtered = [
                row for row in filtered
                if entity_matches_position(normalized_entity, _normalise_text(row[2]))
            ]
            if phrase_filtered:
                filtered = phrase_filtered

    # Deduplicate by full identity tuple while preserving ranking order.
    deduped = []
    seen = set()
    for row in filtered:
        key = (row[0], row[1], row[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
