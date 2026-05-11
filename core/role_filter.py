"""Stage 3: Role-based filtering and reranking.

This module applies generic role-alignment rules to filter and rerank candidates
based on query modifiers, core token overlap, and role specificity.
"""

import constants
import re
from .text_utils import normalize_role_text, role_modifier_tokens


def _normalise_text(text: str) -> str:
    """Normalize Vietnamese text for robust keyword rule checks."""
    normalized = normalize_role_text(text)
    # Common city abbreviation used in user queries.
    normalized = re.sub(r"\btp\s*hcm\b|\btphcm\b", "thanh pho ho chi minh", normalized)
    return " ".join(normalized.split())


def entity_matches_position(normalized_entity: str, normalized_position: str) -> bool:
    """Return True when a role entity matches a position segment.

    Match policy (per semicolon/comma-separated segment):
    - First token of entity must equal first token of a segment (anchored).
    - Each subsequent entity token must appear in order; the ONLY tokens allowed
      between consecutive entity tokens are those from ROLE_MODIFIER_PHRASES.
      Any other intervening token causes the segment to be rejected.

    This correctly matches:
      entity "pho thu tuong chinh phu" vs segment "pho thu tuong thuong truc chinh phu"
      ("thuong truc" is a configured modifier, so it may appear between entity tokens)

    And correctly rejects:
      entity "chu tich nuoc" vs segment "chu tich quoc hoi nuoc chxhcn viet nam"
      ("quoc hoi" is not a modifier, so the segment is rejected even though "nuoc"
      appears later — "quoc hoi" fundamentally changes the title meaning)
    """
    entity_tokens = normalized_entity.split()
    if not entity_tokens:
        return False

    modifier_tokens = role_modifier_tokens()
    # Structural prefix tokens that appear in full official titles without changing
    # the role meaning, e.g. "Bộ" in "Bộ trưởng Bộ Quốc phòng" vs query "Bộ trưởng Quốc phòng".
    _STRUCTURAL = constants.ROLE_STRUCTURAL_GAP_TOKENS

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
            elif seg_tokens[si] in modifier_tokens or seg_tokens[si] in _STRUCTURAL:
                pass  # allowed gap token — skip it
            else:
                # Meaningful gap token changes the title — reject segment.
                break
            si += 1

        if ei == len(entity_tokens):
            return True

    return False


def _extract_modifier_phrases(normalized_text: str) -> set[str]:
    """Return configured modifier phrases appearing in the given normalized text."""
    text_tokens = normalized_text.split()
    found: set[str] = set()

    for phrase in constants.ROLE_MODIFIER_PHRASES:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        if not n:
            continue

        for i in range(len(text_tokens) - n + 1):
            if text_tokens[i : i + n] == phrase_tokens:
                # "pho" is a role modifier in titles like "pho bi thu",
                # but in geographic names "thanh pho" it is not a modifier.
                if phrase == "pho" and i > 0 and text_tokens[i - 1] == "thanh":
                    continue
                found.add(phrase)
                break

    return found


def _extract_core_query_tokens(normalized_query: str) -> list[str]:
    """Extract role-defining tokens by removing filler and modifier tokens."""
    modifier_tokens = role_modifier_tokens()
    return [
        t
        for t in normalized_query.split()
        if t not in constants.ROLE_QUERY_FILLER_TOKENS and t not in modifier_tokens and len(t) > 1
    ]


def _is_role_like_query(normalized_query: str) -> bool:
    """Return True when query appears to target a role/title rather than a person name."""
    return any(hint in normalized_query for hint in constants.SPECIFIC_ROLE_HINTS)


def _candidate_role_features(candidate: tuple, normalized_query: str) -> dict:
    """Compute role-alignment features for one candidate tuple."""
    position_norm = _normalise_text(candidate[2])
    query_modifiers = _extract_modifier_phrases(normalized_query)
    core_tokens = _extract_core_query_tokens(normalized_query)

    segments = [seg.strip() for seg in position_norm.split(";") if seg.strip()]
    if not segments:
        segments = [position_norm]

    segment_features = []
    for seg in segments:
        seg_modifiers = _extract_modifier_phrases(seg)
        seg_words = set(seg.split())
        overlap = sum(1 for token in core_tokens if token in seg_words)
        core_overlap_ratio = overlap / max(len(core_tokens), 1)
        missing_required = len(query_modifiers - seg_modifiers)
        # If query does not ask for a modifier, prefer cleaner primary roles.
        unexpected_modifiers = len(seg_modifiers - query_modifiers)
        segment_features.append(
            {
                "segment": seg,
                "candidate_modifiers": seg_modifiers,
                "core_overlap_ratio": core_overlap_ratio,
                "missing_required": missing_required,
                "unexpected_modifiers": unexpected_modifiers,
            }
        )

    # Score the segment that best reflects the queried role, then use its
    # alignment features for candidate-level filtering/ranking.
    best = min(
        segment_features,
        key=lambda feat: (
            feat["missing_required"],
            -feat["core_overlap_ratio"],
            feat["unexpected_modifiers"],
        ),
    )

    return {
        "position_norm": position_norm,
        "query_modifiers": query_modifiers,
        "candidate_modifiers": best["candidate_modifiers"],
        "core_overlap_ratio": best["core_overlap_ratio"],
        "missing_required": best["missing_required"],
        "unexpected_modifiers": best["unexpected_modifiers"],
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
            + features["core_overlap_ratio"] * constants.ROLE_CORE_OVERLAP_WEIGHT
            - features["missing_required"] * constants.ROLE_MISSING_MODIFIER_PENALTY
            - features["unexpected_modifiers"] * constants.ROLE_EXTRA_MODIFIER_PENALTY
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
        if feat["missing_required"] == 0 and feat["core_overlap_ratio"] >= constants.ROLE_MIN_CORE_OVERLAP
    ]
    pool = eligible if eligible else decorated

    min_unexpected = min(feat["unexpected_modifiers"] for _, feat in pool)
    filtered = [row for row, feat in pool if feat["unexpected_modifiers"] == min_unexpected]

    return filtered if filtered else candidates


# Common Vietnamese surnames/middle-name tokens that alone cannot identify a specific
# . personUsed by filter_by_name_overlap() to require at least one discriminating token
# from the queried name to appear in a candidate's actual name.
_COMMON_VN_NAME_TOKENS: frozenset[str] = frozenset({
    "nguyen", "tran", "le", "pham", "hoang", "huynh",
    "phan", "vu", "vo", "dang", "bui", "do", "ho", "ngo",
    "duong", "ly", "van", "thi",
})


def filter_by_name_overlap(candidates: list, entity_only: str) -> list:
    """For SINGLE person-name queries, discard candidates whose name shares no
    discriminating token with the queried name.

    Prevents KNN position-vector matches from surfacing unrelated people when
    the queried person is absent from the DB (e.g. historical figures, fake names).
    Returns the filtered list, or an empty list when no candidate passes.
    """
    if not candidates or not entity_only:
        return candidates

    all_name_tokens = {t for t in _normalise_text(entity_only).split() if len(t) >= 3}
    discriminating = all_name_tokens - _COMMON_VN_NAME_TOKENS
    check_tokens = discriminating if discriminating else all_name_tokens

    return [
        r for r in candidates
        if check_tokens & set(_normalise_text(r[0]).split())
    ]


def _dedupe_identity_rows(rows: list) -> list:
    """Deduplicate identity tuples while preserving ranking order."""
    deduped = []
    seen = set()
    for row in rows:
        key = (row[0], row[1], row[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _apply_single_list_query_filter(results: list, normalized_query: str) -> list:
    """Apply LIST filtering for one normalized role query/entity phrase."""
    filtered = results

    if _is_role_like_query(normalized_query):
        decorated = []
        for row in filtered:
            features = _candidate_role_features(row, normalized_query)
            decorated.append((row, features))

        role_aligned = [
            (row, feat)
            for row, feat in decorated
            if feat["missing_required"] == 0 and feat["core_overlap_ratio"] >= constants.ROLE_MIN_CORE_OVERLAP
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

    # If query targets a specific ministry, require ALL ministry-discriminating
    # tokens (i.e. non-generic-role tokens) to appear in the candidate's position.
    # This prevents "Thứ trưởng Bộ Quốc phòng" from matching "Thứ trưởng Bộ Ngoại giao".
    query_tokens = [
        t for t in normalized_query.split()
        if t not in constants.LIST_QUERY_FILLER_TOKENS
    ]
    if "bo" in query_tokens:
        ministry_tokens = [
            t for t in query_tokens
            if t not in constants.COMMON_ROLE_TOKENS and len(t) > 1
        ]
        if ministry_tokens:
            org_filtered = [
                row for row in filtered
                if all(t in set(_normalise_text(row[2]).split()) for t in ministry_tokens)
            ]
            if org_filtered:
                filtered = org_filtered

    if _is_role_like_query(normalized_query):
        phrase_filtered = [
            row for row in filtered
            if entity_matches_position(normalized_query, _normalise_text(row[2]))
        ]
        if phrase_filtered:
            filtered = phrase_filtered

    return _dedupe_identity_rows(filtered)


def apply_list_query_filters(results: list, user_input: str, entity_only: str = "") -> list:
    """Apply precise filters for list/count queries to avoid undercounting.

    Use generic role alignment so LIST mode does not rely on title-specific rules.
    """
    if not results:
        return results

    normalized_query = _normalise_text(user_input)
    role_entities = []
    if entity_only:
        role_entities = [
            normalized_part
            for normalized_part in (_normalise_text(part) for part in entity_only.split(","))
            if normalized_part and _is_role_like_query(normalized_part)
        ]

    if len(role_entities) > 1:
        merged = []
        for role_entity in role_entities:
            merged.extend(_apply_single_list_query_filter(results, role_entity))
        if merged:
            return _dedupe_identity_rows(merged)

    filter_query = role_entities[0] if role_entities else normalized_query
    return _apply_single_list_query_filter(results, filter_query)
