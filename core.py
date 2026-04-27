import datetime
import functools
import json
import logging
import re
import unicodedata

from openai import OpenAI

from ai_service import get_internet_info
import config

log = logging.getLogger(__name__)

_llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


# Stage 1: Intent analysis

@functools.lru_cache(maxsize=512)
def analyze_query_intent(user_query):
    """Classify intent, rewrite the query, and extract target entities via LLM.

    Returns (intent, rewritten_query, search_mode, entity_only).
    Cached so repeated identical queries skip the LLM call entirely.
    """
    prompt = f"""Bạn là Bộ xử lý Truy vấn Trung tâm. Phân tích ý định và viết lại câu hỏi.

QUY TẮC CHẾ ĐỘ TÌM KIẾM (search_mode):
- "SINGLE": Khi tra cứu một NGƯỜI CỤ THỂ (VD: "đức thắng viettel", "ai là bộ trưởng bqp").
- "LIST": Khi hỏi SỐ LƯỢNG, DANH SÁCH, tập thể (VD: "có bao nhiêu thứ trưởng bqp", "liệt kê các phó tổng").

QUY TẮC Ý ĐỊNH (intent):
- "DATABASE": Hỏi người, chức vụ nội bộ.
- "INTERNET": Hỏi sự kiện thời sự, người từ trần.

QUY TẮC VIẾT LẠI (rewritten_query):
1. Xóa đại từ xưng hô (VD: "sếp", "lãnh đạo")
2. Dịch từ viết tắt (BQP -> Bộ Quốc phòng).
3. Tự sửa lỗi chính tả.
4. Xóa SẠCH các từ để hỏi thừa thãi. CHỈ GIỮ LẠI danh từ cốt lõi.
   - Ví dụ sai: "Tổng Bí thư là ai"
   - Ví dụ đúng: "Tổng Bí thư"

QUY TẮC entity_only:
- Chỉ giữ tên người hoặc chức vụ NGẮN NHẤT để tìm trong DB.
- Xóa HOÀN TOÀN các từ hành động (chỉ đạo, hoạt động, làm gì, mới nhất, gần đây...).
- PHẢI dịch synonym địa danh:
    "Thủ đô" -> "Hà Nội"
    "thành phố Bác" -> "Hồ Chí Minh"
    "thành phố lớn nhất" -> "Hồ Chí Minh"
    "BQP" -> "Bộ Quốc phòng"
    "BCT" -> "Bộ Chính trị"
- Nếu có NHIỀU chức vụ, tách bằng dấu phẩy. VD: "tổng bí thư, thủ tướng, bộ trưởng công an"
- Ví dụ: "Thủ tướng chỉ đạo mới" -> "Thủ tướng"
- Ví dụ: "Bí thư thủ đô là ai" -> "Bí thư Hà Nội"
- Ví dụ: "đức thắng viettel làm gì" -> "Đức Thắng Viettel"
- LUÔN trả về string, tuyệt đối không trả về boolean (true/false).

Câu hỏi: "{user_query}"
Trả về JSON gồm 4 trường: intent, rewritten_query, search_mode, entity_only"""

    try:
        response = _llm.chat.completions.create(
            model=config.DEEPSEEK_MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        intent = str(result.get("intent", "DATABASE"))
        rewritten = str(result.get("rewritten_query", user_query))
        search_mode = str(result.get("search_mode", "SINGLE"))
        entity_only = result.get("entity_only", user_query)

        # Guard against LLM returning a boolean or empty value
        if isinstance(entity_only, bool) or not entity_only:
            entity_only = user_query
        else:
            entity_only = str(entity_only)

        return intent, rewritten, search_mode, entity_only

    except Exception as exc:
        log.error("Intent analysis failed: %s", exc)
        return "DATABASE", user_query, "SINGLE", user_query


# Stage 2: Database retrieval

def _retrieve_candidates(db, entities, search_mode):
    """Search the database for each entity and return merged, score-sorted results.

    Results are sorted by score descending so the global best is always at index 0,
    regardless of which entity produced it.
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


def _normalise_text(text):
    """Normalize Vietnamese text for robust keyword rule checks."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def _extract_modifier_phrases(normalized_text):
    """Return configured modifier phrases appearing in the given normalized text."""
    return {
        phrase
        for phrase in config.ROLE_MODIFIER_PHRASES
        if phrase in normalized_text
    }


def _extract_core_query_tokens(normalized_query):
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


def _is_role_like_query(normalized_query):
    """Return True when query appears to target a role/title rather than a person name."""
    return any(hint in normalized_query for hint in config.SPECIFIC_ROLE_HINTS)


def _candidate_role_features(candidate, normalized_query):
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


def _rerank_by_generic_role_rules(results, user_input):
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


def _filter_single_role_candidates(candidates, normalized_query):
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


def _is_ambiguous_leadership_query(user_input, entity_only):
    """Return True when query is too generic to safely pick a single person."""
    normalized_query = _normalise_text(user_input)
    normalized_entity = _normalise_text(str(entity_only or ""))

    has_generic = any(phrase in normalized_query for phrase in config.GENERIC_ENTITY_PHRASES)
    has_specific = any(hint in normalized_query for hint in config.SPECIFIC_ROLE_HINTS)
    entity_is_generic = (
        normalized_entity in config.GENERIC_ENTITY_PHRASES
        or normalized_entity.replace(" ", "") in {"lanhdao", "canbo", "nhansu", "sep"}
    )

    broad_leadership_query = any(
        phrase in normalized_query for phrase in {"lanh dao", "can bo", "nhan su"}
    ) and not has_specific

    entity_matches_whole_query = normalized_entity == normalized_query

    query_tokens = normalized_query.split()
    meaningful_tokens = [t for t in query_tokens if t not in config.GENERIC_QUERY_FILLER_TOKENS]
    non_generic_tokens = [
        t
        for t in meaningful_tokens
        if t not in {"lanh", "dao", "can", "bo", "nhan", "su", "sep", "nguoi", "dung", "dau"}
    ]
    generic_only_question = has_generic and not non_generic_tokens

    return (
        (has_generic and not has_specific and entity_is_generic)
        or (has_generic and not has_specific and entity_matches_whole_query)
        or broad_leadership_query
        or generic_only_question
    )

def _extract_web_sources(web_context):
    """Parse formatted web context into structured source entries."""
    if not web_context or "Tin 1" not in web_context:
        return []

    sources = []
    for block in web_context.split("\n\n"):
        if not block.strip():
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        title = ""
        snippet = ""
        date = "Gần đây"
        source = "không rõ nguồn"
        link = ""

        if lines:
            match = re.match(r"Tin\s+\d+\s*\((.*?)\)", lines[0])
            if match:
                date = match.group(1).strip() or "Gần đây"

        for line in lines:
            if line.startswith("Tiêu đề:"):
                title = line.replace("Tiêu đề:", "", 1).strip()
            elif line.startswith("Nội dung:"):
                snippet = line.replace("Nội dung:", "", 1).strip()
            elif line.startswith("Nguồn:"):
                source = line.replace("Nguồn:", "", 1).strip() or "không rõ nguồn"
            elif line.startswith("Link:"):
                link = line.replace("Link:", "", 1).strip()

        if title:
            sources.append(
                {
                    "title": title,
                    "snippet": snippet or "Không có mô tả ngắn từ nguồn.",
                    "date": date,
                    "source": source,
                    "link": link,
                }
            )

    return sources


def _build_db_candidates(candidates):
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


def _estimate_confidence(highest_score, answer_mode, has_web_sources):
    """Estimate confidence from retrieval score and evidence availability."""
    if answer_mode == "no_match":
        return 0.0

    base = min(1.0, highest_score / max(config.MIN_SCORE_THRESHOLD * 2, 1))

    if answer_mode == "database_only":
        base *= 0.9

    if answer_mode == "db_plus_web" and not has_web_sources:
        base *= 0.75

    return round(max(0.05, min(base, 1.0)), 2)


def _rerank_with_query_rules(results, user_input):
    """Apply conservative query-aware reranking rules.

    This keeps the existing ES scoring intact and only adjusts ordering via
    generic role modifiers (e.g. pho/thuong truc/van phong), not hardcoded titles.
    """
    return _rerank_by_generic_role_rules(results, user_input)


def _apply_list_query_filters(results, user_input):
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


# Stage 3: Score filtering

def _filter_and_build(results, search_mode, user_input):
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

    results = _rerank_with_query_rules(results, user_input)

    highest_score = results[0][3]
    if highest_score <= config.MIN_SCORE_THRESHOLD:
        log.info("Top score %.2f below threshold; returning empty metadata", highest_score)
        return None, no_match_msg, [] if search_mode == "LIST" else empty_single, [], highest_score

    if search_mode == "LIST":
        # LIST queries are intended for counting/listing groups, so avoid
        # aggressive top-ratio filtering that can hide valid entries.
        strictly_valid = [r for r in results if r[3] >= config.MIN_SCORE_THRESHOLD]
        strictly_valid = _apply_list_query_filters(strictly_valid, user_input)
        if not strictly_valid:
            strictly_valid = [
                r for r in results if r[3] >= highest_score * config.SCORE_RELEVANCE_RATIO
            ]
    else:
        normalized_query = _normalise_text(user_input)
        strictly_valid = [
            r for r in results if r[3] >= highest_score * config.SCORE_RELEVANCE_RATIO
        ]
        strictly_valid = _filter_single_role_candidates(strictly_valid, normalized_query)

    if not strictly_valid:
        return None, no_match_msg, [] if search_mode == "LIST" else empty_single, [], highest_score

    best_person = strictly_valid[0]
    db_context = "\n".join(f"- {r[0]} ({r[1]}): {r[2]}" for r in strictly_valid)

    # FIX: build metadata according to actual search_mode to avoid type mismatch
    if search_mode == "LIST":
        metadata = [
            {"name": r[0], "nam_sinh": r[1], "chuc_vu": r[2]}
            for r in strictly_valid[:config.LIST_METADATA_CAP]
        ]
    else:
        metadata = {"name": best_person[0], "nam_sinh": best_person[1], "chuc_vu": best_person[2]}

    return best_person, db_context, metadata, strictly_valid, highest_score


# Stage 4: Internet enrichment

def _should_search_internet(intent, user_input):
    """Determine whether an internet search is warranted."""
    if intent == "INTERNET":
        return True
    return any(kw in user_input.lower() for kw in config.NEWS_KEYWORDS)


def _generate_evidence_first_news_answer(person_name, position, web_context):
    """Use the LLM to synthesize a polished, narrative news digest."""
    web_sources = _extract_web_sources(web_context)
    if not web_sources:
        return f"Chưa ghi nhận thông tin cập nhật đủ cụ thể về {person_name} trong thời gian gần đây."

    now = datetime.datetime.now()

    # Build structured source list for the prompt
    source_block = ""
    for idx, item in enumerate(web_sources, start=1):
        source_block += (
            f"Tin {idx}:\n"
            f"  Tiêu đề: {item['title']}\n"
            f"  Nội dung: {item.get('snippet', '')}\n"
            f"  Nguồn: {item.get('source', 'không rõ')}\n"
            f"  Link: {item.get('link', '')}\n\n"
        )

    user_prompt = f"""[NHÂN SỰ]: {person_name} — {position}

[TIN TỨC TỪ BÁO CHÍ]:
{source_block}

[YÊU CẦU]:
Viết một bản tổng hợp tin tức mới nhất về {person_name} dựa HOÀN TOÀN vào các tin tức ở trên.

[QUY TẮC FORMAT BẮT BUỘC]:
1. Dòng đầu: "Đây là những tin tức mới nhất về [chức vụ chính] [tên]:"
2. Mỗi tin tức là MỘT đoạn văn ngắn (2-4 câu), bắt đầu bằng tiêu đề in đậm.
3. Cuối mỗi đoạn, ghi nguồn báo dưới dạng Markdown link: [tên nguồn](URL)
4. Cuối cùng, viết 1-2 câu tóm tắt tổng quan.
5. TUYỆT ĐỐI không bịa thêm sự kiện, ngày tháng, hay chi tiết nào không có trong tin tức.
6. Nếu snippet trống hoặc thiếu chi tiết, chỉ tóm tắt từ tiêu đề, KHÔNG sáng tác thêm.
7. Giọng văn: chuyên nghiệp, mạch lạc, đọc như một bản tin tổng hợp.
8. KHÔNG viết "Dựa trên dữ liệu", "Theo thông tin được cung cấp" hay bất kỳ cụm rào đón nào.
9. KHÔNG dùng emoji, icon hoặc ký hiệu trang trí (ví dụ: ✅ 🤝 ✈️ • ›).

VÍ DỤ FORMAT:
Đây là những tin tức mới nhất về Tổng Bí thư, Chủ tịch nước Tô Lâm:

**Tái đắc cử Chủ tịch nước nhiệm kỳ mới** Ngày 7/4/2026, Quốc hội thông qua Nghị quyết bầu Tổng Bí thư Tô Lâm giữ chức Chủ tịch nước nhiệm kỳ 2026–2031 với 100% đại biểu có mặt đồng ý. [VnExpress](https://vnexpress.net/...)

**Thăm cấp nhà nước tới Trung Quốc** Ngày 17/4/2026, Tổng Bí thư, Chủ tịch nước Tô Lâm đã tham quan triển lãm về AI tại Nam Ninh trong khuôn khổ chuyến thăm cấp nhà nước tới Trung Quốc. [VnExpress](https://vnexpress.net/...)

Tóm lại, ông Tô Lâm hiện đang tích cực hoạt động đối ngoại lẫn đối nội trong thời gian gần đây."""

    system_prompt = (
        f"Bạn là Trợ lý tổng hợp tin tức chính trị Việt Nam. "
        f"Hôm nay là tháng {now.month} năm {now.year}.\n"
        f"CHỈ dùng tin tức được cung cấp, không bịa thêm bất kỳ thông tin nào."
    )

    try:
        response = _llm.chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as exc:
        log.error("News synthesis LLM call failed: %s", exc)
        return f"Chưa ghi nhận thông tin cập nhật đủ cụ thể về {person_name} trong thời gian gần đây."


# Stage 5: Answer generation

def _format_direct_answer(user_input, strict_candidates, search_mode):
    """Format a direct answer from DB data without calling the LLM.

    Used for high-confidence DATABASE_ONLY queries with no news keywords,
    saving one LLM round-trip (~1-3 seconds).
    """
    if search_mode == "LIST":
        lines = [f"- {r[0]} ({r[1]}): {r[2]}" for r in strict_candidates]
        return f"Danh sách tìm được ({len(lines)} người):\n" + "\n".join(lines)

    person = strict_candidates[0]
    name, year, position = person[0], person[1], person[2]
    segments = [s.strip() for s in position.split(";") if s.strip()]
    query_tokens = _extract_core_query_tokens(_normalise_text(user_input))

    # Prefer the role segment that best overlaps with query intent
    # (e.g. "thủ tướng" over generic labels like "ủy viên").
    best_segment = segments[0] if segments else position
    best_score = -1
    for seg in segments:
        seg_norm = _normalise_text(seg)
        score = sum(1 for t in query_tokens if t in seg_norm)
        if score > best_score:
            best_score = score
            best_segment = seg

    primary_position = best_segment
    return f"{name} (sinh năm {year}) hiện giữ chức vụ: {primary_position}."


def _generate_answer(user_input, db_context, web_context):
    """Call the LLM to compose a final answer from the provided evidence."""
    now = datetime.datetime.now()

    internet_section = ""
    if web_context and "Tin 1" in web_context:
        internet_section = f"\n[TIN TỨC INTERNET - CẬP NHẬT MỚI]:\n{web_context}"

    user_prompt = f"""
[DỮ LIỆU NỘI BỘ - CHÂN LÝ]:
{db_context}
{internet_section}

[YÊU CẦU]:
Dựa trên dữ liệu, hãy trả lời câu hỏi: "{user_input}"

[QUY TẮC PHẢI TUÂN THỦ]:
1. CHỈ TRẢ LỜI về nhân sự được xác định trong [DỮ LIỆU NỘI BỘ].
2. Nếu có [TIN TỨC INTERNET], hãy tổng hợp các hoạt động mới nhất. TUYỆT ĐỐI KHÔNG TỰ BỊA RA NGÀY THÁNG hay sự kiện nếu không có trong tin tức.
3. Nếu KHÔNG CÓ thông tin internet, tuyệt đối KHÔNG nhắc đến các từ "internet", "không có dữ liệu web".
4. TUYỆT ĐỐI không thêm biệt danh, mô tả ngoài dữ liệu (VD: không được viết "thành phố mang tên Bác Hồ").
5. Trình bày dõng dạc, chuyên nghiệp.
6. Nếu [TIN TỨC INTERNET] có thông tin về các cuộc tiếp đón, chuyến thăm, ưu tiên trình bày rõ: Tên đối tác, thời gian, địa điểm, nội dung chính.
7. TUYỆT ĐỐI KHÔNG sử dụng các cụm từ rào trước đón sau như "Dựa trên dữ liệu nội bộ", "Theo thông tin được cung cấp", "Theo dữ liệu". TRẢ LỜI TRỰC TIẾP luôn vào trọng tâm câu hỏi.
"""

    system_prompt = (
        f"Bạn là Trợ lý tra cứu nhân sự cấp cao. "
        f"Hôm nay là tháng {now.month} năm {now.year}.\n"
        f"CHỈ dùng dữ liệu được cung cấp, không bịa thêm bất kỳ thông tin nào."
    )

    response = _llm.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# Pipeline entry point

def process_query(user_input, db):
    """Execute the full query pipeline shared by both CLI and API.

    Stages: intent analysis -> retrieval -> filtering -> internet -> answer.
    """
    # Stage 1: Intent analysis
    intent, _rewritten, search_mode, entity_only = analyze_query_intent(user_input)
    log.info("Intent=%s  mode=%s  entity='%s'", intent, search_mode, entity_only)

    if _is_ambiguous_leadership_query(user_input, entity_only):
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
            },
            "answer": clarify_answer,
        }

    # Stage 2: Database retrieval
    entities = [e.strip() for e in entity_only.split(",")]
    all_results, retrieval_trace = _retrieve_candidates(db, entities, search_mode)

    # Stage 3: Score filtering and metadata
    best_person, db_context, metadata, strict_candidates, highest_score = _filter_and_build(
        all_results, search_mode, user_input
    )

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
            },
            "answer": answer,
        }

    # Stage 4: Optional internet enrichment
    # LIST queries (counting/listing groups) don't benefit from per-person news digests
    web_context = ""
    if search_mode != "LIST" and _should_search_internet(intent, user_input):
        search_query = best_person[0]
        log.info("Internet search: %s", search_query)
        web_context = get_internet_info(search_query, person_name=best_person[0])

    web_sources = _extract_web_sources(web_context)
    answer_mode = "db_plus_web" if web_sources else "database_only"
    confidence = _estimate_confidence(highest_score, answer_mode, bool(web_sources))

    # Stage 5: Answer generation
    is_news_query = any(kw in user_input.lower() for kw in config.NEWS_KEYWORDS)

    if is_news_query:
        position = best_person[2].split(";")[0].strip()
        answer = _generate_evidence_first_news_answer(best_person[0], position, web_context)
    elif answer_mode == "database_only" and confidence >= 0.7:
        # High-confidence pure DB result → skip LLM, format directly (~1-3s saved)
        answer = _format_direct_answer(user_input, strict_candidates, search_mode)
    else:
        answer = _generate_answer(user_input, db_context, web_context)

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
        },
        "answer": answer,
    }