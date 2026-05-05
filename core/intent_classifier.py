"""Stage 1: Intent classification and query rewriting.

This module handles user query analysis using LLM to determine:
- intent: DATABASE vs INTERNET
- rewritten_query: cleaned, normalized query
- search_mode: SINGLE (one person) vs LIST (multiple people)
- entity_only: extracted entity names for DB search
"""

import functools
import json
import logging

from openai import OpenAI

import config

log = logging.getLogger(__name__)

_llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def analyze_query_intent(user_query: str) -> tuple[str, str, str, str]:
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


def is_ambiguous_leadership_query(user_input: str, entity_only: str) -> bool:
    """Return True when query is too generic to safely pick a single person."""
    import unicodedata

    def _normalise_text(text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return " ".join(text.split())

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
