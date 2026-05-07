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
from .text_utils import normalize_text

log = logging.getLogger(__name__)

_llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def analyze_query_intent(user_query: str) -> tuple[str, str, str, str]:
    """Classify intent, rewrite the query, and extract target entities via LLM.

    Returns (intent, rewritten_query, search_mode, entity_only).
    Cached so repeated identical queries skip the LLM call entirely.
    """
    prompt = f"""Bạn là Bộ xử lý Truy vấn Trung tâm. Phân tích ý định và viết lại câu hỏi.

QUY TẮC CHẾ ĐỘ TÌM KIẾM (search_mode):
- "SINGLE": Khi tra cứu một NGƯỜI CỤ THỂ hoặc một CHỨC VỤ CỤ THỂ (VD: "đức thắng viettel", "ai là bộ trưởng bqp").
- "LIST": Khi hỏi SỐ LƯỢNG, DANH SÁCH theo nhóm/chức vụ chung (VD: "có bao nhiêu thứ trưởng bqp", "liệt kê các phó tổng").
- "MULTI": Khi hỏi về NHIỀU NGƯỜI CỤ THỂ được nêu tên rõ ràng trong câu hỏi (VD: "cho biết vị trí của Tô Lâm, Lê Minh Hưng, Phan Văn Giang", "Nguyễn Duy Ngọc và Tô Lâm đang giữ chức gì"). Chỉ dùng khi có ít nhất 2 tên người riêng biệt.

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
- Giữ TÊN NGƯỜI hoặc CHỨC VỤ ĐẦY ĐỦ bao gồm tên bộ/cơ quan/địa phương. KHÔNG được rút gọn.
- Ví dụ đúng: "Thứ trưởng Bộ Ngoại giao" (KHÔNG viết tắt thành "Thứ trưởng").
- Ví dụ đúng: "Bộ trưởng Bộ Công an" (KHÔNG viết tắt thành "Bộ trưởng").
- Chỉ xóa các từ hành động hoặc câu hỏi thừa (chỉ đạo, hoạt động, làm gì, mới nhất, gần đây, là ai, gồm những ai...).
- PHẢI dịch synonym địa danh:
    "Thủ đô" -> "Hà Nội"
    "thành phố Bác" -> "Hồ Chí Minh"
    "thành phố lớn nhất" -> "Hồ Chí Minh"
    "BQP" -> "Bộ Quốc phòng"
    "BCT" -> "Bộ Chính trị"
- Nếu có NHIỀU chức vụ, tách bằng dấu phẩy. VD: "tổng bí thư, thủ tướng, bộ trưởng công an"
- Nếu search_mode là "MULTI", entity_only phải là danh sách TÊN NGƯỜI cách nhau bằng dấu phẩy, viết đúng chính tả. VD: "Lê Minh Hưng, Tô Lâm, Nguyễn Duy Ngọc, Phan Văn Giang"
- **QUAN TRỌNG - CHỨC VỤ + TÊN NGƯỜI**: Khi truy vấn gồm cả CHỨC VỤ lẫn TÊN NGƯỜI cụ thể (VD: "tổng bí thư trần cẩm tú", "bộ trưởng công an tô lâm"), entity_only CHỈ được chứa TÊN NGƯỜI, KHÔNG gộp chức vụ vào. Chức vụ chỉ là ngữ cảnh tìm kiếm, không phải thực thể.
  - Ví dụ: "tổng bí thư trần cẩm tú" → entity_only = "Trần Cẩm Tú"
  - Ví dụ: "chủ tịch nước tô lâm" → entity_only = "Tô Lâm"
  - Ví dụ: "bộ trưởng quốc phòng phan văn giang" → entity_only = "Phan Văn Giang"
  - Ngược lại: "tổng bí thư là ai" → entity_only = "Tổng Bí thư" (không có tên người cụ thể)
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
    normalized_query = normalize_text(user_input)
    normalized_entity = normalize_text(str(entity_only or ""))

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
        if t not in config.COMMON_GENERIC_ENTITY_TOKENS
    ]
    generic_only_question = has_generic and not non_generic_tokens

    return (
        (has_generic and not has_specific and entity_is_generic)
        or (has_generic and not has_specific and entity_matches_whole_query)
        or broad_leadership_query
        or generic_only_question
    )
