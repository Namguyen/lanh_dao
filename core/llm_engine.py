"""Stage 5: LLM answer generation and formatting.

This module handles generating final answers from database and internet evidence,
including direct formatting for high-confidence DB-only results.
"""

import logging
from datetime import datetime

from openai import OpenAI

import config

log = logging.getLogger(__name__)

_llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def format_direct_answer(user_input: str, strict_candidates: list, search_mode: str) -> str:
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
    if len(segments) == 1:
        return f"{name} (sinh năm {year}) hiện giữ chức vụ: {segments[0]}."
    positions = "\n".join(f"  - {s}" for s in segments)
    return f"{name} (sinh năm {year}) hiện giữ các chức vụ:\n{positions}"


def format_multi_person_answer(requested_names: list[str], per_entity: dict) -> str:
    """Format a direct answer for MULTI-person queries without calling the LLM.

    For each requested name, shows the best-matched person's primary position.
    Entries with no DB match are reported as not found.
    """
    lines = []
    for name in requested_names:
        hit = per_entity.get(name)
        if hit is None:
            lines.append(f"- **{name}**: Không tìm thấy trong dữ liệu nội bộ.")
        else:
            hit_name, hit_year, hit_position, hit_score = hit[0], hit[1], hit[2], hit[3]
            positions = "; ".join(s.strip() for s in hit_position.split(";") if s.strip())
            lines.append(f"- **{hit_name}** (sinh {hit_year}): {positions}")
    return "\n".join(lines)


def generate_answer(user_input: str, db_context: str, web_context: str) -> str:
    """Call the LLM to compose a final answer from the provided evidence."""
    now = datetime.now()

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
