"""Stage 4: Internet search and news synthesis.

This module handles optional internet enrichment for queries that need
up-to-date information about politicians' recent activities.
"""

import logging
import re
from datetime import datetime

from openai import OpenAI

from ai_service import get_internet_info
import config

log = logging.getLogger(__name__)

_llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def should_search_internet(intent: str, user_input: str) -> bool:
    """Determine whether an internet search is warranted."""
    if intent == "INTERNET":
        return True
    return any(kw in user_input.lower() for kw in config.NEWS_KEYWORDS)


def extract_web_sources(web_context: str) -> list[dict]:
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


def generate_evidence_first_news_answer(
    person_name: str, position: str, web_context: str
) -> str:
    """Use the LLM to synthesize a polished, narrative news digest."""
    web_sources = extract_web_sources(web_context)
    if not web_sources:
        return f"Chưa ghi nhận thông tin cập nhật đủ cụ thể về {person_name} trong thời gian gần đây."

    now = datetime.now()

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
