"""Centralized configuration loaded from environment variables."""

import os
import logging

import dotenv

dotenv.load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Elasticsearch
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "ES_PASS")
ES_INDEX = os.getenv("ES_INDEX", "lanh_dao")

# DeepSeek LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Main model for answer generation. Override via .env if DeepSeek changes the name.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
# Faster/cheaper model for lightweight tasks like intent classification.
# Set to the same value as DEEPSEEK_MODEL if no separate fast model is available.
DEEPSEEK_MODEL_FAST = os.getenv("DEEPSEEK_MODEL_FAST", DEEPSEEK_MODEL)

# Web search (Serper)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"
SERPER_TIMEOUT_SECONDS = 10
SERPER_MAX_RESULTS = 3

# Prioritize official/state and mainstream Vietnamese news sources.
OFFICIAL_NEWS_DOMAINS = {
    "chinhphu.vn",
    "baochinhphu.vn",
    "nhandan.vn",
    "qdnd.vn",
    "qdnd.com.vn",
    "vietnamplus.vn",
    "vnanet.vn",
    "moha.gov.vn",
    "mod.gov.vn",
    "mofa.gov.vn",
    "quochoi.vn",
    "dangcongsan.vn",
    "vtv.vn",
    "vov.vn",
    "tuoitre.vn",
    "thanhnien.vn",
    "vnexpress.net",
}

# Lower-trust sources for factual leadership activity queries.
DEPRIORITIZED_NEWS_DOMAINS = {
}

# Completely blocked: social media, blogs, tabloid / entertainment aggregators.
BLOCKED_NEWS_DOMAINS = {
    # Social media
    "facebook.com",
    "m.facebook.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "threads.net",
    # Blog platforms
    "blogspot.com",
    "wordpress.com",
    # Entertainment / tabloid aggregators
    "kenh14.vn",
    "eva.vn",
    "afamily.vn",
    "gamek.vn",
    "yan.vn",
    "docbao.vn",
    "tinmoi.vn",
    "soha.vn",
    "doisongphapluat.com",
    "nguoiduatin.vn",
    "phunutoday.vn",
    "webtretho.com",
    "2sao.vn",
    "nguoiquansat.vn",
}

# Search parameters
SINGLE_SEARCH_LIMIT = 12
LIST_SEARCH_LIMIT = 20
SCORE_RELEVANCE_RATIO = 0.85
MIN_SCORE_THRESHOLD = 5.0
LIST_METADATA_CAP = 20
KNN_NUM_CANDIDATES = 20

# Internet trigger keywords
NEWS_KEYWORDS = [
    "chỉ đạo", "hoạt động", "mới nhất", "mới", "gần đây",
    "tin tức", "phát biểu", "gặp gỡ", "tiếp đón", "thăm", "ký kết",
]

SPECIFIC_ROLE_HINTS = {
    "bo truong",
    "thu truong",
    "thu tuong",
    "pho thu tuong",
    "tong bi thu",
    "bi thu",
    "chu tich",
    "pho chu tich",
    "tong tham muu truong",
    "chu nhiem",
    "bo quoc phong",
    "bo cong an",
    "bo ngoai giao",
    "quoc hoi",
    "chinh phu",
}

GENERIC_QUERY_FILLER_TOKENS = {
    "la",
    "ai",
    "co",
    "gi",
    "moi",
    "nhat",
    "gan",
    "day",
    "nao",
    "khong",
    "vay",
    "the",
    "sao",
    "cho",
    "toi",
    "xin",
    "hoi",
    "tim",
    "kiem",
}


GENERIC_ENTITY_PHRASES = {
    "lanh dao",
    "lanh đao",
    "can bo",
    "nhan su",
    "nguoi dung dau",
    "sep",
}

# Generic role disambiguation (no title-specific hardcode)
ROLE_MODIFIER_PHRASES = {
    "pho",
    "thuong truc",
    "quyen",
    "van phong",
    "tro ly",
}

ROLE_QUERY_FILLER_TOKENS = GENERIC_QUERY_FILLER_TOKENS | {
    "co",
    "the",
    "cho",
    "biet",
    # Vietnamese verb particles commonly appearing in role questions
    # e.g. "ai đang làm thủ tướng", "hiện nay ai giữ chức"
    "dang",   # đang (currently)
    "lam",    # làm (work as)
    "hien",   # hiện (currently)
    "nay",    # nay (now)
    "giu",    # giữ (hold/occupy a position)
    "chuc",   # chức (generic word for position)
    "dung",   # đứng (head/stand)
    "dau",    # đầu (first/head)
    "duoc",   # được (be/receive)
}

# Rerank/filter weights for role-like queries
ROLE_MISSING_MODIFIER_PENALTY = 25.0
ROLE_EXTRA_MODIFIER_PENALTY = 8.0
ROLE_CORE_OVERLAP_WEIGHT = 5.0
ROLE_MIN_CORE_OVERLAP = 0.6