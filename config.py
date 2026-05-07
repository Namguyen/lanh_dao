"""Centralized configuration loaded from environment variables with validation."""

import os
import logging
from typing import Final

import dotenv

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Elasticsearch configuration
# ---------------------------------------------------------------------------
ES_HOST: Final[str] = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER: Final[str] = os.getenv("ES_USER", "elastic")
ES_PASS: Final[str] = os.getenv("ES_PASS", "")
ES_INDEX: Final[str] = os.getenv("ES_INDEX", "lanh_dao")

# Validate ES configuration
if not ES_PASS:
    log.warning("ES_PASS is not set. Using empty password (insecure for production).")

if ES_HOST.startswith("https://"):
    log.info("Elasticsearch configured with HTTPS (recommended for production).")
else:
    log.warning("Elasticsearch configured without HTTPS. Consider using HTTPS in production.")

# ---------------------------------------------------------------------------
# DeepSeek LLM configuration
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY: Final[str] = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com"
DEEPSEEK_MODEL: Final[str] = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MODEL_FAST: Final[str] = os.getenv("DEEPSEEK_MODEL_FAST", DEEPSEEK_MODEL)

# Validate LLM configuration
if not DEEPSEEK_API_KEY:
    log.error("DEEPSEEK_API_KEY is not set. LLM features will not work.")

# ---------------------------------------------------------------------------
# Web search (Serper) configuration
# ---------------------------------------------------------------------------
SERPER_API_KEY: Final[str] = os.getenv("SERPER_API_KEY", "")
SERPER_URL: Final[str] = "https://google.serper.dev/search"
SERPER_NEWS_URL: Final[str] = "https://google.serper.dev/news"
SERPER_TIMEOUT_SECONDS: Final[int] = 10
SERPER_MAX_RESULTS: Final[int] = 3

# Validate Serper configuration
if not SERPER_API_KEY:
    log.warning("SERPER_API_KEY is not set. Internet search features will be disabled.")

# ---------------------------------------------------------------------------
# News domain filtering
# ---------------------------------------------------------------------------
OFFICIAL_NEWS_DOMAINS: Final[set[str]] = {
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

DEPRIORITIZED_NEWS_DOMAINS: Final[set[str]] = set()

BLOCKED_NEWS_DOMAINS: Final[set[str]] = {
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

# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------
SINGLE_SEARCH_LIMIT: Final[int] = 12
LIST_SEARCH_LIMIT: Final[int] = 20
SCORE_RELEVANCE_RATIO: Final[float] = 0.85
MIN_SCORE_THRESHOLD: Final[float] = 5.0
LIST_METADATA_CAP: Final[int] = 20
KNN_NUM_CANDIDATES: Final[int] = 20

# ---------------------------------------------------------------------------
# Internet trigger keywords
# ---------------------------------------------------------------------------
NEWS_KEYWORDS: Final[list[str]] = [
    "chỉ đạo", "hoạt động", "mới nhất", "mới", "gần đây",
    "tin tức", "phát biểu", "gặp gỡ", "tiếp đón", "thăm", "ký kết",
]

# ---------------------------------------------------------------------------
# Role matching configuration
# ---------------------------------------------------------------------------
SPECIFIC_ROLE_HINTS: Final[set[str]] = {
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
    "uy vien",
}

GENERIC_QUERY_FILLER_TOKENS: Final[set[str]] = {
    "la", "ai", "co", "gi", "moi", "nhat", "gan", "day", "nao",
    "khong", "vay", "the", "sao", "cho", "toi", "xin", "hoi", "tim", "kiem",
}

GENERIC_ENTITY_PHRASES: Final[set[str]] = {
    "lanh dao",
    "lanh đao",
    "can bo",
    "nhan su",
    "nguoi dung dau",
    "sep",
}

ROLE_MODIFIER_PHRASES: Final[set[str]] = {
    "pho",
    "thuong truc",
    "quyen",
    "van phong",
    "tro ly",
}

ROLE_QUERY_FILLER_TOKENS: Final[set[str]] = GENERIC_QUERY_FILLER_TOKENS | {
    "co", "the", "cho", "biet",
    "dang", "lam", "hien", "nay", "giu", "chuc", "dung", "dau", "duoc",
}

# Shared lexical/token constants (avoid hardcoded inline sets across modules)
COMMON_ROLE_TOKENS: Final[frozenset[str]] = frozenset({
    "thu", "truong", "bo", "pho", "uy", "vien",
    "giam", "doc", "chu", "tich", "tong", "bi", "ban",
})

LIST_QUERY_FILLER_TOKENS: Final[frozenset[str]] = frozenset({
    "lanh", "dao", "va", "cac", "nhung", "co", "bao", "nhieu", "la", "ai",
})

COMMON_GENERIC_ENTITY_TOKENS: Final[frozenset[str]] = frozenset({
    "lanh", "dao", "can", "bo", "nhan", "su", "sep", "nguoi", "dung", "dau",
})

PERSON_NAME_LIGHT_STOPWORDS: Final[frozenset[str]] = frozenset({"thi", "van"})

ROLE_STRUCTURAL_GAP_TOKENS: Final[frozenset[str]] = frozenset({"bo", "uy", "ban"})

ROLE_MISSING_MODIFIER_PENALTY: Final[float] = 25.0
ROLE_EXTRA_MODIFIER_PENALTY: Final[float] = 8.0
ROLE_CORE_OVERLAP_WEIGHT: Final[float] = 5.0
ROLE_MIN_CORE_OVERLAP: Final[float] = 0.6


def validate_config() -> bool:
    """Validate critical configuration values.
    
    Returns True if all required configs are present, False otherwise.
    Logs warnings for optional missing configs.
    """
    errors = []
    warnings = []
    
    if not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY is required but not set")
    
    if not ES_PASS:
        warnings.append("ES_PASS is empty (insecure for production)")
    
    if not SERPER_API_KEY:
        warnings.append("SERPER_API_KEY is not set (internet search disabled)")
    
    for warning in warnings:
        log.warning(warning)
    
    if errors:
        for error in errors:
            log.error(error)
        return False
    
    return True