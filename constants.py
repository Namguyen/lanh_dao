"""Application logic constants — hardcoded, not injected from environment."""

from typing import Final

# News domain filtering
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

# Search parameters
SINGLE_SEARCH_LIMIT: Final[int] = 12
LIST_SEARCH_LIMIT: Final[int] = 20
SCORE_RELEVANCE_RATIO: Final[float] = 0.85
MIN_SCORE_THRESHOLD: Final[float] = 5.0
LIST_METADATA_CAP: Final[int] = 20
KNN_NUM_CANDIDATES: Final[int] = 20


# Internet trigger keywords

NEWS_KEYWORDS: Final[list[str]] = [
    "chỉ đạo", "hoạt động", "mới nhất", "mới", "gần đây",
    "tin tức", "phát biểu", "gặp gỡ", "tiếp đón", "thăm", "ký kết",
]

# Role matching configuration
SPECIFIC_ROLE_HINTS: Final[frozenset[str]] = frozenset({
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
})

GENERIC_QUERY_FILLER_TOKENS: Final[frozenset[str]] = frozenset({
    "la", "ai", "co", "gi", "moi", "nhat", "gan", "day", "nao",
    "khong", "vay", "the", "sao", "cho", "toi", "xin", "hoi", "tim", "kiem",
})

GENERIC_ENTITY_PHRASES: Final[frozenset[str]] = frozenset({
    "lanh dao",
    "can bo",
    "nhan su",
    "nguoi dung dau",
    "sep",
})

ROLE_MODIFIER_PHRASES: Final[frozenset[str]] = frozenset({
    "pho",
    "thuong truc",
    "quyen",
    "van phong",
    "tro ly",
})

ROLE_QUERY_FILLER_TOKENS: Final[frozenset[str]] = GENERIC_QUERY_FILLER_TOKENS | frozenset({
    "biet",
    "dang", "lam", "hien", "nay", "giu", "chuc", "dung", "dau", "duoc",
})

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

# Normalized (ASCII) party/hierarchy labels that are too generic to use as a person's primary role
GENERIC_PARTY_LABELS: Final[frozenset[str]] = frozenset({
    "uy vien bo chinh tri",
    "uy vien trung uong dang",
    "uy vien du khuyet trung uong dang",
})

ROLE_STRUCTURAL_GAP_TOKENS: Final[frozenset[str]] = frozenset({"bo", "uy", "ban"})

ROLE_MISSING_MODIFIER_PENALTY: Final[float] = 25.0
ROLE_EXTRA_MODIFIER_PENALTY: Final[float] = 8.0
ROLE_CORE_OVERLAP_WEIGHT: Final[float] = 5.0
ROLE_MIN_CORE_OVERLAP: Final[float] = 0.6
