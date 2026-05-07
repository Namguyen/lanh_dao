"""Web search service using the Serper API for real-time news retrieval."""

import json
import logging
import re
from urllib.parse import urlparse

import requests

import config
from core.text_utils import normalize_text

log = logging.getLogger(__name__)


def _normalise_text(text):
    """Normalize Vietnamese text for robust matching."""
    return normalize_text(text)


def _extract_name_tokens(person_name):
    tokens = [t for t in _normalise_text(person_name).split() if len(t) > 1]
    return [t for t in tokens if t not in config.PERSON_NAME_LIGHT_STOPWORDS]


def _is_result_relevant_to_person(result, person_name):
    """Return True if title/snippet likely refers to the target person."""
    if not person_name:
        return True

    title_norm = _normalise_text(result.get("title", ""))
    combined = _normalise_text(f"{result.get('title', '')} {result.get('snippet', '')}")
    name_tokens = _extract_name_tokens(person_name)
    if not name_tokens:
        return True

    title_matched = sum(1 for token in name_tokens if token in title_norm)
    combined_matched = sum(1 for token in name_tokens if token in combined)

    # Require at least one token in title and stronger support from title+snippet.
    required = min(2, len(name_tokens))
    return title_matched >= 1 and combined_matched >= required


def _has_concrete_time_hint(result):
    """Check whether a result contains a concrete date/time signal."""
    text = f"{result.get('date', '')} {result.get('title', '')} {result.get('snippet', '')}"
    norm = _normalise_text(text)
    if result.get("date") and _normalise_text(result.get("date", "")) not in {"gan day", "moi day", ""}:
        return True
    return bool(re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4}|ngay\s+\d{1,2}|thang\s+\d{1,2}|nam\s+\d{4})\b", norm))


def _extract_domain(url):
    try:
        netloc = urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _is_official_domain(domain):
    return any(domain == d or domain.endswith(f".{d}") for d in config.OFFICIAL_NEWS_DOMAINS)


def _is_deprioritized_domain(domain):
    return any(domain == d or domain.endswith(f".{d}") for d in config.DEPRIORITIZED_NEWS_DOMAINS)


def _is_blocked_domain(domain):
    return any(domain == d or domain.endswith(f".{d}") for d in config.BLOCKED_NEWS_DOMAINS)


def _result_priority_tuple(result):
    domain = _extract_domain(result.get("link", ""))
    return (
        1 if _is_official_domain(domain) else 0,
        1 if _has_concrete_time_hint(result) else 0,
        0 if _is_deprioritized_domain(domain) else 1,
    )


# --------------- Title-based deduplication ---------------

_TITLE_STOP_WORDS = {
    "dong", "chi", "ong", "ba", "va", "cua", "tai", "voi", "trong", "ngoai",
    "duoc", "da", "se", "dang", "la", "o", "cho", "tren", "duoi", "ngay",
    "thang", "nam", "moi", "nhat", "gan", "day", "ve", "theo", "tu", "den",
    "khoa", "xvi", "xv", "vn", "com", "cac", "nhung", "nhu", "hay", "ma",
    "vi", "noi", "cung", "rat", "hon", "nay", "do", "gi", "sao", "neu",
    "pho", "thu", "tuong", "chinh", "phu", "tong", "bi", "thu",
    "chu", "tich", "nuoc", "quoc", "hoi", "bo", "truong", "lanh", "dao",
    "lam", "viec", "hop", "tham", "gap", "go", "tiep", "xuc", "tai", "ha", "noi",
}


def _title_tokens(title):
    """Extract meaningful tokens from a title for similarity comparison."""
    norm = _normalise_text(title)
    return {t for t in norm.split() if len(t) > 1 and t not in _TITLE_STOP_WORDS}


def _titles_are_similar(title_a, title_b, threshold=0.6):
    """Return True if two titles share enough tokens to be about the same event."""
    tokens_a = _title_tokens(title_a)
    tokens_b = _title_tokens(title_b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return (overlap / smaller) >= threshold


def _combined_tokens(result):
    """Extract comparison tokens from title + snippet for event-level deduplication."""
    text = _normalise_text(f"{result.get('title', '')} {result.get('snippet', '')}")
    return {t for t in text.split() if len(t) > 1 and t not in _TITLE_STOP_WORDS}


def _token_overlap_ratio(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))
    return overlap / smaller


def _extract_date_key(result):
    """Extract a stable date key (dd/mm[/yyyy]) when present for same-day matching."""
    raw = _normalise_text(
        f"{result.get('date', '')} {result.get('title', '')} {result.get('snippet', '')}"
    )
    match = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", raw)
    return match.group(1) if match else ""


def _results_are_same_event(result_a, result_b):
    """Return True when two results likely describe the same event across sources."""
    link_a = (result_a.get("link") or "").strip()
    link_b = (result_b.get("link") or "").strip()
    if link_a and link_b and link_a == link_b:
        return True

    if _titles_are_similar(result_a.get("title", ""), result_b.get("title", ""), threshold=0.65):
        return True

    tokens_a = _combined_tokens(result_a)
    tokens_b = _combined_tokens(result_b)
    overlap = _token_overlap_ratio(tokens_a, tokens_b)

    if overlap >= 0.72:
        return True

    date_a = _extract_date_key(result_a)
    date_b = _extract_date_key(result_b)
    if date_a and date_b and date_a == date_b and overlap >= 0.6:
        return True

    return False


def _deduplicate_results(results):
    """Remove results covering the same event, keeping the first (highest priority)."""
    unique = []
    for result in results:
        is_dup = any(_results_are_same_event(result, kept) for kept in unique)
        if not is_dup:
            unique.append(result)
    return unique


def get_internet_info(query, person_name=None):
    """Search for recent Vietnamese news articles related to the query.

    Returns a formatted multi-line string of results, or a fallback message
    if no results are found or the request fails.

    Uses the Serper News endpoint first (purpose-built for news articles),
    then falls back to general search with a time filter if needed.
    """
    headers = {
        "X-API-KEY": config.SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    base_payload = {
        "q": query,
        "hl": "vi",
        "gl": "vn",
        # Fetch a wider candidate pool, then deduplicate and trim to final size.
        "num": max(config.SERPER_MAX_RESULTS * 3, config.SERPER_MAX_RESULTS),
    }

    results = None

    # Primary: dedicated News endpoint for higher-quality news results
    try:
        resp = requests.post(
            config.SERPER_NEWS_URL,
            headers=headers,
            data=json.dumps(base_payload),
            timeout=config.SERPER_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("news")
        if results:
            log.info("News endpoint returned %d results for: %s", len(results), query)
    except requests.RequestException as exc:
        log.warning("Serper News API failed (%s), falling back to general search", exc)

    # Fallback: general search with month time-filter
    if not results:
        try:
            fallback_payload = {**base_payload, "tbs": "qdr:m"}
            resp = requests.post(
                config.SERPER_URL,
                headers=headers,
                data=json.dumps(fallback_payload),
                timeout=config.SERPER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic")
            if results:
                log.info("General search fallback returned %d results for: %s", len(results), query)
        except requests.Timeout:
            log.warning("Serper API timed out for query: %s", query)
            return "Không thể kết nối internet: hết thời gian chờ."
        except requests.RequestException as exc:
            log.error("Serper API request failed: %s", exc)
            return f"Không thể kết nối internet. Chi tiết: {exc}"

    if not results:
        return "Không có tin tức nào được tìm thấy trên Internet trong thời gian gần đây."

    # Hard-block social media, blogs, and tabloid sources.
    results = [r for r in results if not _is_blocked_domain(_extract_domain(r.get("link", "")))]
    if not results:
        return "Không có tin tức nào từ nguồn báo chí chính thống trong thời gian gần đây."

    relevant = [r for r in results if _is_result_relevant_to_person(r, person_name)]
    if relevant:
        results = relevant

    with_time = [r for r in results if _has_concrete_time_hint(r)]
    if with_time:
        results = with_time

    # Keep higher-trust sources first and push lower-trust domains down.
    results = sorted(results, key=_result_priority_tuple, reverse=True)

    # If official sources exist, keep only official sources for cleaner output.
    official_results = [r for r in results if _is_official_domain(_extract_domain(r.get("link", "")))]
    if official_results:
        results = official_results

    # Deduplicate results covering the same event (title similarity).
    results = _deduplicate_results(results)

    results = results[:config.SERPER_MAX_RESULTS]

    lines = []
    for idx, res in enumerate(results, start=1):
        title = res.get("title", "Không có tiêu đề")
        snippet = res.get("snippet", "Không có nội dung")
        date = res.get("date", "Gần đây")
        link = res.get("link", "")
        domain = _extract_domain(link) or "không rõ nguồn"
        lines.append(
            f"Tin {idx} ({date})\n"
            f"Tiêu đề: {title}\n"
            f"Nội dung: {snippet}\n"
            f"Nguồn: {domain}\n"
            f"Link: {link}"
        )

    return "\n\n".join(lines)