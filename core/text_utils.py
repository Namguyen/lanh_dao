"""Shared text normalization and token helpers.

Centralizes Vietnamese normalization logic and avoids duplicated inline token
construction across modules.
"""

import functools
import re
import unicodedata

import config


@functools.lru_cache(maxsize=1)
def role_modifier_tokens() -> frozenset[str]:
    """Return flattened modifier tokens derived from configured phrases."""
    return frozenset(
        token
        for phrase in config.ROLE_MODIFIER_PHRASES
        for token in phrase.split()
    )


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text (lowercase + remove accents + trim spaces)."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.split())


def normalize_role_text(text: str) -> str:
    """Normalize role strings while preserving segment delimiters (; and ,)."""
    text = normalize_text(text)
    text = re.sub(r"[^\w\s;,]", " ", text)
    text = re.sub(r"\s*[;,]\s*", " ; ", text)
    return " ".join(text.split())


def to_ascii_text(text: str) -> str:
    """Normalize text and map Vietnamese 'đ' to 'd' for ASCII matching."""
    return " ".join(normalize_text(text).replace("đ", "d").split())


def tokenize_normalized(text: str, min_len: int = 1) -> list[str]:
    """Normalize and split text into tokens constrained by minimum length."""
    return [t for t in normalize_text(text).split() if len(t) >= min_len]
