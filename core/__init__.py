"""Core module for Vietnamese politician search system.

This package breaks down the monolithic core.py into focused, testable components
following the pipeline architecture: Intent → Retrieval → Filtering → Internet → Answer.
"""

from .intent_classifier import analyze_query_intent, is_ambiguous_leadership_query
from .retriever import retrieve_candidates
from .role_filter import (
    rerank_by_generic_role_rules,
    filter_single_role_candidates,
    apply_list_query_filters,
)
from .internet_search import (
    should_search_internet,
    extract_web_sources,
    generate_evidence_first_news_answer,
)
from .llm_engine import format_direct_answer, generate_answer
from .orchestrator import process_query

__all__ = [
    "analyze_query_intent",
    "is_ambiguous_leadership_query",
    "retrieve_candidates",
    "rerank_by_generic_role_rules",
    "filter_single_role_candidates",
    "apply_list_query_filters",
    "should_search_internet",
    "extract_web_sources",
    "generate_evidence_first_news_answer",
    "format_direct_answer",
    "generate_answer",
    "process_query",
]
