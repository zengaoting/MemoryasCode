"""Dataset metadata for the unified runtime."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    categories: tuple[str, ...]
    category_names: dict[str, str]
    method: str


PROFILES = {
    "locomo": BenchmarkProfile("LoCoMo", ("1", "2", "3", "4"), {"1": "Multi-hop", "2": "Temporal", "3": "Open-domain", "4": "Single-hop"}, "memory_as_code_programmatic_search"),
    "longmemeval": BenchmarkProfile("LongMemEval", ("multi-session", "single-session-user", "temporal-reasoning", "single-session-preference", "knowledge-update", "single-session-assistant"), {"multi-session": "Multi-session", "single-session-user": "Single-session user", "temporal-reasoning": "Temporal reasoning", "single-session-preference": "Single-session preference", "knowledge-update": "Knowledge update", "single-session-assistant": "Single-session assistant"}, "memory_as_code_programmatic_search_view_expansion"),
}

# Existing modules import these containers directly, so mutate them in place.
LM_CATEGORIES: list[str] = []
LM_CATEGORY_NAMES: dict[str, str] = {}
DATASET_NAME = ""
METHOD_NAME = ""
COST_GROUPS = {"construction": ("fact_extraction", "memory_build"), "retrieval": ("candidate_retrieval", "program_execution"), "answer_generation": ("answer_generation",), "evaluation": ("evaluation",)}


def set_dataset(dataset: str) -> BenchmarkProfile:
    global DATASET_NAME, METHOD_NAME
    profile = PROFILES[dataset]
    LM_CATEGORIES[:] = profile.categories
    LM_CATEGORY_NAMES.clear()
    LM_CATEGORY_NAMES.update(profile.category_names)
    DATASET_NAME, METHOD_NAME = profile.name, profile.method
    return profile


set_dataset("longmemeval")
