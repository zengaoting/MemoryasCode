"""Dataset metadata for the unified runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


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


def is_locomo() -> bool:
    return DATASET_NAME == "LoCoMo"


def dataset_key() -> str:
    return "locomo" if is_locomo() else "longmemeval"


def category_id(value):
    """Normalize category identifiers exactly as each reference dataset does."""
    if not is_locomo():
        return str(value or "").strip().lower()
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def should_skip_qa_category(category) -> bool:
    return is_locomo() and category_id(category) == 5


def policy_fingerprint() -> str:
    """Fingerprint the integrated prompts, planning, and answer implementation.

    Include the dataset so checkpoints cannot cross benchmark boundaries.
    Changing source layout invalidates old QA/planner checkpoints, while
    semantic-build fingerprints remain governed by prompt_contracts.py.
    """
    digest = hashlib.sha256(dataset_key().encode("utf-8"))
    for name in ("benchmark.py", "prompts.py", "program_executor.py", "qa_engine.py"):
        digest.update(name.encode("utf-8"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def runtime_defaults(provider: str) -> dict[str, str]:
    """Non-secret settings shared by the two reference projects' .env files.

    Return values rather than changing the process environment: settings belong
    to a run and must also be recorded in its manifest.
    """
    settings = {
        "CROSS_ENCODER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
        "CROSS_ENCODER_LOCAL_FILES_ONLY": "true",
        "CROSS_ENCODER_BATCH_SIZE": "8",
        "CROSS_ENCODER_MAX_LENGTH": "512",
        "OUR_V1_SESSION_MAX_WORKERS": "2",
        "SESSION_FACT_MAX_FACTS_PER_CALL": "64",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if provider in {"deepseek", "qwen"}:
        prefix = provider.upper()
        settings.update({
            f"{prefix}_REQUEST_TIMEOUT": "900",
            f"{prefix}_API_MAX_RETRIES": "3",
            f"{prefix}_API_RETRY_BACKOFF": "2",
        })
    return settings


def set_dataset(dataset: str) -> BenchmarkProfile:
    global DATASET_NAME, METHOD_NAME
    profile = PROFILES[dataset]
    LM_CATEGORIES[:] = profile.categories
    LM_CATEGORY_NAMES.clear()
    LM_CATEGORY_NAMES.update(profile.category_names)
    DATASET_NAME, METHOD_NAME = profile.name, profile.method
    return profile


set_dataset("longmemeval")
