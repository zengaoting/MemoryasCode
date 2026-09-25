from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

from .cost_metrics import lock_excluding_queue_wait


DEFAULT_CROSS_ENCODER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 384
_CROSS_ENCODER_LOAD_LOCK = threading.Lock()
_CROSS_ENCODER_PREDICT_LOCK = threading.Lock()


def rank_facts_with_cross_encoder(
    question: str,
    facts: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Rank every supplied fact with the mandatory Cross Encoder.

    This helper intentionally has no lexical fallback. If the model cannot be
    loaded or scored, the caller should fail so graph-aware runs never silently
    degrade into the old ranking path.
    """
    metadata: dict[str, Any] = {
        "strategy": "cross_encoder",
        "input_fact_count": len(facts),
        "model": os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL),
    }
    if not facts:
        metadata["reason"] = "no_candidates"
        metadata["top_scores"] = []
        return [], metadata

    model_name = metadata["model"]
    max_length = _env_int("CROSS_ENCODER_MAX_LENGTH", DEFAULT_MAX_LENGTH)
    local_only = _env_bool("CROSS_ENCODER_LOCAL_FILES_ONLY", True)
    metadata["local_files_only"] = local_only
    scorer = _get_cross_encoder(model_name, max_length, local_only)
    pairs = [(question, _format_fact_for_reranker(fact)) for fact in facts]
    with lock_excluding_queue_wait(_CROSS_ENCODER_PREDICT_LOCK):
        scores = scorer.predict(
            pairs,
            batch_size=_env_int("CROSS_ENCODER_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            show_progress_bar=False,
        )
    scored = [
        (float(score), idx, _with_cross_encoder_score(fact, float(score)))
        for idx, (score, fact) in enumerate(zip(scores, facts))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [fact for _, _, fact in scored]
    metadata.update(
        {
            "max_length": max_length,
            "top_scores": [
                {
                    "fact_id": fact.get("fact_id", ""),
                    "score": round(score, 6),
                }
                for score, _, fact in scored[: min(8, len(scored))]
            ],
        }
    )
    return ranked, metadata


def _with_cross_encoder_score(fact: dict, score: float) -> dict:
    item = dict(fact)
    item["_cross_encoder_score"] = score
    return item


def _format_fact_for_reranker(fact: dict) -> str:
    dialogue_ids = fact.get("dialogue_ids") or [fact.get("dia_id", "")]
    fields = [
        ("speaker", fact.get("speaker")),
        ("subject", fact.get("subject")),
        ("time", fact.get("normalized_time") or fact.get("time_text")),
        ("topics", ", ".join(str(t) for t in fact.get("topics", []) if t)),
        ("dialogue_id", ",".join(str(d) for d in dialogue_ids if d)),
        ("fact", fact.get("fact_text")),
    ]
    return "\n".join(f"{key}: {value}" for key, value in fields if value)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_cross_encoder(model_name: str, max_length: int, local_only: bool):
    # functools.lru_cache can execute duplicate first calls under concurrent
    # threads. The explicit lock prevents multiple GPU model replicas from
    # loading at once during parallel QA.
    with lock_excluding_queue_wait(_CROSS_ENCODER_LOAD_LOCK):
        return _load_cross_encoder(model_name, max_length, local_only)


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, max_length: int, local_only: bool):
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {"max_length": max_length}
    if local_only:
        kwargs["model_kwargs"] = {"local_files_only": True}
        kwargs["processor_kwargs"] = {"local_files_only": True}
        kwargs["config_kwargs"] = {"local_files_only": True}
    return CrossEncoder(model_name, **kwargs)
