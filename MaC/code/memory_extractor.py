import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from .cost_metrics import failed_usage_metric, local_metric, merge_metrics, usage_metric
from .data_loader import get_sessions, format_session_messages
from .llm_client import LLMClient, capture_token_usage
from .prompts import SESSION_MEMORY_EXTRACTOR_SYSTEM, SESSION_MEMORY_EXTRACTOR_USER, fill_template
from .utils import read_jsonl, write_json, write_jsonl, ensure_dir


logger = logging.getLogger(__name__)
SESSION_FACT_CHECKPOINT_DIR = "atomic_fact_sessions"
ATOMIC_FACTS_DONE_FILE = "atomic_facts.done.json"
DEFAULT_SESSION_FACT_MAX_FACTS_PER_CALL = 64
DEFAULT_SESSION_MAX_WORKERS = 2
EXTRACTION_SCHEMA_VERSION = 6


def _resume_ignores_llm_configuration() -> bool:
    """Whether resumptions may reuse completed LLM stages across config changes.

    Parallelism never affects an extraction result, but model names and prompts are
    intentionally part of the strict checkpoint identity.  Long-running benchmark
    jobs can opt into a less expensive policy when those changes have been reviewed:
    retain the cached extraction as long as it belongs to the same sample and the
    cached fact file itself is intact.  This does *not* accept a missing or corrupted
    fact file.
    """
    return os.getenv("OUR_V1_RESUME_IGNORE_LLM_CONFIG", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _cost_excluded_session_ids() -> set[str]:
    """Return optional session IDs excluded from fact-extraction cost accounting.

    Facts from these sessions are still extracted and used to build memory.  This
    switch only excludes their request usage from the published aggregate, which
    is useful when a separately handled session must not affect a comparable
    cost report.
    """
    return {
        session_id.strip()
        for session_id in os.getenv("EXCLUDE_SESSION_IDS_FROM_COST", "").split(",")
        if session_id.strip()
    }


def extract_atomic_facts_for_sample(
    sample: dict,
    sample_id: str,
    llm: LLMClient,
    out_dir: str,
    force: bool = False,
) -> list[dict]:
    extraction_started = time.monotonic()
    ensure_dir(out_dir)
    facts_path = f"{out_dir}/atomic_facts.jsonl"
    checkpoint_dir = Path(out_dir) / SESSION_FACT_CHECKPOINT_DIR
    done_path = Path(out_dir) / ATOMIC_FACTS_DONE_FILE

    sessions = [
        sess
        for sess in get_sessions(sample)
        if format_session_messages(sess["messages"]).strip()
    ]
    extractor_sha256 = _extractor_fingerprint(llm)
    session_inputs = [
        {
            "session_id": sess["session_id"],
            "input_sha256": _session_input_fingerprint(
                sess,
                format_session_messages(sess["messages"]),
                extractor_sha256,
            ),
        }
        for sess in sessions
    ]
    session_input_by_id = {
        item["session_id"]: item["input_sha256"] for item in session_inputs
    }

    if force:
        Path(facts_path).unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        _clear_session_checkpoints(checkpoint_dir)
    elif _done_marker_valid(
        done_path,
        sample_id,
        session_inputs,
        extractor_sha256,
        Path(facts_path),
    ) or _done_marker_resumable(done_path, sample_id, Path(facts_path)):
        cached = read_jsonl(facts_path)
        logger.info("  Atomic facts complete for %s; reusing %d cached facts", sample_id, len(cached))
        return cached

    conv = sample.get("conversation", {})
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")

    ensure_dir(checkpoint_dir)
    pending_sessions: list[tuple[int, dict]] = []
    for i, sess in enumerate(sessions):
        session_input_sha256 = session_input_by_id[sess["session_id"]]
        checkpoint_path = _session_checkpoint_path(checkpoint_dir, sess["session_id"])
        if _session_checkpoint_valid(
            checkpoint_path,
            sess["session_id"],
            session_input_sha256,
            extractor_sha256,
        ):
            logger.debug("  Reusing atomic fact checkpoint for %s/%s", sample_id, sess["session_id"])
            continue
        pending_sessions.append((i, sess))

    def extract_one_session(item: tuple[int, dict]) -> None:
        i, sess = item
        formatted = format_session_messages(sess["messages"])
        session_input_sha256 = session_input_by_id[sess["session_id"]]
        checkpoint_path = _session_checkpoint_path(checkpoint_dir, sess["session_id"])
        inflight_path = _session_inflight_path(checkpoint_dir, sess["session_id"])
        prior_inflight = _read_session_inflight(
            inflight_path,
            session_id=sess["session_id"],
            session_input_sha256=session_input_sha256,
            extractor_sha256=extractor_sha256,
        )
        prior_metric = prior_inflight.get("cost_metrics") if prior_inflight else None
        write_json(
            {
                "schema": "atomic_fact_session_inflight_v1",
                "sample_id": sample_id,
                "session_id": sess["session_id"],
                "session_input_sha256": session_input_sha256,
                "extractor_sha256": extractor_sha256,
                "cost_metrics": prior_metric,
            },
            inflight_path,
        )
        session_started = time.monotonic()
        try:
            with capture_token_usage() as usage:
                facts = _extract_session_facts(
                    sample_id=sample_id,
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                    sess=sess,
                    formatted=formatted,
                    session_position=i + 1,
                    session_count=len(sessions),
                    llm=llm,
                )
        except BaseException:
            current = failed_usage_metric(
                usage.snapshot(), time.monotonic() - session_started
            )
            failed_metric = merge_metrics(prior_metric, current) if prior_metric else current
            write_json(
                {
                    "schema": "atomic_fact_session_inflight_v1",
                    "sample_id": sample_id,
                    "session_id": sess["session_id"],
                    "session_input_sha256": session_input_sha256,
                    "extractor_sha256": extractor_sha256,
                    "cost_metrics": failed_metric,
                },
                inflight_path,
            )
            raise

        current_metric = usage_metric(usage.snapshot(), time.monotonic() - session_started)
        session_metric = merge_metrics(prior_metric, current_metric) if prior_metric else current_metric
        _write_session_checkpoint(
            checkpoint_path,
            sample_id=sample_id,
            session_id=sess["session_id"],
            session_datetime=sess["datetime"],
            session_input_sha256=session_input_sha256,
            extractor_sha256=extractor_sha256,
            facts=facts,
            cost_metrics=session_metric,
        )
        inflight_path.unlink(missing_ok=True)

    setup_runtime_seconds = time.monotonic() - extraction_started
    failures: list[BaseException] = []
    session_workers = min(_session_max_workers(), max(1, len(pending_sessions)))
    if pending_sessions:
        logger.info(
            "  Extracting %d pending sessions for %s with %d session workers",
            len(pending_sessions),
            sample_id,
            session_workers,
        )
        with ThreadPoolExecutor(
            max_workers=session_workers,
            thread_name_prefix=f"session-{sample_id}",
        ) as executor:
            futures = [
                executor.submit(extract_one_session, item)
                for item in pending_sessions
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    failures.append(exc)
    if failures:
        raise failures[0]

    return _assemble_atomic_facts(
        sample_id=sample_id,
        sessions=sessions,
        checkpoint_dir=checkpoint_dir,
        facts_path=Path(facts_path),
        done_path=done_path,
        session_inputs=session_inputs,
        extractor_sha256=extractor_sha256,
        setup_runtime_seconds=setup_runtime_seconds,
    )


def _extract_session_facts(
    sample_id: str,
    speaker_a: str,
    speaker_b: str,
    sess: dict,
    formatted: str,
    session_position: int,
    session_count: int,
    llm: LLMClient,
) -> list[dict]:
    logger.info(
        "  Extracting facts for %s/%s (%d/%d) as one complete session (%d chars)",
        sample_id,
        sess["session_id"],
        session_position,
        session_count,
        len(formatted),
    )
    user = _build_session_extraction_prompt(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sess=sess,
        session_messages=formatted,
    )
    result = llm.chat_json(SESSION_MEMORY_EXTRACTOR_SYSTEM, user)
    return _clean_extracted_facts(result, sample_id, sess)


def _build_session_extraction_prompt(
    sample_id: str,
    speaker_a: str,
    speaker_b: str,
    sess: dict,
    session_messages: str,
) -> str:
    user = fill_template(
        SESSION_MEMORY_EXTRACTOR_USER,
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        session_id=sess["session_id"],
        session_datetime=sess["datetime"],
        session_messages=session_messages,
    )
    fact_limit = _session_fact_max_facts_per_call()
    limit_note = (
        f"Extraction limit: return at most {fact_limit} high-value atomic facts. "
        "Prioritize user-specific memories, user-stated preferences, plans, events, places, "
        "relationships, and repeated interests. Also preserve answer-bearing information "
        "provided by the assistant, including concrete facts, recommendations, instructions, "
        "lists, and table entries that a later question may ask the user to recall."
    )
    user = user.replace("\nSession messages:\n", f"\n{limit_note}\n\nSession messages:\n", 1)
    return user


def _clean_extracted_facts(result: dict, sample_id: str, sess: dict) -> list[dict]:
    facts = []
    for fact in result.get("atomic_facts", [])[: _session_fact_max_facts_per_call()]:
        if not isinstance(fact, dict):
            continue
        item = dict(fact)
        item.pop("fact_id", None)
        item["sample_id"] = sample_id
        item["session_id"] = sess["session_id"]
        item["session_datetime"] = sess["datetime"]
        if not isinstance(item.get("topics"), list):
            topic = item.get("topics")
            item["topics"] = [str(topic)] if topic else ["general"]
        facts.append(item)
    return facts


def _session_fact_max_facts_per_call() -> int:
    try:
        return max(1, int(os.getenv("SESSION_FACT_MAX_FACTS_PER_CALL", str(DEFAULT_SESSION_FACT_MAX_FACTS_PER_CALL))))
    except ValueError:
        return DEFAULT_SESSION_FACT_MAX_FACTS_PER_CALL


def _session_max_workers() -> int:
    try:
        return max(
            1,
            int(os.getenv("OUR_V1_SESSION_MAX_WORKERS", str(DEFAULT_SESSION_MAX_WORKERS))),
        )
    except ValueError:
        return DEFAULT_SESSION_MAX_WORKERS


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extractor_fingerprint(llm: LLMClient) -> str:
    return extractor_configuration_fingerprint(str(getattr(llm, "model", "")))


def extractor_configuration_fingerprint(model: str) -> str:
    """Fingerprint every input that can change extraction output."""
    payload = {
        "version": EXTRACTION_SCHEMA_VERSION,
        "model": str(model),
        "max_facts_per_call": _session_fact_max_facts_per_call(),
        "system_prompt_sha256": _text_hash(SESSION_MEMORY_EXTRACTOR_SYSTEM),
        "user_prompt_sha256": _text_hash(SESSION_MEMORY_EXTRACTOR_USER),
    }
    return _text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _session_input_fingerprint(
    sess: dict,
    formatted_messages: str,
    extractor_sha256: str,
) -> str:
    payload = {
        "session_id": str(sess.get("session_id", "")),
        "session_datetime": str(sess.get("datetime", "")),
        "formatted_messages": formatted_messages,
        "extractor_sha256": extractor_sha256,
    }
    return _text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fact_id(index: int) -> str:
    return f"fact_{index}"


def _safe_id(value: str) -> str:
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "unknown"


def _session_checkpoint_path(checkpoint_dir: Path, session_id: str) -> Path:
    return checkpoint_dir / f"{_safe_id(session_id)}.json"


def _session_inflight_path(checkpoint_dir: Path, session_id: str) -> Path:
    return checkpoint_dir / f"{_safe_id(session_id)}.inflight.json"


def _read_session_inflight(
    path: Path,
    *,
    session_id: str,
    session_input_sha256: str,
    extractor_sha256: str,
) -> dict | None:
    payload = _read_json_if_exists(path)
    if not payload:
        return None
    if not (
        payload.get("schema") == "atomic_fact_session_inflight_v1"
        and payload.get("session_id") == session_id
        and payload.get("session_input_sha256") == session_input_sha256
        and payload.get("extractor_sha256") == extractor_sha256
    ):
        return None
    return payload


def _write_session_checkpoint(
    path: Path,
    sample_id: str,
    session_id: str,
    session_datetime: str,
    session_input_sha256: str,
    extractor_sha256: str,
    facts: list[dict],
    cost_metrics: dict,
) -> None:
    write_json(
        {
            "schema": "atomic_fact_session_checkpoint_v2",
            "extraction_version": EXTRACTION_SCHEMA_VERSION,
            "extractor_sha256": extractor_sha256,
            "session_input_sha256": session_input_sha256,
            "sample_id": sample_id,
            "session_id": session_id,
            "session_datetime": session_datetime,
            "completed": True,
            "fact_count": len(facts),
            "atomic_facts": facts,
            "cost_metrics": cost_metrics,
        },
        path,
    )


def _session_checkpoint_valid(
    path: Path,
    session_id: str,
    session_input_sha256: str,
    extractor_sha256: str,
) -> bool:
    payload = _read_json_if_exists(path)
    return bool(
        payload
        and payload.get("completed") is True
        and payload.get("extraction_version") == EXTRACTION_SCHEMA_VERSION
        and payload.get("extractor_sha256") == extractor_sha256
        and payload.get("session_input_sha256") == session_input_sha256
        and payload.get("session_id") == session_id
        and isinstance(payload.get("atomic_facts"), list)
    )


def _read_session_checkpoint(path: Path) -> dict:
    payload = _read_json_if_exists(path)
    if not payload or not isinstance(payload.get("atomic_facts"), list):
        raise ValueError(f"Invalid atomic fact checkpoint: {path}")
    return payload


def _assemble_atomic_facts(
    sample_id: str,
    sessions: list[dict],
    checkpoint_dir: Path,
    facts_path: Path,
    done_path: Path,
    session_inputs: list[dict],
    extractor_sha256: str,
    setup_runtime_seconds: float,
) -> list[dict]:
    assembly_started = time.monotonic()
    assembled = []
    fact_counter = 0
    session_metrics = []
    excluded_from_cost = _cost_excluded_session_ids()
    for sess in sessions:
        checkpoint = _read_session_checkpoint(
            _session_checkpoint_path(checkpoint_dir, sess["session_id"])
        )
        session_id = str(sess["session_id"])
        if session_id in excluded_from_cost:
            logger.warning(
                "Excluding fact-extraction cost for %s/%s; facts remain in memory",
                sample_id,
                session_id,
            )
        else:
            session_metrics.append(checkpoint.get("cost_metrics"))
        for raw_fact in checkpoint.get("atomic_facts", []):
            fact = dict(raw_fact)
            fact_counter += 1
            source_fact_id = fact.get("fact_id", "")
            fact["source_fact_id"] = source_fact_id
            fact["fact_id"] = _canonical_fact_id(fact_counter)
            fact.setdefault("sample_id", sample_id)
            fact.setdefault("session_id", sess["session_id"])
            fact.setdefault("session_datetime", sess["datetime"])
            fact.setdefault("topics", ["general"])
            fact.setdefault("semantic_key", _semantic_key(fact))
            assembled.append(fact)

    write_jsonl(assembled, facts_path)
    local_work_metric = local_metric(
        setup_runtime_seconds + (time.monotonic() - assembly_started)
    )
    if session_metrics:
        extraction_metric = merge_metrics(*session_metrics, local_work_metric)
    else:
        extraction_metric = local_work_metric
    _write_done_marker(
        done_path,
        sample_id,
        session_inputs,
        extractor_sha256,
        len(assembled),
        _file_sha256(facts_path),
        cost_metrics=extraction_metric,
        cost_excluded_session_ids=excluded_from_cost,
    )
    logger.info("  Atomic facts complete for %s: %d facts", sample_id, len(assembled))
    return assembled


def _done_marker_valid(
    path: Path,
    sample_id: str,
    expected_session_inputs: list[dict],
    extractor_sha256: str,
    facts_path: Path,
) -> bool:
    payload = _read_json_if_exists(path)
    return bool(
        payload
        and payload.get("schema") == "atomic_facts_done_v2"
        and payload.get("extraction_version") == EXTRACTION_SCHEMA_VERSION
        and payload.get("extractor_sha256") == extractor_sha256
        and payload.get("sample_id") == sample_id
        and payload.get("completed") is True
        and payload.get("session_inputs") == expected_session_inputs
        and facts_path.exists()
        and payload.get("facts_sha256") == _file_sha256(facts_path)
    )


def _done_marker_resumable(path: Path, sample_id: str, facts_path: Path) -> bool:
    """Accept a completed extraction without matching its LLM configuration.

    The opt-in keeps the artifact/content checks, so a changed or damaged
    ``atomic_facts.jsonl`` is never silently reused.
    """
    if not _resume_ignores_llm_configuration():
        return False
    payload = _read_json_if_exists(path)
    return bool(
        payload
        and payload.get("schema") == "atomic_facts_done_v2"
        and payload.get("sample_id") == sample_id
        and payload.get("completed") is True
        and facts_path.exists()
        and payload.get("facts_sha256") == _file_sha256(facts_path)
    )


def _write_done_marker(
    path: Path,
    sample_id: str,
    session_inputs: list[dict],
    extractor_sha256: str,
    fact_count: int,
    facts_sha256: str,
    cost_metrics: dict,
    cost_excluded_session_ids: set[str] | None = None,
) -> None:
    write_json(
        {
            "schema": "atomic_facts_done_v2",
            "extraction_version": EXTRACTION_SCHEMA_VERSION,
            "extractor_sha256": extractor_sha256,
            "sample_id": sample_id,
            "completed": True,
            "session_inputs": session_inputs,
            "fact_count": fact_count,
            "facts_sha256": facts_sha256,
            "stage_costs": {"fact_extraction": cost_metrics},
            "cost_excluded_session_ids": sorted(cost_excluded_session_ids or []),
        },
        path,
    )


def _clear_session_checkpoints(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.rglob("*.json"):
        path.unlink(missing_ok=True)


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring invalid checkpoint JSON: %s", path)
        return {}


def _semantic_key(fact: dict) -> str:
    parts = [
        fact.get("speaker", ""),
        fact.get("subject", ""),
        fact.get("fact_type", ""),
        fact.get("normalized_time", ""),
        fact.get("fact_text", ""),
    ]
    text = " ".join(str(p) for p in parts).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())
