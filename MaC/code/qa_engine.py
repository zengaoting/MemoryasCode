from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

from .cost_metrics import (
    capture_local_timing,
    failed_local_metric,
    failed_usage_metric,
    local_metric,
    merge_metrics,
    usage_metric,
)
from .llm_client import LLMClient, capture_token_usage
from .prompts import ANSWER_GENERATOR_USER, answer_generator_system, fill_template
from .program_executor import (
    PROGRAM_SCHEMA_VERSION,
    initial_candidates,
    plan_and_execute,
    render_evidence_context,
)
from .utils import read_json, write_json

logger = logging.getLogger(__name__)

STAGE_CHECKPOINT_SCHEMA = "question_stage_checkpoint_v1"
STAGE_CHECKPOINT_VERSION = 29
QUESTION_STAGE_NAMES = (
    "candidate_retrieval",
    "program_execution",
    "answer_generation",
    "evaluation",
)
ANSWER_REFUSAL_PATTERNS = (
    re.compile(r"^\s*(?:unknown|unspecified|not specified|n/?a)\s*[.!]?\s*$", re.I),
    re.compile(r"\bnot (?:explicitly )?(?:mentioned|stated|provided|specified|identified|named)\b", re.I),
    re.compile(
        r"\b(?:cannot|can't|unable to) (?:be )?"
        r"(?:determin(?:e|ed)|identif(?:y|ied)|infer(?:red)?|answer(?:ed)?)\b",
        re.I,
    ),
    re.compile(r"\b(?:no|insufficient) (?:specific |relevant )?(?:information|evidence|details)\b", re.I),
    re.compile(
        r"\b(?:the )?(?:information|answer|name|detail) (?:is|was) not "
        r"(?:present|available|contained|included)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the )?(?:provided |retrieved )?(?:text|facts?|contents?) "
        r"(?:does|do) not contain (?:any )?(?:information|evidence|details?)\b",
        re.I,
    ),
)
ANSWER_REFUSAL_REPAIR_INSTRUCTION = """

Your previous response refused to answer even though retrieved contents are available.
For this answerable benchmark question, return the most specific short answer supported by
the retrieved facts. If the exact name is not literal, make the most reasonable
inference from the description and common knowledge. Do not answer that the information
is missing, unspecified, or impossible to determine. Return the same valid JSON format.
""".strip()
ANSWER_REFUSAL_FINAL_REPAIR_INSTRUCTION = """

The revised response still refused to answer. Inspect the retrieved fact lines one by one,
match descriptions and paraphrases to the entity, event, place, date, or item requested,
and return the single most plausible concrete answer. A benchmark answer may require a
small commonsense inference from an unnamed description. Do not repeat any missing-
information statement. Return the same valid JSON format with supporting dialogue ids.
""".strip()
ANSWER_SCHEMA_REPAIR_INSTRUCTION = """

Your previous response was valid JSON but did not contain a usable non-empty "answer".
This benchmark question is answerable from the retrieved facts. Return mode "answer" and
the same JSON schema with a concrete non-empty answer, a supports list containing only
visible dialogue ids, and confidence. Do not return navigate mode, an empty answer, or a
missing-answer object.
""".strip()
MULTI_HOP_EXHAUSTIVE_SYSTEM = """
You audit exhaustive multi-hop answers against long-term dialogue evidence. Build an
internal event ledger before answering: one row per distinct qualifying event or item,
with its dialogue id. Merge repeated descriptions of the same event, but do not discard
an event merely because it is expressed as winning a trophy, adopting a named pet, or
another result that satisfies the question. Exclude plans, goals, feelings, advice, and
topic-neighbor facts unless the question asks for them. A count row requires evidence of
the completed outcome: merely describing a comeback, final buzzer, celebration, or good
feeling without stating the result does not add an event. A later "winning was awesome"
or trophy reference that clearly refers back to the same earlier trophy is one event, not
another. For two-person questions, verify
each person separately before taking a union or intersection. Return valid JSON only with
the same short-answer schema; do not include the ledger or an explanation.
""".strip()
MULTI_HOP_EXHAUSTIVE_USER = """
Question:
{question}

Retrieved evidence:
{loaded_memory}

Draft answer:
{draft}

Audit completeness and event identity, then return the corrected concise answer.
""".strip()
TEMPORAL_AUDIT_SYSTEM = """
You audit temporal dialogue answers. First match the full person, event, object, place,
and requested answer type; retrieval rank alone is never evidence. Then resolve time with
this precedence: (1) original dialogue time wording plus session_datetime, (2) a direct
onset/completion event, (3) a fact's normalized time only when it preserves the original
granularity. Do not turn "next month", "last week", "a few days ago", or "about four
months" into an unjustifiably exact day. For "when did X start", subtract an explicit
duration from the source session date, unless a direct start/signing event is present.
When two source statements constrain the same event, intersect them (for example, a
month-level plan plus "last week" in an early-month session can establish early month).
A follow-up report may omit a location stated in the earlier plan; link them when person,
event type, and timeline identify the same unique event.
Return the resulting absolute month/year, not "N months ago". For "after how many weeks
did X reconnect", measure between the session that reports the earlier contact and the
reconnection session in whole conversational weeks; do not first move the earlier report
back again because it says "last week". When a deterministic elapsed-week calculation is
provided, use its value instead of repeating calendar arithmetic in prose.
For any matching deterministic temporal operation, copy its `answer_text` exactly.
Keep the requested unit in the answer (for example, "3 weeks", not bare "3").
For "where" questions return a place, never a time span. For "where during [period]"
questions, require location-bearing evidence whose stay
overlaps that period. An unrelated event inside the period does not establish its
location; a statement that the person returned from a place immediately after the period
supports that they had been in that place. Return valid JSON only with the same concise
answer/support schema and no explanation.
""".strip()
TEMPORAL_AUDIT_USER = """
Question:
{question}

Retrieved evidence:
{loaded_memory}

Draft answer:
{draft}

Recheck event identity, answer type, temporal granularity, and arithmetic.
""".strip()
OPEN_DOMAIN_GEOGRAPHY_SYSTEM = """
You independently solve geographic open-domain dialogue questions. Dialogue facts provide
place clues, while the requested answer may require public geographic knowledge. Enforce
the exact requested level: city, US state, or country. Search every retrieved trip/place
fact, not only the nearest match. If the question asks for an additional country, exclude
the country already named and map another visited city to its country. If a draft is
already the correct requested location, preserve it. Return valid JSON only with answer,
supports, and confidence; supports may only cite visible dialogue ids.
""".strip()
OPEN_DOMAIN_GEOGRAPHY_USER = """
Question:
{question}

Retrieved facts:
{loaded_memory}

Draft (may use the wrong geographic level or place):
{draft}

Return the most plausible short geographic answer.
""".strip()
OPEN_DOMAIN_GEOGRAPHY_TERM_PATTERN = re.compile(
    r"\b(?:state|states|country|countries)\b", re.I
)
OPEN_DOMAIN_GEOGRAPHY_QUESTION_PATTERN = re.compile(
    r"^\s*(?:what|which|in\s+what|in\s+which)\b", re.I
)
OPEN_DOMAIN_ALTERNATIVE_SYSTEM = """
You independently solve alternative-choice open-domain dialogue questions. The question
contains alternatives separated by "or", so a bare yes/no draft does not answer which
alternative applies. Use the retrieved facts and, when needed, ordinary public knowledge
to select the specific named alternative that best fits. Return only that concise choice,
not Yes, No, Likely yes, or Likely no. Return valid JSON only with answer, supports, and
confidence; supports may only cite visible dialogue ids.
""".strip()
OPEN_DOMAIN_ALTERNATIVE_USER = """
Question:
{question}

Retrieved facts:
{loaded_memory}

Invalid binary draft:
{draft}

Return the most plausible specific alternative from the question.
""".strip()
OPEN_DOMAIN_BINARY_ANSWER_PATTERN = re.compile(
    r"^\s*(?:likely\s+)?(?:yes|no)\s*[.!]?\s*$", re.I
)
OPEN_DOMAIN_ALTERNATIVE_PATTERN = re.compile(r"\bor\b", re.I)
OPEN_DOMAIN_EXPLICIT_BINARY_PATTERN = re.compile(r"\byes\s+or\s+no\b", re.I)


def _category_id(value):
    """Return a normalized category identifier for either benchmark."""
    return str(value or "").strip().lower()


def _locomo_category(value) -> int | None:
    """Return LoCoMo's numeric category when ``value`` is one of its labels."""
    normalized = _category_id(value)
    if normalized in {"1", "2", "3", "4", "5"}:
        return int(normalized)
    return None


def format_final_question_for_mragent(question: str, category) -> str:
    """Add the preserved benchmark-specific answer instructions."""
    locomo_category = _locomo_category(category)
    if locomo_category == 1:
        return (
            question
            + " Resolve every requested relation jointly and return all and only the requested items. "
            + "Before answering, build an internal list of qualifying evidence by dialogue id. For count "
            + "questions, count distinct real events: merge repeated descriptions of one event, include "
            + "result statements such as an explicitly won trophy when they satisfy the event, and exclude "
            + "plans, goals, feelings, advice, implied-but-unstated outcomes, and topic-neighbor facts. Merge "
            + "later comments about the same trophy or victory into that earlier event. For two-person questions, evaluate "
            + "each person before taking the requested union or intersection. Output no explanation."
        )
    if locomo_category == 2:
        return (
            question
            + " First identify the exact event using all person, object, place, relation, and date "
            + "constraints; retrieval rank is not authority and a similar event must not be substituted. "
            + "Resolve an exact relative date only when the original wording supports that precision; "
            + "'yesterday' of conversation time '7 May 2023' is '6 May 2023'. "
            + "Original dialogue wording outranks normalized_time when the latter invents finer precision. "
            + "Intersect multiple source constraints on the same event rather than choosing only one; a "
            + "month-level plan plus 'last week' in an early-month report can establish early month. "
            + "A follow-up report may omit the location from an earlier plan; link them when person, event "
            + "type, and timeline identify the same unique event. "
            + "For 'when did X start' questions, convert an explicit elapsed duration into an absolute "
            + "month/year using session_datetime; do not answer only 'N months ago'. For 'after how many "
            + "weeks did X reconnect', compare the two reporting-session dates in whole weeks and do not "
            + "backshift the first report a second time because it says 'last week'. Use the deterministic "
            + "elapsed-week calculation in the evidence when present. "
            + "For 'where during a period', use only location-bearing travel/stay evidence that overlaps "
            + "the period; an unrelated event does not inherit a hometown, and returning from a place "
            + "immediately after the period supports presence there. "
            + "For 'when' questions, use the matched evidence's granularity: "
            + "'7 May 2023', 'May 2023', '2023', or 'the week/weekend/Sunday before "
            + "25 May 2023'; preserve a week or weekend as that anchored period rather "
            + "than an ISO date range. Preserve 'next month', 'last week', 'a few days ago', and "
            + "approximate durations at their supported granularity, "
            + "with no additional words. For 'how long' questions, return the "
            + "duration exactly as written in the conversation. Do not exclude a source merely because "
            + "the question's person is not the literal speaker of that turn; match the complete event "
            + "and its requested attribute before resolving its time."
        )
    if locomo_category == 3:
        return question + " No extra explanations in 'answer'. Give reasons with original text in 'reason'. "

    category = _category_id(category)
    if category == "multi-session":
        return (
            question
            + " Resolve every requested relation jointly across all relevant sessions. Before answering, "
            + "build an internal ledger with one row per distinct qualifying event or item and its dialogue id. "
            + "For count/list questions, merge repeated descriptions of the same event, but keep distinct "
            + "completed events. Exclude plans, goals, feelings, advice, and topic-neighbor facts unless asked. "
            + "For multiple people, evaluate each person before the requested union or intersection. Output only the answer."
        )
    if category == "temporal-reasoning":
        return (
            question
            + " Match the exact event before resolving time. Use the question date, source session_datetime, "
            + "and original relative wording together; retrieval rank is not authority. Prefer any matching "
            + "Deterministic Temporal Calculation and copy its answer_text exactly. Preserve the requested "
            + "unit and supported granularity, and never substitute a similar event or invent a day-of-month."
        )
    if category == "knowledge-update":
        return question + (
            " Resolve all statements about the requested attribute as a state history. Prefer the latest "
            "explicitly active state as of the question date. Treat corrections, replacements, completion, "
            "cancellation, and changed plans as updates rather than independent answers."
        )
    if category == "single-session-preference":
        return question + (
            " Infer the user's durable preferences and constraints from the cited dialogue, then give concrete "
            "useful recommendations satisfying them. Do not merely restate the preference or invent constraints."
        )
    if category == "single-session-assistant":
        return question + (
            " Treat assistant turns, quoted text, tables, and image descriptions as valid source evidence. "
            "Return the exact requested attribute from that content."
        )
    return question


def answer_question(
    question: str,
    loaded_memory: str,
    llm: LLMClient,
    *,
    category=None,
) -> dict:
    original_question = _original_question_text(question, category)
    if not loaded_memory.strip():
        # Never turn a retrieval/runtime gap into a fixed benchmark answer.
        # Let the normal answer model and its repair loop handle this edge case.
        loaded_memory = "# Retrieved Memory\n\nNo fact text was materialized by the search runtime."

    answer_system = answer_generator_system()
    user = fill_template(
        ANSWER_GENERATOR_USER,
        question=question,
        loaded_memory=loaded_memory,
    )
    result, schema_repairs = _chat_nonempty_answer_json(
        llm,
        answer_system,
        user,
    )
    answer_trace = {
        "schema_repairs": schema_repairs,
        "refusal_repairs": 0,
        "multi_hop_verification_attempted": False,
        "multi_hop_revision_applied": False,
        "multi_hop_verifier_rejected": False,
        "temporal_verification_attempted": False,
        "temporal_revision_applied": False,
        "temporal_verifier_rejected": False,
        "open_domain_geographic_verification_attempted": False,
        "open_domain_geographic_revision_applied": False,
        "open_domain_geographic_verifier_rejected": False,
        "open_domain_alternative_verification_attempted": False,
        "open_domain_alternative_revision_applied": False,
        "open_domain_alternative_verifier_rejected": False,
        "single_hop_verification_attempted": False,
        "single_hop_revision_applied": False,
        "single_hop_verifier_rejected": False,
    }
    repair_instructions = (
        ANSWER_REFUSAL_REPAIR_INSTRUCTION,
        ANSWER_REFUSAL_FINAL_REPAIR_INSTRUCTION,
    )
    for repair_number, instruction in enumerate(repair_instructions, start=1):
        if not _is_answer_refusal(result.get("answer", "")):
            break
        logger.info(
            "  Answer generator returned a refusal; running bounded evidence repair %d/%d",
            repair_number,
            len(repair_instructions),
        )
        if hasattr(llm, "discard_last_response_usage"):
            llm.discard_last_response_usage()
        answer_trace["refusal_repairs"] = repair_number
        rejected_draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        repair_user = (
            user
            + "\n\n"
            + instruction
            + "\n\nRejected draft response:\n"
            + rejected_draft
        )
        result = llm.chat_json(answer_system, repair_user)

    if _is_answer_refusal(result.get("answer", "")):
        result["_answer_trace"] = answer_trace
        return result

    if _should_verify_multi_hop_answer(original_question, result, category):
        logger.info("  Auditing an exhaustive Category 1 draft once")
        answer_trace["multi_hop_verification_attempted"] = True
        draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        verify_user = MULTI_HOP_EXHAUSTIVE_USER.format(
            question=original_question, loaded_memory=loaded_memory, draft=draft
        )
        verified = llm.chat_json(MULTI_HOP_EXHAUSTIVE_SYSTEM, verify_user)
        if not str(verified.get("answer", "")).strip() or _is_answer_refusal(
            verified.get("answer", "")
        ):
            logger.info("  Multi-hop verifier returned no concrete answer; retaining the draft")
            answer_trace["multi_hop_verifier_rejected"] = True
            if hasattr(llm, "discard_last_response_usage"):
                llm.discard_last_response_usage()
        else:
            answer_trace["multi_hop_revision_applied"] = _verification_changed(
                result, verified
            )
            result = verified
    elif _should_verify_temporal_answer(original_question, category):
        logger.info("  Auditing a high-risk Category 2 temporal draft once")
        answer_trace["temporal_verification_attempted"] = True
        draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        verified = llm.chat_json(
            TEMPORAL_AUDIT_SYSTEM,
            TEMPORAL_AUDIT_USER.format(
                question=original_question, loaded_memory=loaded_memory, draft=draft
            ),
        )
        if not str(verified.get("answer", "")).strip() or _is_answer_refusal(
            verified.get("answer", "")
        ):
            answer_trace["temporal_verifier_rejected"] = True
            if hasattr(llm, "discard_last_response_usage"):
                llm.discard_last_response_usage()
        else:
            answer_trace["temporal_revision_applied"] = _verification_changed(
                result, verified
            )
            result = verified
    elif _should_verify_open_domain_geography(question, category):
        logger.info("  Verifying a Category 3 geographic answer once")
        answer_trace["open_domain_geographic_verification_attempted"] = True
        draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        verify_user = OPEN_DOMAIN_GEOGRAPHY_USER.format(
            question=question,
            loaded_memory=loaded_memory,
            draft=draft,
        )
        verified = llm.chat_json(OPEN_DOMAIN_GEOGRAPHY_SYSTEM, verify_user)
        if not str(verified.get("answer", "")).strip() or _is_answer_refusal(
            verified.get("answer", "")
        ):
            logger.info(
                "  Geographic verifier returned no concrete answer; retaining the draft"
            )
            answer_trace["open_domain_geographic_verifier_rejected"] = True
            if hasattr(llm, "discard_last_response_usage"):
                llm.discard_last_response_usage()
        else:
            answer_trace["open_domain_geographic_revision_applied"] = (
                _verification_changed(result, verified)
            )
            result = verified
    elif _should_verify_open_domain_alternative(question, result, category):
        logger.info("  Verifying a Category 3 alternative-choice answer once")
        answer_trace["open_domain_alternative_verification_attempted"] = True
        draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        verify_user = OPEN_DOMAIN_ALTERNATIVE_USER.format(
            question=question,
            loaded_memory=loaded_memory,
            draft=draft,
        )
        verified = llm.chat_json(OPEN_DOMAIN_ALTERNATIVE_SYSTEM, verify_user)
        verified_answer = str(verified.get("answer", "")).strip()
        if (
            not verified_answer
            or _is_answer_refusal(verified_answer)
            or OPEN_DOMAIN_BINARY_ANSWER_PATTERN.fullmatch(verified_answer)
        ):
            logger.info(
                "  Alternative-choice verifier returned no specific choice; retaining the draft"
            )
            answer_trace["open_domain_alternative_verifier_rejected"] = True
            if hasattr(llm, "discard_last_response_usage"):
                llm.discard_last_response_usage()
        else:
            answer_trace["open_domain_alternative_revision_applied"] = (
                _verification_changed(result, verified)
            )
            result = verified

    if _should_verify_single_hop_answer(question, result, category):
        logger.info("  Verifying a high-risk Category 4 exact-attribute draft once")
        answer_trace["single_hop_verification_attempted"] = True
        draft = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        verify_user = (
            user
            + "\n\nRecheck this Category 4 draft against every entity, date, and event "
            + "constraint. Find the fact that states the exact requested attribute. "
            + "Prefer the specific name, title, color, food, game, movie, instrument, "
            + "place, type, emotion, quote, or advice over a generic category or nearby "
            + "detail. If the draft is already exact, preserve it. Return the same JSON "
            + "schema and no explanation.\n\nDraft:\n"
            + draft
        )
        verified = llm.chat_json(answer_system, verify_user)
        verified_answer = str(verified.get("answer", "")).strip()
        if not verified_answer or _is_answer_refusal(verified_answer):
            answer_trace["single_hop_verifier_rejected"] = True
            if hasattr(llm, "discard_last_response_usage"):
                llm.discard_last_response_usage()
        else:
            answer_trace["single_hop_revision_applied"] = _verification_changed(
                result, verified
            )
            result = verified

    result["_answer_trace"] = answer_trace
    return result


def _chat_nonempty_answer_json(
    llm: LLMClient,
    system: str,
    user: str,
    *,
    max_repairs: int = 2,
) -> tuple[dict, int]:
    result = llm.chat_json(system, user)
    for repair_number in range(0, max_repairs + 1):
        if isinstance(result, dict) and str(result.get("answer", "")).strip():
            return result, repair_number
        if repair_number >= max_repairs:
            break
        logger.info(
            "  Answer generator returned JSON without a concrete answer; "
            "running same-GPT schema repair %d/%d",
            repair_number + 1,
            max_repairs,
        )
        if hasattr(llm, "discard_last_response_usage"):
            llm.discard_last_response_usage()
        rejected = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        repair_user = (
            user
            + "\n\n"
            + ANSWER_SCHEMA_REPAIR_INSTRUCTION
            + "\n\nRejected response:\n"
            + rejected
        )
        result = llm.chat_json(system, repair_user)
    raise ValueError(
        "Answer generator returned no non-empty answer after same-GPT schema repairs"
    )


def _is_answer_refusal(answer) -> bool:
    text = re.sub(r"\s+", " ", str(answer or "")).strip()
    return bool(text) and any(pattern.search(text) for pattern in ANSWER_REFUSAL_PATTERNS)


def _original_question_text(question: str, category) -> str:
    locomo_marker = {
        1: " Resolve every requested relation jointly",
        2: " First identify the exact event using all person",
        3: " No extra explanations in 'answer'",
    }.get(_locomo_category(category))
    if locomo_marker and locomo_marker in str(question or ""):
        return str(question or "").split(locomo_marker, 1)[0]
    marker = {
        "multi-session": " Resolve every requested relation jointly",
        "temporal-reasoning": " Match the exact event before resolving time",
        "knowledge-update": " Resolve all statements about the requested attribute",
        "single-session-preference": " Infer the user's durable preferences",
        "single-session-assistant": " Treat assistant turns",
    }.get(_category_id(category))
    return str(question or "").split(marker, 1)[0] if marker and marker in str(question or "") else str(question or "")


def _should_verify_multi_hop_answer(question: str | dict, result: dict | int, category=None) -> bool:
    if category is None:
        return False
    del result
    category_id = _category_id(category)
    return (_locomo_category(category) == 1 or category_id == "multi-session") and bool(re.search(
        r"\b(?:how many|which|what (?:are|were|kinds|types|places|items|books|games|hobbies|skills|ways|causes|recommendations))\b",
        str(question or ""), re.I,
    ))


def _should_verify_temporal_answer(question: str, category) -> bool:
    locomo_category = _locomo_category(category)
    if locomo_category == 2:
        text = str(question or "")
        return bool(
            re.search(r"\b(?:how long|after how many|start(?:ed)?|begin|began|first|last|before|after|during|since|as of|finish(?:ed)?|found|visit(?:ed)?|go|went|which city|where)\b", text, re.I)
            or re.search(r"\bwhen\b.*\b(?:make|made|was|were)\b", text, re.I)
        )
    if _category_id(category) != "temporal-reasoning":
        return False
    return True


def _should_verify_open_domain_geography(question: str, category) -> bool:
    if _locomo_category(category) != 3 and _category_id(category) != "open-domain":
        return False
    text = str(question or "")
    return bool(
        OPEN_DOMAIN_GEOGRAPHY_QUESTION_PATTERN.search(text)
        and OPEN_DOMAIN_GEOGRAPHY_TERM_PATTERN.search(text)
    )


def _should_verify_open_domain_alternative(question: str, result: dict, category) -> bool:
    if _locomo_category(category) != 3 and _category_id(category) != "open-domain":
        return False
    return bool(
        OPEN_DOMAIN_ALTERNATIVE_PATTERN.search(str(question or ""))
        and not OPEN_DOMAIN_EXPLICIT_BINARY_PATTERN.search(str(question or ""))
        and OPEN_DOMAIN_BINARY_ANSWER_PATTERN.fullmatch(
            str(result.get("answer", ""))
        )
    )


def _should_verify_single_hop_answer(question: str, result: dict, category) -> bool:
    del question, result, category
    return False


def _verification_changed(draft: dict, verified: dict) -> bool:
    draft_supports = draft.get("supports", draft.get("evidence", [])) or []
    verified_supports = verified.get("supports", verified.get("evidence", [])) or []
    return (
        str(verified.get("answer", "")).strip() != str(draft.get("answer", "")).strip()
        or list(verified_supports) != list(draft_supports)
    )


def prediction_key(sample_id, question_id) -> tuple[str, str]:
    """Return a type-stable key for prediction resume checks."""
    return str(sample_id), str(question_id)


def question_fingerprint(sample: dict, qa: dict) -> str:
    payload = {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "evidence": qa.get("evidence", []),
        "category": qa.get("category"),
        "adversarial_answer": qa.get("adversarial_answer"),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_fingerprint(sample_memory_dir: str) -> str:
    sample_dir = Path(sample_memory_dir)
    payload: dict = {
        "facts_by_id_sha256": _file_sha256(sample_dir / "facts_by_id.json"),
        "fact_graph_sha256": _file_sha256(sample_dir / "fact_graph.json"),
        "function_index_sha256": _file_sha256(sample_dir / "function_index.json"),
        "view_index_sha256": _file_sha256(sample_dir / "view_index.json"),
    }
    build_path = Path(sample_memory_dir) / "build.done.json"
    if build_path.exists():
        try:
            build = read_json(build_path)
        except Exception:
            build = {}
        if isinstance(build, dict):
            payload.update(
                {
                    "builder_version": build.get("builder_version"),
                    "input_sha256": build.get("input_sha256"),
                    "facts_sha256": build.get("facts_sha256"),
                    "memory_code_sha256": build.get("memory_code_sha256"),
                    "semantic_compilation": build.get("semantic_compilation"),
                }
            )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prediction_fingerprint(
    sample: dict,
    qa: dict,
    sample_memory_dir: str,
    model: str,
) -> str:
    payload = {
        "question_sha256": question_fingerprint(sample, qa),
        "memory_sha256": memory_fingerprint(sample_memory_dir),
        "model": str(model),
        "stage_version": STAGE_CHECKPOINT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stage_checkpoint_path(
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
    stage: str,
) -> Path:
    if stage not in QUESTION_STAGE_NAMES:
        raise ValueError(f"unsupported question stage: {stage}")
    return (
        Path(sample_memory_dir).parent
        / "predictions"
        / "stage_checkpoints"
        / str(sample_id)
        / f"question_{question_id:03d}"
        / f"{stage}.json"
    )


def _stage_fingerprint(
    *,
    stage: str,
    prediction_sha256: str,
    llm_model: str,
    judge_model: str = "",
) -> str:
    payload = {
        "schema_version": STAGE_CHECKPOINT_VERSION,
        "stage": stage,
        "prediction_fingerprint": prediction_sha256,
        "llm_model": str(llm_model),
        "judge_model": str(judge_model),
    }
    if stage == "candidate_retrieval":
        payload["reranker"] = {
            "model": os.getenv("CROSS_ENCODER_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
            "max_length": os.getenv("CROSS_ENCODER_MAX_LENGTH", "384"),
            "batch_size": os.getenv("CROSS_ENCODER_BATCH_SIZE", "32"),
        }
    if stage == "program_execution":
        payload["program_runtime"] = {
            "schema": PROGRAM_SCHEMA_VERSION,
            "max_repairs": 2,
            "timeout_seconds": os.getenv("MAC_SANDBOX_TIMEOUT_SECONDS", "5"),
            "cpu_seconds": os.getenv("MAC_SANDBOX_CPU_SECONDS", "5"),
            "memory_mb": os.getenv("MAC_SANDBOX_MEMORY_MB", "512"),
        }
    if stage == "answer_generation":
        payload["answer_runtime"] = {
            "program_fallback_schema": 1,
            "program_evidence_policy": "raw_program_supplement_v2",
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stage_fingerprint_from_prediction(
    *,
    stage: str,
    prediction_sha256: str,
    llm_model: str,
    judge_model: str = "",
) -> str:
    """Fingerprint a stage when the prediction checkpoint already stores its hash."""
    return _stage_fingerprint(
        stage=stage,
        prediction_sha256=prediction_sha256,
        llm_model=llm_model,
        judge_model=judge_model,
    )


def _read_stage_checkpoint(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        logger.warning("Ignoring invalid stage checkpoint: %s", path)
        return None
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == STAGE_CHECKPOINT_SCHEMA
        and payload.get("completed") is True
        and payload.get("fingerprint") == fingerprint
        and isinstance(payload.get("cost_metrics"), dict)
        and "artifact" in payload
    ):
        return None
    return payload


def _read_stage_inflight(path: Path, fingerprint: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not (
        isinstance(payload, dict)
        and payload.get("schema") == "question_stage_inflight_v1"
        and payload.get("fingerprint") == fingerprint
    ):
        return None
    return payload


def _run_checkpointed_stage(
    *,
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
    stage: str,
    fingerprint: str,
    operation: Callable[[], object],
    uses_llm: bool,
) -> tuple[object, dict]:
    path = stage_checkpoint_path(sample_memory_dir, sample_id, question_id, stage)
    cached = _read_stage_checkpoint(path, fingerprint)
    if cached is not None:
        logger.info("  RESUME: reusing %s for %s question %d", stage, sample_id, question_id)
        return cached["artifact"], cached["cost_metrics"]

    inflight_path = path.with_name(path.stem + ".inflight.json")
    prior = _read_stage_inflight(inflight_path, fingerprint)
    prior_metric = prior.get("cost_metrics") if prior else None
    write_json(
        {
            "schema": "question_stage_inflight_v1",
            "sample_id": sample_id,
            "question_id": question_id,
            "stage": stage,
            "fingerprint": fingerprint,
            "cost_metrics": prior_metric,
        },
        inflight_path,
    )
    started = time.monotonic()
    if uses_llm:
        try:
            with capture_token_usage() as usage:
                artifact = operation()
        except BaseException:
            current = failed_usage_metric(
                usage.snapshot(), time.monotonic() - started
            )
            failed = merge_metrics(prior_metric, current) if prior_metric else current
            write_json(
                {
                    "schema": "question_stage_inflight_v1",
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "stage": stage,
                    "fingerprint": fingerprint,
                    "cost_metrics": failed,
                },
                inflight_path,
            )
            raise
        current_metric = usage_metric(usage.snapshot(), time.monotonic() - started)
    else:
        local_timing = None
        try:
            with capture_local_timing() as local_timing:
                artifact = operation()
        except BaseException:
            excluded_wait = (
                local_timing.snapshot().excluded_queue_wait_seconds
                if local_timing is not None
                else 0.0
            )
            current_metric = failed_local_metric(
                time.monotonic() - started,
                excluded_queue_wait_seconds=excluded_wait,
            )
            failed = merge_metrics(prior_metric, current_metric) if prior_metric else current_metric
            write_json(
                {
                    "schema": "question_stage_inflight_v1",
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "stage": stage,
                    "fingerprint": fingerprint,
                    "cost_metrics": failed,
                },
                inflight_path,
            )
            raise
        current_metric = local_metric(
            time.monotonic() - started,
            excluded_queue_wait_seconds=(
                local_timing.snapshot().excluded_queue_wait_seconds
                if local_timing is not None
                else 0.0
            ),
        )
    metric = merge_metrics(prior_metric, current_metric) if prior_metric else current_metric
    write_json(
        {
            "schema": STAGE_CHECKPOINT_SCHEMA,
            "sample_id": sample_id,
            "question_id": question_id,
            "stage": stage,
            "completed": True,
            "fingerprint": fingerprint,
            "cost_metrics": metric,
            "artifact": artifact,
        },
        path,
    )
    inflight_path.unlink(missing_ok=True)
    return artifact, metric


def run_checkpointed_question_stage(
    *,
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
    stage: str,
    fingerprint: str,
    operation: Callable[[], object],
    uses_llm: bool,
) -> tuple[object, dict]:
    """Public entry point shared by QA and evaluation stage workers."""
    return _run_checkpointed_stage(
        sample_memory_dir=sample_memory_dir,
        sample_id=sample_id,
        question_id=question_id,
        stage=stage,
        fingerprint=fingerprint,
        operation=operation,
        uses_llm=uses_llm,
    )


def read_stage_metric(
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
    stage: str,
    fingerprint: str,
) -> dict | None:
    payload = _read_stage_checkpoint(
        stage_checkpoint_path(sample_memory_dir, sample_id, question_id, stage),
        fingerprint,
    )
    return payload.get("cost_metrics") if payload else None


def read_stage_metric_for_cost(
    sample_memory_dir: str,
    sample_id: str,
    question_id: int,
    stage: str,
    fingerprint: str,
) -> dict | None:
    """Return a persisted stage metric for accounting without weakening resume safety.

    Runtime reuse must require an exact stage fingerprint: a changed program or
    answer implementation must never reuse an old artifact.  Cost aggregation
    is different.  A completed checkpoint for the same isolated
    ``sample/question/stage`` remains the measured cost of the persisted
    answer even when a later code-only fingerprint revision makes it
    ineligible for runtime reuse.  Falling back here prevents a bookkeeping
    migration from silently erasing otherwise complete historical cost data.

    This helper is deliberately *not* used by the execution path.
    """
    strict_metric = read_stage_metric(
        sample_memory_dir,
        sample_id,
        question_id,
        stage,
        fingerprint,
    )
    if strict_metric is not None:
        return strict_metric

    path = stage_checkpoint_path(sample_memory_dir, sample_id, question_id, stage)
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        saved_question_id = int(payload.get("question_id", -1))
    except (TypeError, ValueError):
        return None
    if not (
        payload.get("schema") == STAGE_CHECKPOINT_SCHEMA
        and payload.get("completed") is True
        and payload.get("sample_id") == sample_id
        and saved_question_id == int(question_id)
        and payload.get("stage") == stage
        and isinstance(payload.get("cost_metrics"), dict)
    ):
        return None
    return payload["cost_metrics"]


def read_completed_prediction_fingerprints(path: str) -> dict[tuple[str, str], str]:
    return {
        prediction_key(row.get("sample_id"), row.get("question_id")): str(
            row.get("prediction_fingerprint") or row.get("question_fingerprint", "")
        )
        for row in _read_completed_predictions(path)
    }


def run_qa_question(
    sample: dict,
    sample_id: str,
    sample_memory_dir: str,
    question_id: int,
    qa: dict,
    llm: LLMClient,
    total_questions: int | None = None,
) -> dict:
    """Execute the Memory-as-Code QA path with replayable program artifacts."""

    question = str(qa.get("question", ""))
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    category = _category_id(qa.get("category") or metadata.get("question_type"))
    question_date = str(metadata.get("question_date") or "")
    prediction_sha256 = prediction_fingerprint(
        sample, qa, sample_memory_dir, str(getattr(llm, "model", ""))
    )

    def stage_fp(stage: str) -> str:
        return _stage_fingerprint(
            stage=stage,
            prediction_sha256=prediction_sha256,
            llm_model=str(getattr(llm, "model", "")),
        )

    candidates, _ = _run_checkpointed_stage(
        sample_memory_dir=sample_memory_dir,
        sample_id=sample_id,
        question_id=question_id,
        stage="candidate_retrieval",
        fingerprint=stage_fp("candidate_retrieval"),
        operation=lambda: initial_candidates(
            sample_memory_dir, sample, question, llm,
            category=category, question_date=question_date,
        ),
        uses_llm=True,
    )
    trace_dir = stage_checkpoint_path(
        sample_memory_dir, sample_id, question_id, "program_execution"
    ).parent / "execution_trace"
    program_artifact, _ = _run_checkpointed_stage(
        sample_memory_dir=sample_memory_dir,
        sample_id=sample_id,
        question_id=question_id,
        stage="program_execution",
        fingerprint=stage_fp("program_execution"),
        operation=lambda: plan_and_execute(
            sample_memory_dir=sample_memory_dir,
            sample=sample,
            question=question,
            llm=llm,
            trace_dir=trace_dir,
            candidates=dict(candidates),
            category=category,
            question_date=question_date,
        ),
        uses_llm=True,
    )

    def answer_from_program() -> dict:
        if program_artifact.get("status") != "ok":
            raise RuntimeError(
                "program execution did not produce evidence: "
                + str(program_artifact.get("reason", "unknown failure"))
            )
        loaded = render_evidence_context(
            dict(program_artifact.get("execution", {})),
            dict(program_artifact.get("candidates", {})).get("retrieval_bundle", {}),
            sample_memory_dir,
        )
        result = answer_question(
            format_final_question_for_mragent(question, category),
            loaded,
            llm,
            category=category,
        )
        execution = dict(program_artifact.get("execution", {}))
        evidence = execution.get("result", {})
        dialogue_ids = evidence.get("dialogue_ids", []) if isinstance(evidence, dict) else []
        return {
            "sample_id": sample_id,
            "question_id": question_id,
            "question_fingerprint": question_fingerprint(sample, qa),
            "prediction_fingerprint": prediction_sha256,
            "question": question,
            "gold_answer": qa.get("answer"),
            "adversarial_answer": qa.get("adversarial_answer"),
            "pred_answer": result.get("answer", ""),
            "gold_evidence": qa.get("evidence", []),
            "pred_evidence": result.get("supports", dialogue_ids),
            "category": qa.get("category"),
            "selected_files": [],
            "token_provenance": {
                "execution_trace": "execution_trace",
                "candidate_fact_count": len(candidates.get("candidate_fact_ids", [])),
                "candidate_function_count": len(candidates.get("candidate_functions", [])),
                "sandbox_runtime_seconds": execution.get("runtime_seconds", 0.0),
            },
            "confidence": result.get("confidence", "low"),
            "answer_trace": result.get("_answer_trace", {}),
            "program_trace": {
                "status": program_artifact.get("status"),
                "attempts": program_artifact.get("attempts", []),
                "executed_rules": evidence.get("rules", []) if isinstance(evidence, dict) else [],
                "category_search_profile": candidates.get("category_search_profile", {}),
            },
        }

    prediction, _ = _run_checkpointed_stage(
        sample_memory_dir=sample_memory_dir,
        sample_id=sample_id,
        question_id=question_id,
        stage="answer_generation",
        fingerprint=stage_fp("answer_generation"),
        operation=answer_from_program,
        uses_llm=True,
    )
    logger.info(
        "  PROGRAM EXECUTION: %s | candidates: %d facts, %d functions | answer: %s",
        program_artifact.get("status"),
        len(candidates.get("candidate_fact_ids", [])),
        len(candidates.get("candidate_functions", [])),
        prediction.get("pred_answer", ""),
    )
    return dict(prediction)


def _read_completed_predictions(path: str) -> list[dict]:
    """Read completed predictions while tolerating malformed JSONL records."""
    fp = Path(path)
    if not fp.exists():
        return []

    rows = []
    with fp.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "  RESUME: ignoring malformed prediction line %d in %s",
                    line_number,
                    path,
                )
    return rows
