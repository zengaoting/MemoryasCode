"""Evaluation metrics aligned with MRAgent: stemmer-based F1 + LLM-as-judge."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import string
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import regex
from nltk.stem import PorterStemmer
from tqdm import tqdm

from .benchmark import LM_CATEGORIES, LM_CATEGORY_NAMES
from .llm_client import LLMClient
from .qa_engine import (
    read_stage_metric,
    run_checkpointed_question_stage,
    stage_fingerprint_from_prediction,
)
from .utils import read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stemmer (shared across all F1 calls)
# ---------------------------------------------------------------------------
_ps = PorterStemmer()


# ---------------------------------------------------------------------------
# Answer normalisation (exactly as in MRAgent)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Normalise an answer string: remove commas, articles, punctuation, lowercase, collapse whitespace."""
    s = str(s).replace(",", "")

    def remove_articles(text: str) -> str:
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ---------------------------------------------------------------------------
# Token-level F1 (Porter stemmer, same as MRAgent)
# ---------------------------------------------------------------------------
def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth after stemming."""
    prediction_tokens = [_ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [_ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# LLM-as-Judge (aligned with MRAgent eval/judge.py)
# ---------------------------------------------------------------------------
ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    llm,  # LLMClient instance
    max_retries: int = 3,
) -> int:
    """Score the generated answer against gold using an LLM judge. Returns 1 (CORRECT) or 0 (WRONG)."""
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
    retry_prompt = prompt
    last_label = ""
    for _ in range(max_retries):
        result = llm.chat_json_mode(system="", user=retry_prompt, temperature=0.0)
        label = str(result.get("label", "")).strip().upper()
        if label in {"CORRECT", "WRONG"}:
            return 1 if label == "CORRECT" else 0
        if hasattr(llm, "discard_last_response_usage"):
            llm.discard_last_response_usage()
        last_label = label
        retry_prompt = (
            prompt
            + "\n\nYour previous JSON did not contain a valid label. "
            + 'Return exactly {"label": "CORRECT"} or {"label": "WRONG"}.'
        )
    raise ValueError(f"LLM judge returned invalid label after retries: {last_label!r}")


# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------
def _category_id(value) -> str:
    return str(value or "").strip()


def _is_evaluated_category(category) -> bool:
    return _category_id(category) in LM_CATEGORIES


def _normalize_answer_for_eval(ans) -> str:
    """Convert answer to string (handles lists and numbers from the dataset)."""
    if isinstance(ans, list):
        return " ".join(str(a) for a in ans)
    return str(ans)


def _build_table_metrics(per_category: dict, judge_result: dict | None) -> dict:
    """Return LongMemEval table-ready percentages for all six categories."""
    judge_by_cat = (judge_result or {}).get("per_category", {})
    table_metrics = {}
    for category in LM_CATEGORIES:
        f1_entry = per_category.get(category)
        if not f1_entry:
            continue
        values = {"f1": round(float(f1_entry["f1"]) * 100, 2)}
        judge_entry = judge_by_cat.get(category)
        if judge_entry:
            values["binary_llm_judge"] = round(float(judge_entry["accuracy"]) * 100, 2)
        table_metrics[category] = values
    return table_metrics


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def evaluate_predictions(
    predictions_path: str,
    llm=None,                # optional LLMClient for judge
    judge_out_path: str = "",  # where to write judge JSONL results
    judge_workers: int = 8,
    generation_model: str = "",
) -> dict:
    """Run F1 (and optionally LLM judge) evaluation over predictions, aligned with MRAgent."""
    raw_preds = read_jsonl(predictions_path)
    raw_prediction_count = len(raw_preds)
    excluded_category5_rows = sum(
        1 for row in raw_preds if str(row.get("category", "")).strip() == "5"
    )
    if not raw_preds:
        logger.warning("No predictions found at %s", predictions_path)
        return {}
    raw_preds = _dedupe_predictions(raw_preds)
    preds = [r for r in raw_preds if _is_evaluated_category(r.get("category", 0))]
    if not preds:
        logger.warning("No LongMemEval predictions found at %s", predictions_path)
        return {
            "total_prediction_rows_raw": raw_prediction_count,
            "total_questions": 0,
            "excluded_rows": raw_prediction_count,
            "excluded_category5_rows": excluded_category5_rows,
            "evaluated_categories": list(LM_CATEGORIES),
            "overall_f1": 0.0,
            "per_category": {},
            "judge": None,
            "table_metrics": {},
        }

    # ── F1 by category ────────────────────────────────────────────────
    f1_by_cat: dict[str, list[float]] = defaultdict(list)

    for r in preds:
        category = _category_id(r.get("category", ""))
        prediction = _normalize_answer_for_eval(r.get("pred_answer", ""))
        reference = _normalize_answer_for_eval(r.get("gold_answer", ""))
        f1_by_cat[category].append(f1_score(prediction, reference))

    # ── Print F1 results ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (aligned with MRAgent)")
    logger.info("=" * 60)
    logger.info("Total prediction rows: %d", raw_prediction_count)
    logger.info("Excluded rows with unknown categories: %d", raw_prediction_count - len(preds))
    logger.info("Total LongMemEval questions: %d", len(preds))

    print("\n== F1 by category ==")
    per_category = {}
    all_f1 = []
    for cat in LM_CATEGORIES:
        if cat not in f1_by_cat:
            continue
        vals = f1_by_cat[cat]
        avg = sum(vals) / len(vals)
        all_f1.extend(vals)
        cat_name = LM_CATEGORY_NAMES[cat]
        print(f"  {cat} ({cat_name}): n={len(vals)} F1={avg:.4f}")
        per_category[str(cat)] = {
            "name": cat_name,
            "count": len(vals),
            "f1": avg,
        }
    overall_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    print(f"  OVERALL: n={len(all_f1)} F1={overall_f1:.4f}")

    # ── LLM Judge (optional) ──────────────────────────────────────────
    judge_result = None
    if llm is not None:
        out_path = judge_out_path or "result_judge.jsonl"
        judge_result = _evaluate_with_llm_judge(
            preds,
            llm,
            out_path,
            judge_workers,
            predictions_path,
            generation_model,
        )

    return {
        "total_prediction_rows_raw": raw_prediction_count,
        "total_questions": len(preds),
        "excluded_rows": raw_prediction_count - len(preds),
        "excluded_category5_rows": excluded_category5_rows,
        "evaluated_categories": list(LM_CATEGORIES),
        "overall_f1": overall_f1,
        "per_category": per_category,
        "judge": judge_result,
        "table_metrics": _build_table_metrics(per_category, judge_result),
    }


class _ThreadLocalJudgeClient:
    """Provide one real LLMClient per judge worker thread."""

    def __init__(self, source) -> None:
        self._source = source
        self._state = threading.local()
        self._created = []
        self._created_lock = threading.Lock()
        self._clone_args = None
        if isinstance(source, LLMClient):
            self._clone_args = (
                str(source.client.api_key),
                str(source.client.base_url),
                source.model,
            )

    def get(self):
        client = getattr(self._state, "client", None)
        if client is not None:
            return client
        if self._clone_args is None:
            client = self._source
        else:
            client = type(self._source)(*self._clone_args)
            with self._created_lock:
                self._created.append(client)
        self._state.client = client
        return client

    def close(self) -> None:
        for client in self._created:
            close = getattr(client, "close", None)
            if callable(close):
                close()
                continue
            underlying = getattr(client, "client", None)
            close = getattr(underlying, "close", None)
            if callable(close):
                close()


def _evaluate_with_llm_judge(
    predictions: list[dict],
    llm,
    out_path: str,
    judge_workers: int,
    predictions_path: str,
    generation_model: str,
) -> dict:
    worker_count = _validate_worker_count(judge_workers)
    client_provider = _ThreadLocalJudgeClient(llm)
    expected_by_key = {_prediction_judge_key(row): row for row in predictions}
    completed_rows = _prepare_judge_cache(out_path, expected_by_key)
    checkpointed_rows = [
        row
        for row in completed_rows
        if _evaluation_checkpoint_complete(
            expected_by_key[_saved_judge_key(row)],
            predictions_path,
            generation_model,
            str(getattr(llm, "model", "")),
        )
    ]
    if len(checkpointed_rows) != len(completed_rows):
        _rewrite_jsonl(Path(out_path), checkpointed_rows)
        logger.warning(
            "Discarded %d judge cache row(s) without matching evaluation checkpoints",
            len(completed_rows) - len(checkpointed_rows),
        )
    completed_rows = checkpointed_rows
    completed_keys = {_saved_judge_key(row) for row in completed_rows}

    judge_by_cat: dict[str, list[int]] = defaultdict(list)
    for row in completed_rows:
        judge_by_cat[_category_id(row["category"])].append(int(row["llm_score"]))

    pending_rows = []
    scheduled_keys = set(completed_keys)
    for row in predictions:
        key = _prediction_judge_key(row)
        if key in scheduled_keys:
            continue
        scheduled_keys.add(key)
        pending_rows.append(row)

    failures = []
    new_completed = 0
    if pending_rows:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="longmemeval-judge",
                ) as executor,
                output.open("a", encoding="utf-8") as handle,
            ):
                future_to_row = {
                    executor.submit(
                        _judge_prediction,
                        row,
                        client_provider,
                        predictions_path,
                        generation_model,
                    ): row
                    for row in pending_rows
                }
                progress = tqdm(
                    as_completed(future_to_row),
                    desc="LongMemEval LLM judge",
                    unit="q",
                    initial=len(completed_rows),
                    total=len(completed_rows) + len(pending_rows),
                    dynamic_ncols=True,
                )
                for future in progress:
                    source_row = future_to_row[future]
                    try:
                        judge_row = future.result()
                    except Exception as exc:
                        failure = {
                            "sample_id": str(source_row.get("sample_id", "")),
                            "question_id": str(source_row.get("question_id", "")),
                            "category": _category_id(source_row.get("category")),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        failures.append(failure)
                        logger.exception(
                            "LLM judge failed for sample=%s question_id=%s",
                            failure["sample_id"],
                            failure["question_id"],
                        )
                        continue

                    handle.write(json.dumps(judge_row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    category = _category_id(judge_row["category"])
                    judge_by_cat[category].append(int(judge_row["llm_score"]))
                    new_completed += 1
        finally:
            client_provider.close()
    elif completed_rows:
        logger.info("LLM judge cache is complete; no new judge calls needed.")

    print("\n== LLM-judge accuracy by category ==")
    per_category = {}
    total_ok = 0
    total = 0
    for category in LM_CATEGORIES:
        values = judge_by_cat.get(category, [])
        if not values:
            continue
        correct = sum(values)
        accuracy = correct / len(values)
        total_ok += correct
        total += len(values)
        per_category[str(category)] = {
            "count": len(values),
            "correct": correct,
            "accuracy": accuracy,
        }
        print(
            f"  {category} ({LM_CATEGORY_NAMES[category]}): "
            f"n={len(values)} acc={accuracy:.4f}"
        )
    if total:
        print(f"  OVERALL: {total_ok}/{total} = {total_ok / total:.4f}")

    expected_total = len(expected_by_key)
    return {
        "per_category": per_category,
        "correct": total_ok,
        "total": total,
        "expected_total": expected_total,
        "overall_accuracy": total_ok / total if total else 0.0,
        "cached": len(completed_rows),
        "newly_completed": new_completed,
        "failed": len(failures),
        "failures": failures,
        "complete": total == expected_total and not failures,
        "workers": worker_count,
        "output_path": str(out_path),
    }


def _judge_prediction(
    row: dict,
    client_provider,
    predictions_path: str,
    generation_model: str,
) -> dict:
    category = _category_id(row.get("category", ""))
    question = _normalize_answer_for_eval(row.get("question", ""))
    gold = _normalize_answer_for_eval(row.get("gold_answer", ""))
    prediction = _normalize_answer_for_eval(row.get("pred_answer", ""))
    sample_id = str(row.get("sample_id", ""))
    question_id = int(row.get("question_id", 0))
    judge_model = str(getattr(client_provider._source, "model", ""))
    sample_memory_dir = str(_evaluation_output_dir(predictions_path) / sample_id)
    prediction_sha256 = _stored_prediction_fingerprint(row)
    fingerprint = stage_fingerprint_from_prediction(
        stage="evaluation",
        prediction_sha256=prediction_sha256,
        llm_model=generation_model,
        judge_model=judge_model,
    )

    def evaluate_one() -> dict:
        local_f1 = f1_score(prediction, gold)
        score = evaluate_llm_judge(
            question,
            gold,
            prediction,
            client_provider.get(),
        )
        return {
            "llm_score": score,
            "f1": local_f1,
            "question": question,
            "prediction": prediction,
            "reference": gold,
            "category": category,
            "sample": sample_id,
            "question_id": row.get("question_id", ""),
            "prediction_fingerprint": prediction_sha256,
        }

    artifact, _ = run_checkpointed_question_stage(
        sample_memory_dir=sample_memory_dir,
        sample_id=sample_id,
        question_id=question_id,
        stage="evaluation",
        fingerprint=fingerprint,
        operation=evaluate_one,
        uses_llm=True,
    )
    return dict(artifact)


def _evaluation_output_dir(predictions_path: str) -> Path:
    path = Path(predictions_path)
    return path.parent.parent if path.parent.name == "predictions" else path.parent


def _stored_prediction_fingerprint(row: dict) -> str:
    stored = str(row.get("prediction_fingerprint", "")).strip()
    if stored:
        return stored
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evaluation_checkpoint_complete(
    row: dict,
    predictions_path: str,
    generation_model: str,
    judge_model: str,
) -> bool:
    sample_id = str(row.get("sample_id", ""))
    question_id = int(row.get("question_id", 0))
    sample_memory_dir = str(_evaluation_output_dir(predictions_path) / sample_id)
    fingerprint = stage_fingerprint_from_prediction(
        stage="evaluation",
        prediction_sha256=_stored_prediction_fingerprint(row),
        llm_model=generation_model,
        judge_model=judge_model,
    )
    return read_stage_metric(
        sample_memory_dir,
        sample_id,
        question_id,
        "evaluation",
        fingerprint,
    ) is not None


def _prepare_judge_cache(out_path: str, expected_by_key: dict) -> list[dict]:
    """Keep only current unique judge rows and repair interrupted JSONL."""
    path = Path(out_path)
    if not path.exists():
        return []
    reusable = []
    reusable_keys = set()
    discarded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            discarded += 1
            continue
        try:
            row = json.loads(line)
            key = _saved_judge_key(row)
            source = expected_by_key.get(key)
            score = int(row.get("llm_score"))
            category = _category_id(row.get("category"))
            if score not in {0, 1}:
                raise ValueError("invalid score")
            if source is None or category != _category_id(source.get("category")):
                raise ValueError("stale cache row")
            if key in reusable_keys:
                raise ValueError("duplicate cache row")
        except (json.JSONDecodeError, TypeError, ValueError):
            discarded += 1
            continue
        reusable_keys.add(key)
        row["llm_score"] = score
        row["category"] = category
        reusable.append(row)

    if discarded:
        _rewrite_jsonl(path, reusable)
        logger.warning(
            "Repaired judge cache %s: retained %d row(s), discarded %d row(s)",
            path,
            len(reusable),
            discarded,
        )
    elif reusable:
        logger.info("Loaded %d reusable judge row(s) from %s", len(reusable), path)
    return reusable


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _dedupe_predictions(rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("sample_id", "")), str(row.get("question_id", "")))
        if key in latest:
            del latest[key]
        latest[key] = row
    discarded = len(rows) - len(latest)
    if discarded:
        logger.warning("Ignoring %d superseded prediction row(s)", discarded)
    return list(latest.values())


def _validate_worker_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"judge_workers must be an integer >= 1, got {value!r}")
    return value


def _prediction_judge_key(row: dict) -> tuple[str, str, str, str, str]:
    """Stable key used to resume LLM judge without duplicating completed rows."""
    return (
        str(row.get("sample_id", "")),
        str(row.get("question_id", "")),
        str(row.get("question", "")),
        _normalize_answer_for_eval(row.get("gold_answer", "")),
        _normalize_answer_for_eval(row.get("pred_answer", "")),
    )


def _saved_judge_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("sample", row.get("sample_id", ""))),
        str(row.get("question_id", "")),
        str(row.get("question", "")),
        _normalize_answer_for_eval(row.get("reference", row.get("gold_answer", ""))),
        _normalize_answer_for_eval(row.get("prediction", row.get("pred_answer", ""))),
    )
