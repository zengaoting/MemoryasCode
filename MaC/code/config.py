from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .benchmark import set_dataset


PROVIDERS = ("qwen", "deepseek", "gpt")
DATASETS = ("locomo", "longmemeval")


@dataclass
class Config:
    api_key: str
    base_url: str
    model: str
    judge_api_key: str
    judge_base_url: str
    judge_model: str
    data_path: str
    output_dir: str
    llm_provider: str = "qwen"
    dataset: str = "longmemeval"
    build_max_workers: int = 10
    semantic_max_workers: int = 4
    semantic_request_timeout: float = 300.0
    semantic_api_max_retries: int = 1
    qa_max_workers: int = 10
    judge_max_workers: int = 10
    llm_max_concurrency: int = 10
    max_samples: int | None = None
    samples_per_category: int | None = None
    max_questions_per_sample: int | None = None


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return _positive_int_env(name, 1)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _provider_settings(provider: str) -> tuple[str, str, str]:
    if provider == "qwen":
        return os.getenv("QWEN_API_KEY", ""), os.getenv("QWEN_BASE_URL", "https://api.openai.com/v1"), os.getenv("QWEN_MODEL", "qwen3.6-27b-awq2")
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY", ""), os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if provider == "gpt":
        # The GPT provider always uses the official OpenAI API; it never
        # reads Codex authentication files or CODEX_* environment variables.
        return os.getenv("OPENAI_API_KEY", ""), "https://api.openai.com/v1", os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def load_config(*, provider: str = "qwen", dataset: str = "longmemeval", model: str | None = None) -> Config:
    if provider not in PROVIDERS:
        raise ValueError(f"--llm must be one of: {', '.join(PROVIDERS)}")
    if dataset not in DATASETS:
        raise ValueError(f"--dataset must be one of: {', '.join(DATASETS)}")
    load_dotenv(".env")
    set_dataset(dataset)
    api_key, base_url, default_model = _provider_settings(provider)
    selected_model = str(model or default_model).strip()
    if not selected_model:
        raise ValueError("A model must be supplied with --model or its provider environment variable")
    data_name = "locomo10.json" if dataset == "locomo" else "dataset_LM.json"
    return Config(
        llm_provider=provider, dataset=dataset, api_key=api_key, base_url=base_url, model=selected_model,
        judge_api_key=os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        judge_base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        judge_model=os.getenv("JUDGE_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        data_path=os.getenv("DATA_PATH", str(Path("data") / data_name)),
        output_dir=os.getenv("OUTPUT_DIR", str(Path("outputs") / dataset / provider)),
        build_max_workers=_positive_int_env("MAC_BUILD_MAX_WORKERS", 10),
        semantic_max_workers=_positive_int_env("MAC_SEMANTIC_MAX_WORKERS", 4),
        semantic_request_timeout=_positive_float_env("SEMANTIC_REQUEST_TIMEOUT", 300.0),
        semantic_api_max_retries=_positive_int_env("SEMANTIC_API_MAX_RETRIES", 1),
        qa_max_workers=_positive_int_env("MAC_QA_MAX_WORKERS", 10),
        judge_max_workers=_positive_int_env("MAC_JUDGE_MAX_WORKERS", 10),
        llm_max_concurrency=_positive_int_env("MAC_LLM_MAX_CONCURRENCY", 10),
        max_samples=_optional_positive_int_env("MAX_SAMPLES"),
        samples_per_category=_optional_positive_int_env("SAMPLES_PER_CATEGORY"),
        max_questions_per_sample=_optional_positive_int_env("MAX_QUESTIONS_PER_SAMPLE"),
    )
