from __future__ import annotations

import ipaddress
import json
import logging
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_PROXY_BYPASS_HOSTS = {"202.120.5.12"}
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_REQUEST_TIMEOUT = 1200.0
DEFAULT_JUDGE_REQUEST_TIMEOUT = 180.0
_concurrency_config_lock = threading.Lock()
_request_semaphore = threading.BoundedSemaphore(DEFAULT_MAX_CONCURRENCY)
_max_concurrency = DEFAULT_MAX_CONCURRENCY
_active_usage: ContextVar["UsageAccumulator | None"] = ContextVar(
    "active_llm_usage", default=None
)


@dataclass(frozen=True)
class UsageSnapshot:
    input_tokens: int | None
    attempts: int
    unknown_attempts: int
    usage_complete: bool
    accepted_attempts: int = 0
    accepted_runtime_seconds: float = 0.0
    failed_attempts: int = 0
    failed_runtime_seconds: float = 0.0
    retry_backoff_seconds: float = 0.0
    discarded_attempts: int = 0
    discarded_input_tokens: int = 0
    discarded_runtime_seconds: float = 0.0


class UsageAccumulator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted: list[tuple[int | None, float]] = []
        self._attempts = 0
        self._failed_attempts = 0
        self._failed_runtime_seconds = 0.0
        self._retry_backoff_seconds = 0.0
        self._discarded_attempts = 0
        self._discarded_input_tokens = 0
        self._discarded_runtime_seconds = 0.0

    def record(self, prompt_tokens: int | None, runtime_seconds: float = 0.0) -> None:
        runtime = max(0.0, float(runtime_seconds))
        with self._lock:
            self._attempts += 1
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0:
                normalized_tokens: int | None = prompt_tokens
            else:
                normalized_tokens = None
            self._accepted.append((normalized_tokens, runtime))

    def record_failure(self, runtime_seconds: float = 0.0) -> None:
        with self._lock:
            self._attempts += 1
            self._failed_attempts += 1
            self._failed_runtime_seconds += max(0.0, float(runtime_seconds))

    def record_retry_backoff(self, runtime_seconds: float) -> None:
        with self._lock:
            self._retry_backoff_seconds += max(0.0, float(runtime_seconds))

    def discard_last_success(self) -> bool:
        """Move the most recent response out of successful-only accounting."""
        with self._lock:
            if not self._accepted:
                return False
            prompt_tokens, runtime = self._accepted.pop()
            self._discarded_attempts += 1
            if prompt_tokens is not None:
                self._discarded_input_tokens += prompt_tokens
            self._discarded_runtime_seconds += runtime
            return True

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            unknown = sum(1 for tokens, _ in self._accepted if tokens is None)
            complete = unknown == 0
            input_tokens = sum(
                int(tokens) for tokens, _ in self._accepted if tokens is not None
            )
            return UsageSnapshot(
                input_tokens=input_tokens if complete else None,
                attempts=self._attempts,
                unknown_attempts=unknown,
                usage_complete=complete,
                accepted_attempts=len(self._accepted),
                accepted_runtime_seconds=sum(runtime for _, runtime in self._accepted),
                failed_attempts=self._failed_attempts,
                failed_runtime_seconds=self._failed_runtime_seconds,
                retry_backoff_seconds=self._retry_backoff_seconds,
                discarded_attempts=self._discarded_attempts,
                discarded_input_tokens=self._discarded_input_tokens,
                discarded_runtime_seconds=self._discarded_runtime_seconds,
            )


@contextmanager
def capture_token_usage():
    accumulator = UsageAccumulator()
    token = _active_usage.set(accumulator)
    try:
        yield accumulator
    finally:
        _active_usage.reset(token)


def _record_attempt(prompt_tokens: int | None, runtime_seconds: float = 0.0) -> None:
    accumulator = _active_usage.get()
    if accumulator is not None:
        accumulator.record(prompt_tokens, runtime_seconds)


def _record_failed_attempt(runtime_seconds: float = 0.0) -> None:
    accumulator = _active_usage.get()
    if accumulator is not None:
        accumulator.record_failure(runtime_seconds)


def _record_retry_backoff(runtime_seconds: float) -> None:
    accumulator = _active_usage.get()
    if accumulator is not None:
        accumulator.record_retry_backoff(runtime_seconds)


def _discard_last_successful_response() -> bool:
    accumulator = _active_usage.get()
    return accumulator.discard_last_success() if accumulator is not None else False


def _sleep_as_retry_backoff(delay: float) -> None:
    started = time.monotonic()
    time.sleep(delay)
    _record_retry_backoff(time.monotonic() - started)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(str(base_url))
    return (parsed.hostname or "").strip().lower()


def _extra_proxy_bypass_hosts() -> set[str]:
    raw = os.getenv("LLM_BYPASS_PROXY_HOSTS", "")
    return {host.strip().lower() for host in raw.split(",") if host.strip()}


def _is_local_or_internal_host(host: str) -> bool:
    if not host:
        return False
    if host in {"localhost", *DEFAULT_PROXY_BYPASS_HOSTS, *_extra_proxy_bypass_hosts()}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _should_bypass_proxy(base_url: str) -> bool:
    if os.getenv("LLM_BYPASS_PROXY") is not None:
        return _env_bool("LLM_BYPASS_PROXY", False)
    return _is_local_or_internal_host(_base_url_host(base_url))


def configure_max_concurrency(limit: int) -> None:
    """Set the process-wide cap for in-flight LLM network attempts."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"LLM max concurrency must be an integer >= 1, got {limit!r}")

    global _request_semaphore, _max_concurrency
    with _concurrency_config_lock:
        if limit == _max_concurrency:
            return
        _request_semaphore = threading.BoundedSemaphore(limit)
        _max_concurrency = limit


def _get_request_semaphore() -> threading.BoundedSemaphore:
    with _concurrency_config_lock:
        return _request_semaphore


class LLMClient:
    """OpenAI-compatible client used by Qwen and DeepSeek providers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        provider: str = "qwen",
        request_timeout: float | None = None,
        max_api_retries: int | None = None,
    ):
        if provider not in {"qwen", "deepseek"}:
            raise ValueError(f"LLMClient only supports OpenAI-compatible providers, got {provider!r}")
        if not api_key:
            raise ValueError(f"{provider.upper()} API key is required")
        self.provider = provider
        prefix = provider.upper()
        self.timeout = (
            float(request_timeout)
            if request_timeout is not None
            else _env_float(f"{prefix}_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
        )
        if self.timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.max_api_retries = (
            max(1, int(max_api_retries))
            if max_api_retries is not None
            else max(1, _env_int(f"{prefix}_API_MAX_RETRIES", 3))
        )
        self.retry_backoff = max(1.0, _env_float(f"{prefix}_API_RETRY_BACKOFF", 2.0))
        kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if _should_bypass_proxy(base_url):
            kwargs["http_client"] = httpx.Client(
                trust_env=False,
                timeout=self.timeout,
            )
            logger.info(
                "Bypassing proxy environment for LLM base_url host=%s",
                _base_url_host(base_url),
            )
        self.client = OpenAI(**kwargs)
        self.model = model

    def _generation_extra_body(self) -> dict:
        if self.provider == "qwen":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        # DeepSeek's OpenAI-compatible API uses this request extension.
        return {"thinking": {"type": "disabled"}}

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_format: Optional[dict] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        kwargs["extra_body"] = self._generation_extra_body()
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exc: Exception | None = None
        for attempt in range(1, self.max_api_retries + 1):
            request_started: float | None = None
            try:
                with _get_request_semaphore():
                    request_started = time.monotonic()
                    response = self.client.chat.completions.create(**kwargs)
                    request_runtime = time.monotonic() - request_started
                usage = getattr(response, "usage", None)
                _record_attempt(
                    getattr(usage, "prompt_tokens", None),
                    request_runtime,
                )
                return response.choices[0].message.content or ""
            except APIStatusError as exc:
                failed_runtime = (
                    time.monotonic() - request_started
                    if request_started is not None
                    else 0.0
                )
                _record_failed_attempt(failed_runtime)
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if (
                    status not in RETRYABLE_STATUS_CODES
                    or attempt >= self.max_api_retries
                ):
                    raise
                self._sleep_before_retry(attempt, f"status={status}")
            except (APITimeoutError, APIConnectionError) as exc:
                failed_runtime = (
                    time.monotonic() - request_started
                    if request_started is not None
                    else 0.0
                )
                _record_failed_attempt(failed_runtime)
                last_exc = exc
                if attempt >= self.max_api_retries:
                    raise
                self._sleep_before_retry(attempt, exc.__class__.__name__)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception")

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        delay = min(self.retry_backoff ** (attempt - 1), 60.0)
        delay += random.uniform(0.0, min(1.0, delay * 0.1))
        logger.warning(
            "%s request failed (%s), retrying %d/%d in %.1fs",
            self.provider,
            reason,
            attempt + 1,
            self.max_api_retries,
            delay,
        )
        _sleep_as_retry_backoff(delay)

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> Any:
        last_text = ""
        retry_user = user
        for _ in range(max_retries):
            text = self.chat(system, retry_user, temperature=temperature)
            last_text = text
            try:
                return parse_json_response(text)
            except Exception:
                self.discard_last_response_usage()
                retry_user = (
                    user
                    + "\n\nYour previous response was not valid JSON. "
                    + "Return valid JSON only."
                )
                _sleep_as_retry_backoff(1.0)
        raise ValueError(
            f"Failed to parse JSON after retries. Last response:\n{last_text}"
        )

    def chat_json_mode(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> dict:
        last_text = ""
        last_error: Exception | None = None
        retry_user = user
        for attempt in range(max_retries):
            received_response = False
            try:
                text = self.chat(
                    system,
                    retry_user,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                received_response = True
                last_text = text or ""
                if not last_text.strip():
                    raise ValueError("empty response")
                return parse_json_response(last_text)
            except Exception as exc:
                if received_response:
                    self.discard_last_response_usage()
                last_error = exc
                retry_user = (
                    user
                    + "\n\nYour previous response was not valid JSON. "
                    + 'Return valid JSON only, for example: {"label": "CORRECT"}.'
                )
                _sleep_as_retry_backoff(min(2**attempt, 8))
        raise ValueError(
            "Failed to parse JSON mode response after retries. "
            f"Last error: {last_error}. Last response:\n{last_text}"
        )

    def discard_last_response_usage(self) -> bool:
        """Exclude an application-rejected response from cost metrics."""
        return _discard_last_successful_response()


class OpenAIResponsesClient:
    """Official OpenAI Responses API client with reasoning fixed to off."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        request_timeout: float | None = None,
        max_api_retries: int | None = None,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for --llm gpt")
        self.timeout = float(request_timeout) if request_timeout is not None else _env_float("OPENAI_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
        if self.timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.max_api_retries = max(1, int(max_api_retries)) if max_api_retries is not None else max(1, _env_int("OPENAI_API_MAX_RETRIES", 3))
        self.retry_backoff = max(1.0, _env_float("OPENAI_API_RETRY_BACKOFF", 2.0))
        # Do not pass a base URL: this intentionally pins GPT to api.openai.com.
        self.client = OpenAI(api_key=api_key, timeout=self.timeout, max_retries=0)
        self.model = model
        self.provider = "gpt"

    def chat(self, system: str, user: str, temperature: float = 0.0, response_format: Optional[dict] = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": "none"},
            "temperature": temperature,
            "store": False,
        }
        if response_format is not None:
            kwargs["text"] = {"format": {"type": "json_object"}}
        last_exc: Exception | None = None
        for attempt in range(1, self.max_api_retries + 1):
            request_started: float | None = None
            try:
                with _get_request_semaphore():
                    request_started = time.monotonic()
                    response = self.client.responses.create(**kwargs)
                    runtime = time.monotonic() - request_started
                usage = getattr(response, "usage", None)
                _record_attempt(getattr(usage, "input_tokens", None), runtime)
                text = str(getattr(response, "output_text", "") or "")
                if not text:
                    raise ValueError("OpenAI Responses API returned no output text")
                return text
            except APIStatusError as exc:
                failed = time.monotonic() - request_started if request_started is not None else 0.0
                _record_failed_attempt(failed)
                last_exc = exc
                if getattr(exc, "status_code", None) not in RETRYABLE_STATUS_CODES or attempt >= self.max_api_retries:
                    raise
                self._sleep_before_retry(attempt, f"status={getattr(exc, 'status_code', None)}")
            except (APITimeoutError, APIConnectionError) as exc:
                failed = time.monotonic() - request_started if request_started is not None else 0.0
                _record_failed_attempt(failed)
                last_exc = exc
                if attempt >= self.max_api_retries:
                    raise
                self._sleep_before_retry(attempt, exc.__class__.__name__)
        raise last_exc or RuntimeError("OpenAI request failed without an exception")

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        delay = min(self.retry_backoff ** (attempt - 1), 60.0) + random.uniform(0.0, 1.0)
        logger.warning("OpenAI request failed (%s), retrying %d/%d in %.1fs", reason, attempt + 1, self.max_api_retries, delay)
        _sleep_as_retry_backoff(delay)

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_retries: int = 3) -> Any:
        last_text = ""
        retry_user = user
        for _ in range(max_retries):
            last_text = self.chat(system, retry_user, temperature=temperature)
            try:
                return parse_json_response(last_text)
            except Exception:
                self.discard_last_response_usage()
                retry_user = user + "\n\nYour previous response was not valid JSON. Return valid JSON only."
                _sleep_as_retry_backoff(1.0)
        raise ValueError(f"Failed to parse JSON after retries. Last response:\n{last_text}")

    def chat_json_mode(self, system: str, user: str, temperature: float = 0.0, max_retries: int = 3) -> dict:
        last_text = ""
        for attempt in range(max_retries):
            try:
                last_text = self.chat(system, user, temperature=temperature, response_format={"type": "json_object"})
                return parse_json_response(last_text)
            except Exception:
                self.discard_last_response_usage()
                user += "\n\nReturn valid JSON only, for example: {\"label\": \"CORRECT\"}."
                _sleep_as_retry_backoff(min(2**attempt, 8))
        raise ValueError(f"Failed to parse JSON mode response. Last response:\n{last_text}")

    def discard_last_response_usage(self) -> bool:
        return _discard_last_successful_response()


def create_llm_client(
    api_key: str, base_url: str, model: str, *, provider: str,
    request_timeout: float | None = None, max_api_retries: int | None = None,
) -> LLMClient | OpenAIResponsesClient:
    if provider == "gpt":
        return OpenAIResponsesClient(api_key, model, request_timeout=request_timeout, max_api_retries=max_api_retries)
    return LLMClient(api_key, base_url, model, provider=provider, request_timeout=request_timeout, max_api_retries=max_api_retries)


class OpenAIJudgeClient:
    """Existing OpenAI-compatible HTTP client retained for DeepSeek judging."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.timeout = _env_float("JUDGE_REQUEST_TIMEOUT", DEFAULT_JUDGE_REQUEST_TIMEOUT)
        self.max_api_retries = max(1, _env_int("JUDGE_API_MAX_RETRIES", 10))
        self.retry_backoff = max(1.0, _env_float("JUDGE_API_RETRY_BACKOFF", 2.0))
        kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if _should_bypass_proxy(base_url):
            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=self.timeout)
            logger.info("Bypassing proxy environment for LLM base_url host=%s", _base_url_host(base_url))
        self.client = OpenAI(**kwargs)
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.0,
             response_format: Optional[dict] = None) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        # Judge requests use the DeepSeek API by default and must not enable
        # its reasoning mode. This is intentionally fixed, not user-configured.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        last_exc: Exception | None = None
        for attempt in range(1, self.max_api_retries + 1):
            request_started: float | None = None
            try:
                # Limit only the network attempt. The context manager releases the
                # permit before exception handling enters retry backoff sleep.
                with _get_request_semaphore():
                    request_started = time.monotonic()
                    resp = self.client.chat.completions.create(**kwargs)
                    request_runtime = time.monotonic() - request_started
                usage = getattr(resp, "usage", None)
                _record_attempt(
                    getattr(usage, "prompt_tokens", None),
                    request_runtime,
                )
                return resp.choices[0].message.content or ""
            except APIStatusError as exc:
                failed_runtime = (
                    time.monotonic() - request_started
                    if request_started is not None
                    else 0.0
                )
                _record_failed_attempt(failed_runtime)
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status not in RETRYABLE_STATUS_CODES or attempt >= self.max_api_retries:
                    raise
                self._sleep_before_retry(attempt, f"status={status}")
            except (APITimeoutError, APIConnectionError) as exc:
                failed_runtime = (
                    time.monotonic() - request_started
                    if request_started is not None
                    else 0.0
                )
                _record_failed_attempt(failed_runtime)
                last_exc = exc
                if attempt >= self.max_api_retries:
                    raise
                self._sleep_before_retry(attempt, exc.__class__.__name__)

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception")

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        delay = min(self.retry_backoff ** (attempt - 1), 60.0)
        delay += random.uniform(0.0, min(1.0, delay * 0.1))
        logger.warning(
            "LLM request failed (%s), retrying %d/%d in %.1fs",
            reason,
            attempt + 1,
            self.max_api_retries,
            delay,
        )
        _sleep_as_retry_backoff(delay)

    def discard_last_response_usage(self) -> bool:
        """Exclude an application-rejected response from the main cost metric."""
        return _discard_last_successful_response()

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_retries: int = 3) -> Any:
        last_text = ""
        for attempt in range(max_retries):
            text = self.chat(system, user, temperature=temperature)
            last_text = text
            try:
                return parse_json_response(text)
            except Exception:
                self.discard_last_response_usage()
                user = user + "\n\nYour previous response was not valid JSON. Return valid JSON only."
                _sleep_as_retry_backoff(1.0)
        raise ValueError(f"Failed to parse JSON after retries. Last response:\n{last_text}")

    def chat_json_mode(self, system: str, user: str, temperature: float = 0.0, max_retries: int = 3) -> dict:
        """Chat with response_format=json_object for strict JSON output (for LLM judge)."""
        last_text = ""
        last_error: Exception | None = None
        retry_user = user
        for attempt in range(max_retries):
            received_response = False
            try:
                text = self.chat(
                    system,
                    retry_user,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                received_response = True
                last_text = text or ""
                if not last_text.strip():
                    raise ValueError("empty response")
                return parse_json_response(last_text)
            except Exception as exc:
                if received_response:
                    self.discard_last_response_usage()
                last_error = exc
                retry_user = (
                    user
                    + "\n\nYour previous response was not valid JSON. "
                    + 'Return valid JSON only, for example: {"label": "CORRECT"}.'
                )
                _sleep_as_retry_backoff(min(2 ** attempt, 8))
        raise ValueError(
            "Failed to parse JSON mode response after retries. "
            f"Last error: {last_error}. Last response:\n{last_text}"
        )


def extract_json(text: str) -> str:
    """Return the complete top-level JSON value embedded in an LLM response.

    Fact extraction is allowed to return a list of facts.  Do not strip an
    array's outer brackets: doing so turns a valid ``[{...}, {...}]`` response
    into invalid comma-separated objects.
    """
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return text
    start = min(starts)

    # Prefer a decoder boundary: this tolerates surrounding prose without
    # guessing where nested objects end.
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
        return text[start:end]
    except json.JSONDecodeError:
        # Preserve the full root container so the narrow field-quote repair in
        # parse_json_response can still recover slightly malformed JSON.
        # A truncated object can still contain a complete trailing array (the
        # common ``{"atomic_facts": [...]`` Qwen failure).  Keep that array
        # rather than cutting at its final nested object.
        end = max(text.rfind("}"), text.rfind("]"))
        if end > start:
            return text[start : end + 1]
    return text


def parse_json_response(text: str) -> Any:
    """Parse an LLM JSON response, repairing one known field-quote defect.

    Some Qwen responses are valid except for a line such as
    ``session_id": "..."`` where the opening quote of the object key is
    missing.  The repair is intentionally restricted to line-leading object
    keys and runs only after ordinary JSON parsing has failed.
    """
    payload = extract_json(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as original_error:
        repaired = re.sub(
            r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:',
            r'\1"\2":',
            payload,
        )
        for candidate in dict.fromkeys((payload, repaired)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Accept only a fully validated recovery for the specific
                # one-character truncation observed in Qwen fact extraction:
                # an object root whose final atomic-facts array is complete
                # but whose outer closing brace is absent.
                if candidate.lstrip().startswith("{") and candidate.rstrip().endswith("]"):
                    try:
                        return json.loads(candidate + "}")
                    except json.JSONDecodeError:
                        pass
        raise original_error
