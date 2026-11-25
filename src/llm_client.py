import hashlib
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .answer_extractor import extract_final_number
from .utils import count_tokens


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMClient:
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


def _normalize_stop_sequences(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    normalized = [str(item) for item in values if str(item)]
    return normalized or None


class FakeLLMClient(BaseLLMClient):
    """Deterministic fake model for pipeline debugging without external calls."""

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        question = _extract_section(prompt, "Question") or prompt
        is_reflection = "Current answer:" in prompt or "Reflection round:" in prompt
        gold_answer = kwargs.get("gold_answer")
        answer = str(gold_answer) if gold_answer is not None else _heuristic_math_answer(question)
        if answer is None:
            answer = _fallback_number(question)

        if is_reflection:
            noisy_answer = _maybe_add_noise(question + "::reflect", answer, error_mod=11)
            text = (
                "I rechecked the arithmetic and corrected the solution if needed. "
                f"Final answer: {noisy_answer}"
            )
        else:
            noisy_answer = _maybe_add_noise(question + "::direct", answer, error_mod=2)
            text = f"Reasoning omitted for fake mode. Final answer: {noisy_answer}"

        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(text)
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            metadata={"mode": "fake"},
        )


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        enable_thinking: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        seed: Optional[int] = None,
        api_mode: str = "chat_completions",
        completion_prompt_prefix: str = "",
        completion_prompt_suffix: str = "",
        stop: Any = None,
        retry_on_length: bool = False,
        max_tokens_cap: Optional[int] = None,
        fail_on_truncation: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed
        self.api_mode = str(api_mode)
        self.completion_prompt_prefix = str(completion_prompt_prefix)
        self.completion_prompt_suffix = str(completion_prompt_suffix)
        self.stop = _normalize_stop_sequences(stop)
        self.retry_on_length = bool(retry_on_length)
        self.max_tokens_cap = max(
            self.max_tokens,
            int(max_tokens_cap) if max_tokens_cap is not None else self.max_tokens,
        )
        self.fail_on_truncation = bool(fail_on_truncation)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        api_mode = str(kwargs.get("api_mode", self.api_mode))
        enable_thinking = kwargs.get("enable_thinking", self.enable_thinking)
        if "/no_think" in prompt:
            enable_thinking = False
        initial_max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        payload: Dict[str, Any] = {
            "model": kwargs.get("model_name", self.model_name),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": initial_max_tokens,
        }
        usage_prompt = prompt
        completion_prefix = kwargs.get(
            "completion_prompt_prefix", self.completion_prompt_prefix
        )
        completion_suffix = kwargs.get(
            "completion_prompt_suffix", self.completion_prompt_suffix
        )
        if api_mode == "chat_completions":
            url = f"{self.base_url}/chat/completions"
            payload["messages"] = [{"role": "user", "content": prompt}]
        elif api_mode == "completions":
            url = f"{self.base_url}/completions"
            usage_prompt = f"{completion_prefix}{prompt}{completion_suffix}"
            payload["prompt"] = usage_prompt
        else:
            raise ValueError(
                f"Unknown OpenAI-compatible api_mode={api_mode}; "
                "choose chat_completions or completions"
            )
        optional_parameters = {
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "min_p": kwargs.get("min_p", self.min_p),
            "seed": kwargs.get("seed", self.seed),
        }
        payload.update({key: value for key, value in optional_parameters.items() if value is not None})
        stop = _normalize_stop_sequences(kwargs.get("stop", self.stop))
        if stop is not None:
            payload["stop"] = stop
        if api_mode == "chat_completions" and enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

        retry_on_length = bool(kwargs.get("retry_on_length", self.retry_on_length))
        fail_on_truncation = bool(kwargs.get("fail_on_truncation", self.fail_on_truncation))
        max_tokens_cap = max(
            initial_max_tokens,
            int(kwargs.get("max_tokens_cap", self.max_tokens_cap)),
        )
        attempts: List[Dict[str, Any]] = []
        aggregate_prompt_tokens = 0
        aggregate_completion_tokens = 0
        aggregate_reported_total = 0
        final_usage: Dict[str, Any] = {}
        final_text = ""
        final_choice: Dict[str, Any] = {}
        final_effective_max = initial_max_tokens

        while True:
            requested_max = int(payload["max_tokens"])
            raw, effective_max, context_retry = self._request_payload(
                url=url,
                payload=payload,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                },
            )
            parsed = json.loads(raw)
            choice = parsed["choices"][0]
            text = (
                choice.get("text", "")
                if api_mode == "completions"
                else choice["message"]["content"]
            )
            usage = parsed.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", count_tokens(usage_prompt)))
            completion_tokens = int(usage.get("completion_tokens", count_tokens(text)))
            reported_total = usage.get("total_tokens")
            aggregate_prompt_tokens += prompt_tokens
            aggregate_completion_tokens += completion_tokens
            if reported_total is not None:
                aggregate_reported_total += int(reported_total)
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "requested_max_tokens": requested_max,
                    "effective_max_tokens": effective_max,
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "context_window_retry": context_retry,
                }
            )
            final_usage = usage
            final_text = text
            final_choice = choice
            final_effective_max = effective_max

            if choice.get("finish_reason") != "length":
                break

            context_exhausted = context_retry is not None and effective_max < requested_max
            next_max = min(max_tokens_cap, max(effective_max + 1, effective_max * 2))
            can_retry = retry_on_length and not context_exhausted and next_max > effective_max
            if can_retry:
                print(
                    f"[llm-client] output hit max_tokens={effective_max}; "
                    f"retrying with max_tokens={next_max}",
                    flush=True,
                )
                payload["max_tokens"] = next_max
                continue

            if fail_on_truncation:
                reason = (
                    "the context window left no additional output capacity"
                    if context_exhausted
                    else f"the retry cap {max_tokens_cap} was reached"
                )
                raise RuntimeError(
                    f"LLM output was truncated for model={payload['model']}: {reason}. "
                    "No trace row was written; increase MAX_MODEL_LEN or max_tokens_cap and resume."
                )
            break

        total_tokens = aggregate_prompt_tokens + aggregate_completion_tokens
        return LLMResponse(
            text=final_text,
            prompt_tokens=aggregate_prompt_tokens,
            completion_tokens=aggregate_completion_tokens,
            total_tokens=total_tokens,
            metadata={
                "mode": "openai_compatible",
                "model": payload["model"],
                "finish_reason": final_choice.get("finish_reason"),
                "request": {
                    "temperature": payload["temperature"],
                    "max_tokens": initial_max_tokens,
                    "max_tokens_cap": max_tokens_cap,
                    "effective_max_tokens": final_effective_max,
                    "retry_on_length": retry_on_length,
                    "fail_on_truncation": fail_on_truncation,
                    "top_p": payload.get("top_p"),
                    "top_k": payload.get("top_k"),
                    "min_p": payload.get("min_p"),
                    "seed": payload.get("seed"),
                    "stop": payload.get("stop"),
                    "enable_thinking": enable_thinking,
                    "api_mode": api_mode,
                    "completion_prompt_prefix": (
                        completion_prefix if api_mode == "completions" else None
                    ),
                    "completion_prompt_suffix": (
                        completion_suffix if api_mode == "completions" else None
                    ),
                    "wire_prompt_sha256": hashlib.sha256(
                        usage_prompt.encode("utf-8")
                    ).hexdigest(),
                },
                "generation_attempts": attempts,
                "raw_usage": final_usage,
                "reported_total_tokens": (
                    aggregate_reported_total if aggregate_reported_total else None
                ),
            },
        )

    def _request_payload(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> tuple[str, int, Dict[str, int] | None]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            return self._read_request(request), int(payload["max_tokens"]), None
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retry = self._retry_if_max_tokens_too_large(
                url=url,
                payload=payload,
                headers=headers,
                error_body=error_body,
            )
            if retry is None:
                raise RuntimeError(
                    f"OpenAI-compatible request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            return retry

    def _read_request(self, request: urllib.request.Request) -> str:
        transient_errors = (
            http.client.RemoteDisconnected,
            ConnectionError,
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError:
                raise
            except transient_errors as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI-compatible request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = self.retry_backoff * (2 ** attempt)
                print(
                    f"[llm-client] transient request failure: {exc}; "
                    f"retry {attempt + 1}/{self.max_retries} in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _retry_if_max_tokens_too_large(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        error_body: str,
    ) -> Optional[tuple[str, int, Dict[str, int]]]:
        match = re.search(r"\((\d+)\s*>\s*\d+\s*-\s*(\d+)\)", error_body)
        if not match:
            return None
        context_limit = int(match.group(1))
        input_tokens = int(match.group(2))
        retry_max_tokens = max(1, context_limit - input_tokens)
        original_max_tokens = int(payload.get("max_tokens", retry_max_tokens))
        if retry_max_tokens >= original_max_tokens:
            return None
        retry_payload = dict(payload)
        retry_payload["max_tokens"] = retry_max_tokens
        retry_data = json.dumps(retry_payload).encode("utf-8")
        retry_request = urllib.request.Request(url, data=retry_data, headers=headers, method="POST")
        raw = self._read_request(retry_request)
        return raw, retry_max_tokens, {
            "context_limit": context_limit,
            "input_tokens": input_tokens,
            "requested_max_tokens": original_max_tokens,
            "effective_max_tokens": retry_max_tokens,
        }


def build_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    mode = config.get("mode", "fake")
    if mode == "fake":
        return FakeLLMClient()
    if mode == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key") or os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), ""),
            model_name=config.get("model_name", "default"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 512)),
            timeout=int(config.get("timeout", 120)),
            max_retries=int(config.get("max_retries", 3)),
            retry_backoff=float(config.get("retry_backoff", 2.0)),
            enable_thinking=config.get("enable_thinking"),
            top_p=float(config["top_p"]) if config.get("top_p") is not None else None,
            top_k=int(config["top_k"]) if config.get("top_k") is not None else None,
            min_p=float(config["min_p"]) if config.get("min_p") is not None else None,
            seed=int(config["seed"]) if config.get("seed") is not None else None,
            api_mode=str(config.get("api_mode", "chat_completions")),
            completion_prompt_prefix=str(config.get("completion_prompt_prefix", "")),
            completion_prompt_suffix=str(config.get("completion_prompt_suffix", "")),
            stop=config.get("stop"),
            retry_on_length=bool(config.get("retry_on_length", False)),
            max_tokens_cap=(
                int(config["max_tokens_cap"])
                if config.get("max_tokens_cap") is not None
                else None
            ),
            fail_on_truncation=bool(config.get("fail_on_truncation", False)),
        )
    raise ValueError(f"Unknown llm mode: {mode}")


def _extract_section(prompt: str, title: str) -> Optional[str]:
    pattern = rf"{re.escape(title)}:\s*(.*?)(?:\n\n[A-Z][A-Za-z ]+:|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _numbers(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)]


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("%f" % value).rstrip("0").rstrip(".")


def _heuristic_math_answer(question: str) -> Optional[str]:
    q = question.lower()
    nums = _numbers(question)
    if not nums:
        return extract_final_number(question)

    if any(word in q for word in ["left", "remain", "remaining"]) and len(nums) >= 2:
        return _format_number(nums[0] - sum(nums[1:]))

    if any(word in q for word in ["each", "per", "apiece"]) and len(nums) >= 2:
        product = 1.0
        for n in nums:
            product *= n
        return _format_number(product)

    if any(word in q for word in ["total", "altogether", "in all", "sum"]):
        return _format_number(sum(nums))

    if any(word in q for word in ["more", "buys", "gets", "earns", "adds"]):
        return _format_number(sum(nums))

    if any(word in q for word in ["twice", "double"]) and len(nums) == 1:
        return _format_number(nums[0] * 2)

    return _format_number(sum(nums))


def _fallback_number(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return str(int(digest[:4], 16) % 100)


def _maybe_add_noise(key: str, answer: str, error_mod: int) -> str:
    digest = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    if digest % error_mod != 0:
        return answer
    try:
        value = float(answer)
    except ValueError:
        return answer
    return _format_number(value + 1)
