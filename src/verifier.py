import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .answer_extractor import extract_answer
from .llm_client import BaseLLMClient, build_llm_client
from .prompts import verify_prompt


@dataclass
class VerifierResult:
    verifier_score: float
    pred: Optional[str]
    reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_cost: int = 0
    mode: str = "rule_based"
    metadata: Dict[str, Any] | None = None


class BaseVerifier:
    def verify(self, question: str, answer_text: str, task_type: str | None = None) -> VerifierResult:
        raise NotImplementedError


class RuleBasedVerifier(BaseVerifier):
    def verify(self, question: str, answer_text: str, task_type: str | None = None) -> VerifierResult:
        return rule_based_verify(answer_text, task_type=task_type)


class LLMVerifier(BaseVerifier):
    def __init__(
        self,
        llm: BaseLLMClient,
        temperature: float = 0.0,
        max_tokens: int = 128,
        fallback_to_rule: bool = True,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_to_rule = fallback_to_rule
        self._cache: Dict[tuple[str, str, str], VerifierResult] = {}

    def verify(self, question: str, answer_text: str, task_type: str | None = None) -> VerifierResult:
        task = str(task_type or "math").lower()
        cache_key = (task, question, answer_text)
        if cache_key in self._cache:
            cached = copy.deepcopy(self._cache[cache_key])
            if cached.metadata is None:
                cached.metadata = {}
            cached.metadata["cache_hit"] = True
            return cached

        pred = extract_answer(answer_text, task_type=task)
        if pred is None:
            result = VerifierResult(
                verifier_score=0.0,
                pred=None,
                reason="No extractable final answer.",
                token_cost=0,
                mode="llm_skipped",
                metadata={"skipped_reason": "no_final_answer", "task_type": task},
            )
            self._cache[cache_key] = copy.deepcopy(result)
            return result

        prompt = verify_prompt(question, answer_text, task_type=task)
        response = self.llm.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed = _parse_verifier_json(response.text)
        if parsed is None:
            if not self.fallback_to_rule:
                result = VerifierResult(
                    verifier_score=0.0,
                    pred=pred,
                    reason="LLM verifier response could not be parsed.",
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    token_cost=response.total_tokens,
                    mode="llm_parse_error",
                    metadata={"raw_response": response.text, "task_type": task},
                )
                self._cache[cache_key] = copy.deepcopy(result)
                return result
            fallback = rule_based_verify(answer_text, task_type=task)
            fallback.prompt_tokens = response.prompt_tokens
            fallback.completion_tokens = response.completion_tokens
            fallback.token_cost = response.total_tokens
            fallback.mode = "llm_fallback_rule_based"
            fallback.reason = f"LLM verifier parse failed; fallback used. Raw: {response.text[:200]}"
            fallback.metadata = {"raw_response": response.text, "task_type": task}
            self._cache[cache_key] = copy.deepcopy(fallback)
            return fallback

        score = _clamp_score(parsed.get("score", 0.0))
        verdict = str(parsed.get("verdict", "uncertain"))
        reason = str(parsed.get("reason", "")) or verdict
        result = VerifierResult(
            verifier_score=score,
            pred=pred,
            reason=f"{verdict}: {reason}",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            token_cost=response.total_tokens,
            mode="llm",
            metadata={"raw_response": response.text, "parsed": parsed, "task_type": task},
        )
        self._cache[cache_key] = copy.deepcopy(result)
        return result


def build_verifier(config: Dict[str, Any], default_llm: BaseLLMClient | None = None) -> BaseVerifier:
    mode = config.get("mode", "rule_based")
    if mode == "rule_based":
        return RuleBasedVerifier()
    if mode == "llm":
        llm = default_llm
        if not config.get("use_main_llm", True):
            llm = build_llm_client(config.get("llm", {"mode": "fake"}))
        if llm is None:
            llm = build_llm_client(config.get("llm", {"mode": "fake"}))
        return LLMVerifier(
            llm=llm,
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 128)),
            fallback_to_rule=bool(config.get("fallback_to_rule", True)),
        )
    raise ValueError(f"Unknown verifier mode: {mode}")


def rule_based_verify(answer_text: str, task_type: str | None = None) -> VerifierResult:
    task = str(task_type or "math").lower()
    pred = extract_answer(answer_text, task_type=task)
    if pred is None:
        return VerifierResult(
            verifier_score=0.0,
            pred=None,
            reason="No extractable final answer.",
            mode="rule_based",
            metadata={"task_type": task},
        )
    return VerifierResult(
        verifier_score=0.5,
        pred=pred,
        reason="Final answer was extracted; rule verifier is uncertain.",
        mode="rule_based",
        metadata={"task_type": task},
    )


def _parse_verifier_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    score_match = re.search(r"(?:score|confidence)\D+([01](?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "verdict": "uncertain",
            "reason": cleaned[:200],
        }
    return None


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))
