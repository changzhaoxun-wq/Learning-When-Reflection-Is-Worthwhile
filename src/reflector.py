from copy import deepcopy
import re

from .answer_extractor import extract_final_text_answer
from .llm_client import BaseLLMClient, LLMResponse
from .prompts import qa_format_repair_prompt, qa_truncation_recovery_prompt, reflection_prompt
from .qa_reflection_profile import profile_hash, profile_prompt


QA_FORMAT_REPAIR_MAX_TOKENS = 128
QA_FORMAT_REPAIR_MAX_TOKENS_CAP = 256
QA_FORMAT_REPAIR_STOP = ["\n"]
QA_FORMAT_REPAIR_COMPLETION_SUFFIX = "<｜Assistant｜><think>\n</think>\nAnswer span: "
QA_REFLECTION_COMPLETION_SUFFIX = "<｜Assistant｜><think>\n</think>\nDecision: "
QA_REFLECTION_PRIMARY_MAX_TOKENS = 256
QA_TRUNCATION_RECOVERY_MAX_TOKENS = 128
QA_TRUNCATION_RECOVERY_MAX_TOKENS_CAP = 256
QA_ANSWER_CANONICALIZER_VERSION = "qa_answer_canonicalizer_v3"


def _attempts(response: LLMResponse, phase: str) -> list[dict]:
    attempts = []
    for item in response.metadata.get("generation_attempts") or []:
        attempt = deepcopy(item)
        attempt["phase"] = phase
        attempt["attempt"] = len(attempts) + 1
        attempts.append(attempt)
    return attempts


def _raw_qa_question(question: str) -> str:
    marker = "\nQuestion:\n"
    return question.rsplit(marker, 1)[-1].strip() if marker in question else question.strip()


def _comparison_subject(question: str, extracted: str) -> str | None:
    raw_question = _raw_qa_question(question).lower()
    asks_for_entity = bool(re.search(r"\b(who|which)\b", raw_question))
    comparison = bool(
        re.search(
            r"\b(more|fewer|less|older|younger|earlier|later|first|larger|smaller|higher|lower)\b",
            raw_question,
        )
    )
    if not (asks_for_entity and comparison):
        return None
    match = re.match(
        r"^(.+?)\s+(?:has\s+won|had\s+won|won|wins|has|have|had|was|were|is|are)\b",
        extracted.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    subject = match.group(1).strip().strip("<>\"'` ").strip("*_ ")
    if not subject or len(subject.split()) > 8 or re.search(r"[.!?;:]", subject):
        return None
    return subject


def _recover_truncated_qa_reflection(
    llm: BaseLLMClient,
    question: str,
    response: LLMResponse,
    gold_answer: str | None,
) -> LLMResponse:
    recovery_prompt = qa_truncation_recovery_prompt(question)
    recovered = llm.generate(
        recovery_prompt,
        gold_answer=gold_answer,
        max_tokens=QA_TRUNCATION_RECOVERY_MAX_TOKENS,
        max_tokens_cap=QA_TRUNCATION_RECOVERY_MAX_TOKENS_CAP,
        stop=QA_FORMAT_REPAIR_STOP,
        completion_prompt_suffix=QA_FORMAT_REPAIR_COMPLETION_SUFFIX,
        retry_on_length=True,
        fail_on_truncation=True,
    )
    candidate = recovered.text.strip().splitlines()[0] if recovered.text.strip() else ""
    candidate = re.sub(r"^final\s+answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^answer\s+span\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip().strip("<>\"'` ").strip("*_ ")
    if not candidate:
        raise RuntimeError(
            "QA truncation recovery returned an empty answer span; no trace row was written."
        )

    original_attempts = _attempts(response, "reflection_truncated")
    recovery_attempts = _attempts(recovered, "truncation_recovery")
    attempts = [*original_attempts, *recovery_attempts]
    for index, attempt in enumerate(attempts, start=1):
        attempt["attempt"] = index
    reported_totals = [
        value
        for value in (
            response.metadata.get("reported_total_tokens"),
            recovered.metadata.get("reported_total_tokens"),
        )
        if value is not None
    ]
    canonical_output = f"Final answer: {candidate}"
    metadata = {
        **response.metadata,
        "finish_reason": recovered.metadata.get("finish_reason"),
        "generation_attempts": attempts,
        "reported_total_tokens": sum(int(value) for value in reported_totals) or None,
        "qa_truncation_recovery": {
            "applied": True,
            "method": "independent_minimal_answer",
            "truncated_output": response.text,
            "truncated_response_metadata": response.metadata,
            "recovery_prompt": recovery_prompt,
            "recovery_output": recovered.text,
            "canonical_output": canonical_output,
            "recovery_response_metadata": recovered.metadata,
        },
    }
    return LLMResponse(
        text=canonical_output,
        prompt_tokens=response.prompt_tokens + recovered.prompt_tokens,
        completion_tokens=response.completion_tokens + recovered.completion_tokens,
        total_tokens=response.total_tokens + recovered.total_tokens,
        metadata=metadata,
    )


def _abbreviation_expansion(question: str, draft: str) -> str | None:
    if "stand for" not in _raw_qa_question(question).lower():
        return None
    matches = re.findall(
        r"(?:stands?\s+for|(?:is\s+)?an?\s+abbreviation\s+(?:for|of))\s+(.+?)(?:[,.;\n]|$)",
        draft,
        flags=re.IGNORECASE,
    )
    for value in matches:
        candidate = value.strip().strip("<>\"'` ").strip("*_ ")
        if 1 <= len(candidate.split()) <= 8 and not re.search(r"[;:]", candidate):
            return candidate
    return None


def _repair_qa_format(
    llm: BaseLLMClient,
    question: str,
    current_answer: str,
    response: LLMResponse,
    gold_answer: str | None,
    qa_profile: dict | None = None,
) -> LLMResponse:
    extracted = extract_final_text_answer(response.text) or ""
    words = extracted.split()
    contains_placeholder = bool(
        re.search(r"<?\s*minimal\s+answer\s+span\s*>?", extracted, flags=re.IGNORECASE)
    )
    needs_repair = (
        "final answer:" not in response.text.lower()
        or len(words) > 8
        or contains_placeholder
    )

    abbreviation_expansion = _abbreviation_expansion(question, response.text)
    if abbreviation_expansion is not None:
        canonical_output = f"Final answer: {abbreviation_expansion}"
        metadata = {
            **response.metadata,
            "qa_format_repair": {
                "applied": True,
                "method": "deterministic_abbreviation_expansion",
                "reason": "semantic_target_span",
                "original_output": response.text,
                "canonical_output": canonical_output,
            },
        }
        return LLMResponse(
            text=canonical_output,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            metadata=metadata,
        )

    if not needs_repair:
        return response

    compact_subject = _comparison_subject(question, extracted)
    if compact_subject is not None:
        canonical_output = f"Final answer: {compact_subject}"
        metadata = {
            **response.metadata,
            "qa_format_repair": {
                "applied": True,
                "method": "deterministic_comparison_subject",
                "reason": "non_minimal_final_span",
                "original_output": response.text,
                "canonical_output": canonical_output,
            },
        }
        return LLMResponse(
            text=canonical_output,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            metadata=metadata,
        )

    repair_prompt = (
        qa_profile["format_repair"].format(
            question=question, draft_answer=response.text, current_answer=current_answer
        )
        if qa_profile else qa_format_repair_prompt(question, response.text, current_answer)
    )
    repaired = llm.generate(
        repair_prompt,
        gold_answer=gold_answer,
        max_tokens=QA_FORMAT_REPAIR_MAX_TOKENS,
        max_tokens_cap=QA_FORMAT_REPAIR_MAX_TOKENS_CAP,
        stop=QA_FORMAT_REPAIR_STOP,
        completion_prompt_suffix=QA_FORMAT_REPAIR_COMPLETION_SUFFIX,
        retry_on_length=True,
        fail_on_truncation=False,
    )
    if repaired.metadata.get("finish_reason") == "length":
        candidate = extracted.strip().strip("<>\"'` ").strip("*_ ")
        if not candidate or contains_placeholder:
            raise RuntimeError(
                "QA format repair was truncated and the complete draft did not "
                "contain a usable answer span; no trace row was written."
            )
        canonical_output = f"Final answer: {candidate}"
        original_attempts = _attempts(response, "reflection")
        repair_attempts = _attempts(repaired, "format_repair_truncated")
        attempts = [*original_attempts, *repair_attempts]
        for index, attempt in enumerate(attempts, start=1):
            attempt["attempt"] = index
        reported_totals = [
            value
            for value in (
                response.metadata.get("reported_total_tokens"),
                repaired.metadata.get("reported_total_tokens"),
            )
            if value is not None
        ]
        metadata = {
            **response.metadata,
            "finish_reason": response.metadata.get("finish_reason"),
            "generation_attempts": attempts,
            "reported_total_tokens": sum(int(value) for value in reported_totals) or None,
            "qa_format_repair": {
                "applied": True,
                "method": "complete_draft_extraction_after_truncated_repair",
                "reason": "format_repair_truncated",
                "original_output": response.text,
                "original_response_metadata": response.metadata,
                "repair_prompt": repair_prompt,
                "repair_output": repaired.text,
                "repair_output_used": False,
                "canonical_output": canonical_output,
                "repair_response_metadata": repaired.metadata,
            },
        }
        return LLMResponse(
            text=canonical_output,
            prompt_tokens=response.prompt_tokens + repaired.prompt_tokens,
            completion_tokens=response.completion_tokens + repaired.completion_tokens,
            total_tokens=response.total_tokens + repaired.total_tokens,
            metadata=metadata,
        )
    candidate = repaired.text.strip().splitlines()[0] if repaired.text.strip() else ""
    candidate = re.sub(r"^final\s+answer\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"^<?\s*minimal\s+answer\s+span\s*>?\s*[:=-]?\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = candidate.strip().strip("<>\"'` ").strip("*_ ")
    if not candidate:
        raise RuntimeError(
            "QA format repair returned an empty answer span; no trace row was written."
        )
    canonical_output = f"Final answer: {candidate}"

    original_attempts = _attempts(response, "reflection")
    repair_attempts = _attempts(repaired, "format_repair")
    for index, attempt in enumerate([*original_attempts, *repair_attempts], start=1):
        attempt["attempt"] = index
    reported_totals = [
        value
        for value in (
            response.metadata.get("reported_total_tokens"),
            repaired.metadata.get("reported_total_tokens"),
        )
        if value is not None
    ]
    metadata = {
        **response.metadata,
        "finish_reason": repaired.metadata.get("finish_reason"),
        "generation_attempts": [*original_attempts, *repair_attempts],
        "reported_total_tokens": sum(int(value) for value in reported_totals) or None,
        "qa_format_repair": {
            "applied": True,
            "method": "llm_span_extraction",
            "reason": (
                "missing_final_marker"
                if "final answer:" not in response.text.lower()
                else "non_minimal_final_span"
            ),
            "original_output": response.text,
            "original_response_metadata": response.metadata,
            "repair_prompt": repair_prompt,
            "repair_output": repaired.text,
            "canonical_output": canonical_output,
            "repair_response_metadata": repaired.metadata,
        },
    }
    return LLMResponse(
        text=canonical_output,
        prompt_tokens=response.prompt_tokens + repaired.prompt_tokens,
        completion_tokens=response.completion_tokens + repaired.completion_tokens,
        total_tokens=response.total_tokens + repaired.total_tokens,
        metadata=metadata,
    )


def run_reflection(
    llm: BaseLLMClient,
    question: str,
    current_answer: str,
    round_id: int,
    gold_answer: str | None = None,
    task_type: str | None = None,
    qa_profile: dict | None = None,
) -> LLMResponse:
    current_span = extract_final_text_answer(current_answer) or current_answer
    current_span = current_span.strip()
    canonical_current_answer = f"Final answer: {current_span}"
    prompt = reflection_prompt(
        question=question,
        current_answer=canonical_current_answer,
        round_id=round_id,
        task_type=task_type,
    )
    if str(task_type or "math").lower() == "qa":
        if qa_profile:
            prompt = profile_prompt(qa_profile, question, current_answer, round_id)
        response = llm.generate(
            prompt,
            gold_answer=gold_answer,
            max_tokens=QA_REFLECTION_PRIMARY_MAX_TOKENS,
            max_tokens_cap=QA_REFLECTION_PRIMARY_MAX_TOKENS,
            retry_on_length=False,
            fail_on_truncation=False,
            completion_prompt_suffix=(
                qa_profile["completion_suffix"] if qa_profile
                else QA_REFLECTION_COMPLETION_SUFFIX
            ),
        )
        if (not qa_profile or qa_profile.get("prepend_decision", False)) and not re.match(r"(?i)^\s*decision\s*:", response.text):
            response.text = f"Decision: {response.text.lstrip()}"
        if response.metadata.get("finish_reason") == "length":
            response = _recover_truncated_qa_reflection(
                llm, question, response, gold_answer
            )
        response = _repair_qa_format(
            llm,
            question,
            canonical_current_answer,
            response,
            gold_answer,
            qa_profile=qa_profile,
        )
    else:
        response = llm.generate(prompt, gold_answer=gold_answer)
    response.metadata = {**response.metadata, "prompt": prompt}
    if qa_profile and str(task_type or "math").lower() == "qa":
        response.metadata["qa_profile_sha256"] = profile_hash(qa_profile)
        response.metadata["qa_profile_version"] = qa_profile["version"]
    return response
