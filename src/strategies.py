from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .answer_extractor import extract_answer
from .llm_client import BaseLLMClient
from .reflector import run_reflection
from .verifier import BaseVerifier, rule_based_verify


ACTIONS = ["direct", "reflect_1", "reflect_2"]


@dataclass
class StrategyResult:
    action: str
    final_answer: str
    pred: str | None
    prompt_tokens: int
    completion_tokens: int
    token_cost: int
    reflected_rounds: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def execute_strategy(
    action: str,
    llm: BaseLLMClient,
    question: str,
    initial_answer: str,
    initial_token_cost: int,
    initial_prompt_tokens: int = 0,
    initial_completion_tokens: int = 0,
    initial_prompt: str = "",
    verifier_threshold: float = 0.75,
    gold_answer: str | None = None,
    verifier: BaseVerifier | None = None,
    task_type: str | None = None,
    initial_response_metadata: Dict[str, Any] | None = None,
    qa_profile: Dict[str, Any] | None = None,
) -> StrategyResult:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    if action == "direct":
        return _make_result(
            action,
            initial_answer,
            initial_token_cost,
            0,
            {"calls": [_call_record(0, initial_prompt, initial_answer, initial_prompt_tokens, initial_completion_tokens, initial_token_cost, initial_response_metadata)]},
            prompt_tokens=initial_prompt_tokens,
            completion_tokens=initial_completion_tokens,
            task_type=task_type,
        )

    if action == "reflect_1":
        return _reflect_n(
            action, llm, question, initial_answer, initial_token_cost,
            n_rounds=1, initial_prompt_tokens=initial_prompt_tokens,
            initial_completion_tokens=initial_completion_tokens,
            initial_prompt=initial_prompt,
            initial_response_metadata=initial_response_metadata,
            gold_answer=gold_answer, task_type=task_type, qa_profile=qa_profile,
        )

    if action == "reflect_2":
        return _reflect_n(
            action, llm, question, initial_answer, initial_token_cost,
            n_rounds=2, initial_prompt_tokens=initial_prompt_tokens,
            initial_completion_tokens=initial_completion_tokens,
            initial_prompt=initial_prompt,
            initial_response_metadata=initial_response_metadata,
            gold_answer=gold_answer, task_type=task_type, qa_profile=qa_profile,
        )



def execute_all_strategies(
    llm: BaseLLMClient,
    question: str,
    initial_answer: str,
    initial_token_cost: int,
    initial_prompt_tokens: int = 0,
    initial_completion_tokens: int = 0,
    initial_prompt: str = "",
    verifier_threshold: float = 0.75,
    gold_answer: str | None = None,
    verifier: BaseVerifier | None = None,
    task_type: str | None = None,
    initial_response_metadata: Dict[str, Any] | None = None,
) -> List[StrategyResult]:
    return [
        execute_strategy(
            action=action,
            llm=llm,
            question=question,
            initial_answer=initial_answer,
            initial_token_cost=initial_token_cost,
            initial_prompt_tokens=initial_prompt_tokens,
            initial_completion_tokens=initial_completion_tokens,
            initial_prompt=initial_prompt,
            verifier_threshold=verifier_threshold,
            gold_answer=gold_answer,
            verifier=verifier,
            task_type=task_type,
            initial_response_metadata=initial_response_metadata,
        )
        for action in ACTIONS
    ]


def _reflect_n(
    action: str,
    llm: BaseLLMClient,
    question: str,
    initial_answer: str,
    initial_token_cost: int,
    n_rounds: int,
    initial_prompt_tokens: int = 0,
    initial_completion_tokens: int = 0,
    initial_prompt: str = "",
    gold_answer: str | None = None,
    task_type: str | None = None,
    initial_response_metadata: Dict[str, Any] | None = None,
    qa_profile: Dict[str, Any] | None = None,
) -> StrategyResult:
    current_answer = initial_answer
    total_cost = initial_token_cost
    total_prompt_tokens = initial_prompt_tokens
    total_completion_tokens = initial_completion_tokens
    round_costs = []
    round_usage = []
    call_records = [_call_record(0, initial_prompt, initial_answer, initial_prompt_tokens, initial_completion_tokens, initial_token_cost, initial_response_metadata)]
    for round_id in range(1, n_rounds + 1):
        kwargs = {"qa_profile": qa_profile} if qa_profile else {}
        response = run_reflection(llm, question, current_answer, round_id, gold_answer=gold_answer, task_type=task_type, **kwargs)
        current_answer = response.text
        total_cost += response.total_tokens
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens
        round_costs.append(response.total_tokens)
        round_usage.append({
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "finish_reason": response.metadata.get("finish_reason"),
        })
        call_records.append(_call_record(
            round_id,
            response.metadata.get("prompt", ""),
            response.text,
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
            response.metadata,
        ))
    return _make_result(
        action,
        current_answer,
        total_cost,
        n_rounds,
        {
            "reflection_round_costs": round_costs,
            "reflection_round_usage": round_usage,
            "calls": call_records,
        },
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        task_type=task_type,
    )



def _call_record(
    round_id: int,
    prompt: str,
    output: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    response_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record = {
        "round_id": round_id,
        "prompt": prompt,
        "output": output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if response_metadata:
        record["response_metadata"] = response_metadata
        record["finish_reason"] = response_metadata.get("finish_reason")
    return record

def _make_result(
    action: str,
    final_answer: str,
    token_cost: int,
    reflected_rounds: int,
    metadata: Dict[str, Any],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    task_type: str | None = None,
) -> StrategyResult:
    return StrategyResult(
        action=action,
        final_answer=final_answer,
        pred=extract_answer(final_answer, task_type=task_type),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_cost=token_cost,
        reflected_rounds=reflected_rounds,
        metadata=metadata,
    )
