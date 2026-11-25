from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from src.evaluate_policy_offline import _add, _candidate_usage, _policy_row, _rows_from_stats
from src.neural_policy import _encode_hf_hidden, build_policy_text
from src.reward import compute_reward
from src.train_policy import predict_action
from src.transformer_policy import TorchPolicyMLP  # noqa: F401
from src.utils import load_pickle, print_table, read_jsonl, save_json


def _probabilities(bundle: Dict[str, Any], row: Dict[str, Any]) -> np.ndarray:
    text = build_policy_text(
        row,
        max_text_chars=int(bundle.get("max_text_chars", 8000)),
        include_state=not bool(bundle.get("pure_text", True)),
    )
    features, _ = _encode_hf_hidden(
        [text],
        encoder_name=bundle["encoder_name"],
        max_length=int(bundle.get("max_length", 1024)),
        batch_size=1,
        device=bundle.get("device", "auto"),
        pooling=bundle.get("pooling", "mean"),
        layer_indices=list(bundle.get("layers", [-1])),
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = bundle["scaler"].transform(features).astype("float32")
    with torch.no_grad():
        logits = bundle["model"](torch.from_numpy(features))
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()


def evaluate(rows: List[Dict[str, Any]], gate_path: str, round_paths: List[str], lambda_cost: float) -> List[Dict[str, Any]]:
    gate = load_pickle(gate_path)
    round_models = [load_pickle(path) for path in round_paths]
    methods = ["direct", "fixed_reflect_1", "fixed_reflect_2", "learned_policy", "oracle"]
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        candidates = {str(x.get("action")): x for x in row.get("candidate_results", [])}
        gate_action = predict_action(gate, _policy_row(row))
        if gate_action == "direct":
            learned_action = "direct"
        else:
            probabilities = np.mean([_probabilities(model, _policy_row(row)) for model in round_models], axis=0)
            actions = list(round_models[0]["action_labels"])
            learned_action = actions[int(np.argmax(probabilities))]

        oracle_action = max(
            [candidates[action] for action in ("direct", "reflect_1", "reflect_2")],
            key=lambda item: (
                compute_reward(int(float(item.get("correct", 0))), int(round(_candidate_usage(item)[2])), lambda_cost),
                -_candidate_usage(item)[2],
            ),
        )["action"]
        selected = {
            "direct": candidates["direct"],
            "fixed_reflect_1": candidates["reflect_1"],
            "fixed_reflect_2": candidates["reflect_2"],
            "learned_policy": candidates[learned_action],
            "oracle": candidates[oracle_action],
        }
        for method, candidate in selected.items():
            _add(stats[method], candidate, lambda_cost)
    return _rows_from_stats({"overall": stats}, None, methods)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation for an ensemble hierarchical policy.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gate_model", required=True)
    parser.add_argument("--round_models", required=True, help="Comma-separated round policy pickle paths")
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = evaluate(
        read_jsonl(args.trace),
        args.gate_model,
        [path for path in args.round_models.split(",") if path],
        args.lambda_cost,
    )
    save_json(results, args.output)
    print(print_table(results, ["method", "n", "accuracy", "avg_prompt_tokens", "avg_completion_tokens", "avg_token_cost", "reflection_rate", "avg_reward"]))
    print("Read saved traces; no LLM calls were made.")
    print(f"Wrote ensemble results to {args.output}")


if __name__ == "__main__":
    main()
