from __future__ import annotations

import argparse
import copy

import numpy as np

from .qa_dual_candidate_policy import align_dual_rows, build_dual_targets
from .qa_top_layer_policy import (
    _build_examples,
    _data_loader,
    _import_torch,
    _load_model,
    _predict_examples,
    _validation_diagnostics,
)
from .utils import load_json, load_pickle, read_jsonl, save_json, save_pickle


def recalibrate(
    bundle: dict,
    old_rows: list[dict],
    new_rows: list[dict],
    validation_indices: list[int],
    max_invocation_rate: float,
    threshold_grid_points: int,
    device: str,
) -> tuple[dict, dict]:
    torch, _, _, _ = _import_torch()
    pairs = align_dual_rows(old_rows, new_rows)
    selected_pairs = [pairs[int(index)] for index in validation_indices]
    targets = build_dual_targets(selected_pairs)
    local_indices = np.arange(len(selected_pairs), dtype="int64")
    gate_target_name = str(bundle.get("gate_target_name", "direct_error"))
    ranker_target_name = str(
        bundle.get("ranker_target_name", "candidate_correctness")
    )
    selection_score_mode = str(
        bundle.get("selection_score_mode", "correctness_margin")
    )
    examples = _build_examples(
        selected_pairs,
        targets,
        local_indices,
        int(bundle["max_text_chars"]),
        None,
        0,
        gate_target_name=gate_target_name,
        ranker_target_name=ranker_target_name,
        min_utility_gain=float(bundle.get("min_utility_gain", 0.05)),
    )
    tokenizer, model, _, resolved_device = _load_model(
        bundle["encoder_name"],
        int(bundle["trainable_blocks"]),
        bundle["layer_indices"],
        bundle["pooling"],
        bundle["hidden_sizes"],
        float(bundle["dropout"]),
        device,
        state_dict=bundle["state_dict"],
    )
    loader = _data_loader(
        tokenizer,
        examples,
        int(bundle["max_length"]),
        int(bundle.get("eval_batch_size", 8)),
        False,
        0,
    )
    probability, label, row, source, validation_loss = _predict_examples(
        torch,
        model,
        loader,
        resolved_device,
        progress_label="recalibration inference",
    )
    diagnostics = _validation_diagnostics(
        selected_pairs,
        targets,
        local_indices,
        probability,
        label,
        row,
        source,
        float(max_invocation_rate),
        float(bundle.get("lambda_cost", 0.0001)),
        int(threshold_grid_points),
        selection_score_mode,
    )
    output = copy.deepcopy(bundle)
    original = {
        "gate_threshold": float(bundle["gate_threshold"]),
        "margin_threshold": float(bundle["margin_threshold"]),
        "max_invocation_rate": float(bundle.get("max_invocation_rate", 0.15)),
    }
    output["gate_threshold"] = float(diagnostics["gate_threshold"])
    output["margin_threshold"] = float(diagnostics["margin_threshold"])
    output["max_invocation_rate"] = float(max_invocation_rate)
    output["recalibration"] = {
        "original": original,
        "validation_rows": len(selected_pairs),
        "validation_loss": float(validation_loss),
        "max_invocation_rate": float(max_invocation_rate),
        "threshold_grid_points": int(threshold_grid_points),
        "gate_threshold": float(diagnostics["gate_threshold"]),
        "margin_threshold": float(diagnostics["margin_threshold"]),
        "selection": diagnostics["selection"],
        "gate_metrics": diagnostics["gate"],
        "ranker_metrics": diagnostics["ranker"],
        "test_labels_used": False,
    }
    del model
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output, output["recalibration"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate a saved QA top-layer policy on its fixed validation "
            "split without updating model weights or using test labels."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--old_train_trace", required=True)
    parser.add_argument("--new_train_trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--history")
    parser.add_argument("--max_invocation_rate", type=float, default=0.05)
    parser.add_argument("--threshold_grid_points", type=int, default=81)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bundle = load_pickle(args.model)
    history_path = args.history or bundle.get("history_output")
    if not history_path:
        raise ValueError("Training history path is required for validation indices")
    history = load_json(history_path)
    validation_indices = history.get("validation_indices")
    if not validation_indices:
        raise ValueError("Training history has no validation_indices")
    output, summary = recalibrate(
        bundle,
        read_jsonl(args.old_train_trace),
        read_jsonl(args.new_train_trace),
        validation_indices,
        args.max_invocation_rate,
        args.threshold_grid_points,
        args.device,
    )
    save_pickle(output, args.output)
    summary["input_model"] = args.model
    summary["output_model"] = args.output
    summary["history"] = history_path
    save_json(summary, args.summary_output)
    print(summary)


if __name__ == "__main__":
    main()
