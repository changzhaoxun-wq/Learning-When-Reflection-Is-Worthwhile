from __future__ import annotations

import argparse
import copy
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .evaluate_policy_offline import _candidate_utility
from .neural_policy import (
    _clip_context,
    _ensure_torchvision_nms_compat,
    _qa_question_and_context,
    build_policy_text,
)
from .qa_dual_candidate_policy import (
    CANDIDATE_SOURCES,
    _candidate_map,
    _correct,
    align_dual_rows,
    build_candidate_text,
    build_dual_targets,
    build_usage_arrays,
    pair_candidates,
    reflection_decision_score,
    select_dual_thresholds,
)
from .utils import read_jsonl, save_json, save_pickle


POLICY_TYPE = "qa_top_layer_dual_candidate"
TASK_GATE = 0
TASK_RANKER = 1


def _import_torch() -> tuple[Any, Any, Any, Any]:
    import torch

    _ensure_torchvision_nms_compat(torch)
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    return torch, nn, DataLoader, Dataset


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return str(device)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _encoder_layers(encoder: Any) -> Any:
    if hasattr(encoder, "layers"):
        return encoder.layers
    if hasattr(encoder, "model") and hasattr(encoder.model, "layers"):
        return encoder.model.layers
    raise ValueError("Cannot locate Transformer blocks on the encoder")


def _encoder_norm(encoder: Any) -> Any | None:
    if hasattr(encoder, "norm"):
        return encoder.norm
    if hasattr(encoder, "model") and hasattr(encoder.model, "norm"):
        return encoder.model.norm
    return None


def configure_trainable_top_layers(encoder: Any, trainable_blocks: int) -> Dict[str, int]:
    """Freeze the encoder except its final N blocks and final norm."""

    layers = _encoder_layers(encoder)
    count = len(layers)
    if trainable_blocks < 1 or trainable_blocks > count:
        raise ValueError(
            f"trainable_blocks must be in [1, {count}], got {trainable_blocks}"
        )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for layer in layers[-int(trainable_blocks) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    norm = _encoder_norm(encoder)
    if norm is not None:
        for parameter in norm.parameters():
            parameter.requires_grad = True
    total = sum(parameter.numel() for parameter in encoder.parameters())
    trainable = sum(
        parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
    )
    return {
        "total_blocks": int(count),
        "frozen_blocks": int(count - trainable_blocks),
        "trainable_blocks": int(trainable_blocks),
        "encoder_parameters": int(total),
        "trainable_encoder_parameters": int(trainable),
    }


def _pool_hidden(
    torch: Any,
    hidden_states: Sequence[Any],
    attention_mask: Any,
    layer_indices: Sequence[int],
    pooling: str,
) -> Any:
    parts = []
    for index in layer_indices:
        hidden = hidden_states[int(index)]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        last = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), lengths
        ]
        if pooling == "mean":
            parts.append(mean)
        elif pooling == "last":
            parts.append(last)
        elif pooling == "mean_last":
            parts.extend([mean, last])
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
    return torch.cat(parts, dim=1).float()


def _make_head(nn: Any, input_dim: int, hidden_sizes: Sequence[int], dropout: float) -> Any:
    modules: List[Any] = [nn.LayerNorm(int(input_dim))]
    previous = int(input_dim)
    for width in hidden_sizes:
        modules.extend(
            [
                nn.Linear(previous, int(width)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        )
        previous = int(width)
    modules.append(nn.Linear(previous, 1))
    return nn.Sequential(*modules)


def _make_model_class(torch: Any, nn: Any) -> type:
    class TopLayerDualPolicy(nn.Module):
        def __init__(
            self,
            encoder: Any,
            hidden_size: int,
            layer_indices: Sequence[int],
            pooling: str,
            hidden_sizes: Sequence[int],
            dropout: float,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.layer_indices = tuple(int(value) for value in layer_indices)
            self.pooling = str(pooling)
            multiplier = len(self.layer_indices) * (2 if pooling == "mean_last" else 1)
            feature_dim = int(hidden_size) * multiplier
            self.gate_head = _make_head(nn, feature_dim, hidden_sizes, dropout)
            self.ranker_head = _make_head(nn, feature_dim, hidden_sizes, dropout)
            self.feature_dim = feature_dim

        def forward(self, input_ids: Any, attention_mask: Any, task_ids: Any) -> Any:
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            features = _pool_hidden(
                torch,
                output.hidden_states,
                attention_mask,
                self.layer_indices,
                self.pooling,
            )
            gate_logits = self.gate_head(features).squeeze(-1)
            ranker_logits = self.ranker_head(features).squeeze(-1)
            return torch.where(task_ids == TASK_GATE, gate_logits, ranker_logits)

    return TopLayerDualPolicy


def _load_model(
    encoder_name: str,
    trainable_blocks: int,
    layer_indices: Sequence[int],
    pooling: str,
    hidden_sizes: Sequence[int],
    dropout: float,
    device: str,
    state_dict: Dict[str, Any] | None = None,
) -> tuple[Any, Any, Dict[str, int], str]:
    torch, nn, _, _ = _import_torch()
    from transformers import AutoModel, AutoTokenizer

    resolved_device = _resolve_device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    dtype = (
        torch.bfloat16
        if resolved_device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    encoder = AutoModel.from_pretrained(
        encoder_name,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    encoder.config.use_cache = False
    trainable_info = configure_trainable_top_layers(encoder, trainable_blocks)
    model_class = _make_model_class(torch, nn)
    model = model_class(
        encoder,
        int(encoder.config.hidden_size),
        layer_indices,
        pooling,
        hidden_sizes,
        dropout,
    )
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected[:5]}")
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        missing_trainable = sorted(trainable_names.intersection(missing))
        if missing_trainable:
            raise ValueError(
                f"Checkpoint is missing trainable keys: {missing_trainable[:5]}"
            )
    model.to(resolved_device)
    return tokenizer, model, trainable_info, resolved_device


def _split_rows(
    targets: Dict[str, np.ndarray], validation_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(targets["complete"]), dtype="int64")
    if validation_size <= 0 or len(indices) < 10:
        return indices, np.asarray([], dtype="int64")
    strata = np.asarray(
        [
            f"{int(targets['direct_error_target'][i])}_"
            f"{int(targets['gate_target'][i])}_"
            f"{int(targets['complete'][i])}"
            for i in indices
        ]
    )
    counts = Counter(strata.tolist())
    stratify = strata if min(counts.values()) >= 2 else None
    train, validation = train_test_split(
        indices,
        test_size=float(validation_size),
        random_state=int(seed),
        stratify=stratify,
    )
    return np.asarray(train, dtype="int64"), np.asarray(validation, dtype="int64")


def attach_sampled_augmentations(
    pairs: Sequence[Dict[str, Any]],
    augmentation_datasets: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, int]:
    pair_by_id = {str(pair["id"]): pair for pair in pairs}
    counts: Counter[str] = Counter()
    for pass_index, rows in enumerate(augmentation_datasets, 1):
        seen: set[str] = set()
        for row in rows:
            row_id = str(row.get("id"))
            if row_id in seen:
                raise ValueError(
                    f"Duplicate id in augmentation pass {pass_index}: {row_id}"
                )
            seen.add(row_id)
            pair = pair_by_id.get(row_id)
            if pair is None:
                raise ValueError(
                    f"Unknown augmentation id in pass {pass_index}: {row_id}"
                )
            candidates = _candidate_map(row)
            direct = candidates.get("direct")
            reflected = candidates.get("reflect_1")
            if direct is None or reflected is None:
                raise ValueError(
                    f"Augmentation row {row_id} must contain direct and reflect_1"
                )
            if _correct(direct) != _correct(pair["new_candidates"]["direct"]):
                raise ValueError(
                    f"Direct label mismatch in augmentation row {row_id}"
                )
            pair.setdefault("sampled_augmentation_candidates", []).append(
                {
                    "candidate": reflected,
                    "source_tag": "sampled_reflect_1_train_only",
                    "pass_index": pass_index,
                }
            )
            counts["rows_attached"] += 1
            counts["correct_candidates"] += _correct(reflected)
        counts["passes"] += 1
    return dict(counts)



def _train_only_candidate_text(
    pair: Dict[str, Any],
    candidate: Dict[str, Any],
    source_tag: str,
    max_text_chars: int,
) -> str:
    row = pair["new_row"]
    state = row.get("state_features", {}) or {}
    question, context = _qa_question_and_context(row, state)
    initial_answer = str(
        row.get("initial_answer") or state.get("initial_answer") or ""
    )
    candidate_answer = str(
        candidate.get("final_answer") or candidate.get("pred") or ""
    )
    fixed = '\n'.join(
        [
            "Question:",
            question,
            "Initial response:",
            initial_answer,
            f"Candidate source: {source_tag}",
            "Candidate response:",
            candidate_answer,
        ]
    )
    if not context:
        return fixed[:max_text_chars] if max_text_chars else fixed
    header = '\nContext:\n'
    budget = len(context)
    if max_text_chars:
        budget = max(0, max_text_chars - len(fixed) - len(header))
    text = fixed + header + _clip_context(context, budget)
    return text[:max_text_chars] if max_text_chars else text

def _candidate_training_examples(
    pairs: Sequence[Dict[str, Any]],
    targets: Dict[str, np.ndarray],
    row_indices: Iterable[int],
    max_text_chars: int,
    negative_ratio: float,
    seed: int,
    include_reflect_2_augmentation: bool,
    include_sampled_augmentation: bool,
    ranker_target_name: str,
    min_utility_gain: float,
) -> List[Dict[str, Any]]:
    examples = []
    for row_index in row_indices:
        row_index = int(row_index)
        if not targets["complete"][row_index]:
            continue
        direct_candidate = pairs[row_index]["new_candidates"]["direct"]
        source_indices = [1, 2]
        if (
            include_reflect_2_augmentation
            and "reflect_2" in pairs[row_index]["old_candidates"]
        ):
            source_indices.append(3)
        for source_index in source_indices:
            if source_index == 3:
                candidate = pairs[row_index]["old_candidates"]["reflect_2"]
                label = _ranker_training_label(
                    direct_candidate,
                    candidate,
                    ranker_target_name,
                    min_utility_gain,
                )
                text = _train_only_candidate_text(
                    pairs[row_index],
                    candidate,
                    "v5_reflect_2_train_only",
                    max_text_chars,
                )
            else:
                candidate = (
                    pairs[row_index]["old_candidates"]["reflect_1"]
                    if source_index == 1
                    else pairs[row_index]["new_candidates"]["reflect_1"]
                )
                label = _ranker_training_label(
                    direct_candidate,
                    candidate,
                    ranker_target_name,
                    min_utility_gain,
                )
                text = build_candidate_text(
                    pairs[row_index], source_index, max_text_chars
                )
            examples.append(
                {
                    "text": text,
                    "task": TASK_RANKER,
                    "label": label,
                    "row_index": row_index,
                    "source_index": source_index,
                }
            )
        sampled_candidates = (
            pairs[row_index].get("sampled_augmentation_candidates", [])
            if include_sampled_augmentation
            else []
        )
        for offset, sampled in enumerate(sampled_candidates):
            candidate = sampled["candidate"]
            examples.append(
                {
                    "text": _train_only_candidate_text(
                        pairs[row_index],
                        candidate,
                        sampled["source_tag"],
                        max_text_chars,
                    ),
                    "task": TASK_RANKER,
                    "label": _ranker_training_label(
                        direct_candidate,
                        candidate,
                        ranker_target_name,
                        min_utility_gain,
                    ),
                    "row_index": row_index,
                    "source_index": 4 + offset,
                }
            )
    positives = [item for item in examples if item["label"] == 1]
    negatives = [item for item in examples if item["label"] == 0]
    if negative_ratio > 0 and positives:
        keep = min(len(negatives), int(math.ceil(len(positives) * negative_ratio)))
        rng = random.Random(int(seed))
        negatives = rng.sample(negatives, keep)
    return positives + negatives


def _ranker_training_label(
    direct_candidate: Dict[str, Any],
    candidate: Dict[str, Any],
    ranker_target_name: str,
    min_utility_gain: float = 0.05,
) -> int:
    direct_correct = _correct(direct_candidate)
    candidate_correct = _correct(candidate)
    if ranker_target_name == "candidate_correctness":
        return candidate_correct
    if ranker_target_name == "repair_advantage":
        return int(not direct_correct and candidate_correct)
    if ranker_target_name == "utility_improvement":
        direct_utility = _candidate_utility(direct_candidate)
        candidate_utility = _candidate_utility(candidate)
        return int(candidate_utility > direct_utility + float(min_utility_gain))
    raise ValueError(f"Unknown ranker target: {ranker_target_name}")


def _gate_training_label(
    pair: Dict[str, Any],
    targets: Dict[str, np.ndarray],
    row_index: int,
    gate_target_name: str,
    include_sampled_augmentation: bool,
    min_utility_gain: float = 0.05,
) -> int:
    if gate_target_name == "direct_error":
        return int(targets["direct_error_target"][row_index])
    if gate_target_name not in ("union_repair", "union_utility_gain"):
        raise ValueError(f"Unknown gate target: {gate_target_name}")

    direct = pair["new_candidates"]["direct"]
    if gate_target_name == "union_repair":
        label = int(targets["gate_target"][row_index])
    elif not pair["complete"]:
        label = 0
    else:
        direct_utility = _candidate_utility(direct)
        label = int(
            any(
                _candidate_utility(candidate)
                > direct_utility + float(min_utility_gain)
                for candidate in pair_candidates(pair)[1:]
            )
        )
    if label or not include_sampled_augmentation:
        return label
    sampled = pair.get("sampled_augmentation_candidates", [])
    if gate_target_name == "union_repair":
        sampled_fix = (not _correct(direct)) and any(
            _correct(item.get("candidate")) for item in sampled
        )
    else:
        direct_utility = _candidate_utility(direct)
        sampled_fix = any(
            _candidate_utility(item.get("candidate") or {})
            > direct_utility + float(min_utility_gain)
            for item in sampled
        )
    return int(sampled_fix)


def _build_examples(
    pairs: Sequence[Dict[str, Any]],
    targets: Dict[str, np.ndarray],
    row_indices: Iterable[int],
    max_text_chars: int,
    negative_ratio: float | None,
    seed: int,
    include_reflect_2_augmentation: bool = False,
    include_sampled_augmentation: bool = False,
    gate_target_name: str = "direct_error",
    ranker_target_name: str = "candidate_correctness",
    min_utility_gain: float = 0.05,
) -> List[Dict[str, Any]]:
    row_indices = [int(index) for index in row_indices]
    gate = [
        {
            "text": build_policy_text(
                pairs[index]["new_row"],
                max_text_chars=max_text_chars,
                include_state=False,
            ),
            "task": TASK_GATE,
            "label": _gate_training_label(
                pairs[index],
                targets,
                index,
                gate_target_name,
                include_sampled_augmentation,
                min_utility_gain,
            ),
            "row_index": index,
            "source_index": 0,
        }
        for index in row_indices
    ]
    ranker = _candidate_training_examples(
        pairs,
        targets,
        row_indices,
        max_text_chars,
        float(negative_ratio or 0.0),
        seed,
        include_reflect_2_augmentation,
        include_sampled_augmentation,
        ranker_target_name,
        min_utility_gain,
    )
    return gate + ranker


def _data_loader(
    tokenizer: Any,
    examples: List[Dict[str, Any]],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch, _, DataLoader, Dataset = _import_torch()

    class TextDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return examples[index]

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokens = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        tokens["task_ids"] = torch.tensor(
            [item["task"] for item in batch], dtype=torch.long
        )
        tokens["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.float32
        )
        tokens["row_indices"] = torch.tensor(
            [item["row_index"] for item in batch], dtype=torch.long
        )
        tokens["source_indices"] = torch.tensor(
            [item["source_index"] for item in batch], dtype=torch.long
        )
        return tokens

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TextDataset(),
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _positive_weights(examples: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    result = {}
    for task in (TASK_GATE, TASK_RANKER):
        labels = [int(item["label"]) for item in examples if item["task"] == task]
        positives = sum(labels)
        negatives = len(labels) - positives
        ratio = negatives / max(positives, 1)
        result[task] = float(min(8.0, max(1.0, math.sqrt(ratio))))
    return result


def _move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: str,
    positive_weights: Dict[int, float],
    gradient_accumulation: int,
    gradient_clip: float,
) -> float:
    import torch.nn.functional as functional

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    steps = len(loader)
    for step, batch in enumerate(loader, 1):
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        batch.pop("row_indices")
        batch.pop("source_indices")
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
        ):
            logits = model(**batch)
            losses = functional.binary_cross_entropy_with_logits(
                logits.float(), labels, reduction="none"
            )
            weights = torch.ones_like(losses)
            for task, value in positive_weights.items():
                weights = torch.where(
                    (batch["task_ids"] == int(task)) & (labels > 0.5),
                    torch.full_like(weights, float(value)),
                    weights,
                )
            loss = (losses * weights).mean()
        (loss / max(1, int(gradient_accumulation))).backward()
        if step % max(1, int(gradient_accumulation)) == 0 or step == steps:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(gradient_clip),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if step == 1 or step % 100 == 0 or step == steps:
            print(
                f"[top-layer] train step={step}/{steps} "
                f"loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )
    return total_loss / max(total_examples, 1)


def _predict_examples(
    torch: Any,
    model: Any,
    loader: Any,
    device: str,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    import torch.nn.functional as functional

    model.eval()
    probabilities = []
    labels = []
    row_indices = []
    source_indices = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            batch = _move_batch(batch, device)
            target = batch.pop("labels")
            rows = batch.pop("row_indices")
            sources = batch.pop("source_indices")
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
            ):
                logits = model(**batch)
            loss = functional.binary_cross_entropy_with_logits(
                logits.float(), target, reduction="sum"
            )
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(target.cpu().numpy())
            row_indices.append(rows.cpu().numpy())
            source_indices.append(sources.cpu().numpy())
            loss_sum += float(loss.cpu())
            count += int(target.shape[0])
            if progress_label and (
                batch_index == 1
                or batch_index % 100 == 0
                or batch_index == len(loader)
            ):
                print(
                    f"[top-layer] {progress_label} batch="
                    f"{batch_index}/{len(loader)}",
                    flush=True,
                )
    return (
        np.concatenate(probabilities) if probabilities else np.asarray([]),
        np.concatenate(labels) if labels else np.asarray([]),
        np.concatenate(row_indices) if row_indices else np.asarray([], dtype="int64"),
        np.concatenate(source_indices) if source_indices else np.asarray([], dtype="int64"),
        loss_sum / max(count, 1),
    )


def _safe_binary_metrics(labels: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not len(labels):
        return {"accuracy": 0.0, "roc_auc": 0.0, "average_precision": 0.0}
    prediction = probability >= 0.5
    result = {"accuracy": float(np.mean(prediction == labels))}
    if len(np.unique(labels)) < 2:
        result.update({"roc_auc": 0.0, "average_precision": float(np.mean(labels))})
    else:
        result.update(
            {
                "roc_auc": float(roc_auc_score(labels, probability)),
                "average_precision": float(
                    average_precision_score(labels, probability)
                ),
            }
        )
    return result


def _validation_diagnostics(
    pairs: Sequence[Dict[str, Any]],
    targets: Dict[str, np.ndarray],
    validation_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    sources: np.ndarray,
    max_invocation_rate: float,
    lambda_cost: float,
    grid_points: int,
    selection_score_mode: str,
) -> Dict[str, Any]:
    local = {int(global_index): index for index, global_index in enumerate(validation_indices)}
    gate_probability = np.zeros(len(validation_indices), dtype="float32")
    candidate_probability = np.zeros(
        (len(validation_indices), len(CANDIDATE_SOURCES)), dtype="float32"
    )
    gate_labels = []
    gate_probs = []
    ranker_labels = []
    ranker_probs = []
    for probability, label, global_index, source_index in zip(
        probabilities, labels, rows, sources
    ):
        local_index = local[int(global_index)]
        if int(source_index) == 0:
            gate_probability[local_index] = float(probability)
            gate_labels.append(float(label))
            gate_probs.append(float(probability))
        else:
            candidate_probability[local_index, int(source_index)] = float(probability)
            ranker_labels.append(float(label))
            ranker_probs.append(float(probability))
    candidate_probability[:, 0] = 1.0 - gate_probability
    val_pairs = [pairs[int(index)] for index in validation_indices]
    val_targets = build_dual_targets(val_pairs)
    usage = build_usage_arrays(val_pairs)
    gate_threshold, margin_threshold, selection = select_dual_thresholds(
        gate_probability,
        candidate_probability,
        val_targets,
        usage,
        max_invocation_rate=max_invocation_rate,
        lambda_cost=lambda_cost,
        grid_points=grid_points,
        selection_score_mode=selection_score_mode,
    )
    return {
        "gate": _safe_binary_metrics(
            np.asarray(gate_labels), np.asarray(gate_probs)
        ),
        "ranker": _safe_binary_metrics(
            np.asarray(ranker_labels), np.asarray(ranker_probs)
        ),
        "gate_threshold": float(gate_threshold),
        "margin_threshold": float(margin_threshold),
        "selection": selection,
    }


def _checkpoint_state(model: Any) -> Dict[str, Any]:
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def _plot_history(history: Sequence[Dict[str, Any]], output: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
    axes[0].set_title("Multitask BCE")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["gate_roc_auc"] for item in history], label="gate ROC-AUC")
    axes[1].plot(epochs, [item["gate_average_precision"] for item in history], label="gate AP")
    axes[1].plot(epochs, [item["ranker_average_precision"] for item in history], label="ranker AP")
    axes[1].set_title("Validation discrimination")
    axes[1].legend()
    axes[2].plot(epochs, [item["selected_accuracy"] for item in history], label="selected accuracy")
    axes[2].plot(epochs, [item["direct_accuracy"] for item in history], label="direct accuracy")
    axes[2].plot(epochs, [item["invocation_rate"] for item in history], label="invocation rate")
    axes[2].set_title("Validation policy")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def train_top_layer_policy(
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    output: str,
    encoder_name: str,
    curve_output: str,
    history_output: str,
    trainable_blocks: int = 2,
    layer_indices: Sequence[int] = (-2, -1),
    pooling: str = "mean_last",
    hidden_sizes: Sequence[int] = (128, 32),
    dropout: float = 0.2,
    max_length: int = 2048,
    max_text_chars: int = 8000,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    gradient_accumulation: int = 4,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    validation_size: float = 0.2,
    negative_ratio: float = 8.0,
    include_reflect_2_augmentation: bool = True,
    patience: int = 2,
    gradient_clip: float = 1.0,
    random_state: int = 42,
    device: str = "cuda",
    max_invocation_rate: float = 0.15,
    lambda_cost: float = 0.0001,
    threshold_grid_points: int = 81,
    limit: int | None = None,
    augmentation_datasets: Sequence[Sequence[Dict[str, Any]]] | None = None,
    gate_target_name: str = "direct_error",
    ranker_target_name: str = "candidate_correctness",
    selection_score_mode: str = "correctness_margin",
    min_utility_gain: float = 0.05,
) -> Dict[str, Any]:
    torch, _, _, _ = _import_torch()
    _seed_everything(torch, random_state)
    if limit is not None:
        old_rows = old_rows[: int(limit)]
        new_rows = new_rows[: int(limit)]
        retained_ids = {str(row.get("id")) for row in new_rows}
        augmentation_datasets = [
            [
                row
                for row in dataset
                if str(row.get("id")) in retained_ids
            ]
            for dataset in (augmentation_datasets or [])
        ]
    pairs = align_dual_rows(old_rows, new_rows)
    augmentation_stats = attach_sampled_augmentations(
        pairs, augmentation_datasets or []
    )
    targets = build_dual_targets(pairs)
    train_indices, validation_indices = _split_rows(
        targets, validation_size, random_state
    )
    if not len(validation_indices):
        raise ValueError("A non-empty validation split is required")
    train_examples = _build_examples(
        pairs,
        targets,
        train_indices,
        max_text_chars,
        negative_ratio,
        random_state,
        include_reflect_2_augmentation=include_reflect_2_augmentation,
        include_sampled_augmentation=bool(augmentation_datasets),
        gate_target_name=gate_target_name,
        ranker_target_name=ranker_target_name,
        min_utility_gain=min_utility_gain,
    )
    validation_examples = _build_examples(
        pairs,
        targets,
        validation_indices,
        max_text_chars,
        None,
        random_state,
        gate_target_name=gate_target_name,
        ranker_target_name=ranker_target_name,
        min_utility_gain=min_utility_gain,
    )
    tokenizer, model, trainable_info, resolved_device = _load_model(
        encoder_name,
        trainable_blocks,
        layer_indices,
        pooling,
        hidden_sizes,
        dropout,
        device,
    )
    train_loader = _data_loader(
        tokenizer,
        train_examples,
        max_length,
        train_batch_size,
        True,
        random_state,
    )
    validation_loader = _data_loader(
        tokenizer,
        validation_examples,
        max_length,
        eval_batch_size,
        False,
        random_state,
    )
    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if "_head." in name else encoder_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(learning_rate)},
            {"params": head_parameters, "lr": float(head_learning_rate)},
        ],
        weight_decay=float(weight_decay),
    )
    positive_weights = _positive_weights(train_examples)
    train_gate_positive = sum(
        _gate_training_label(
            pairs[int(index)],
            targets,
            int(index),
            gate_target_name,
            bool(augmentation_datasets),
            min_utility_gain,
        )
        for index in train_indices
    )
    validation_gate_positive = sum(
        _gate_training_label(
            pairs[int(index)],
            targets,
            int(index),
            gate_target_name,
            False,
            min_utility_gain,
        )
        for index in validation_indices
    )
    print(
        f"[top-layer] rows train={len(train_indices)} validation={len(validation_indices)} "
        f"examples train={len(train_examples)} validation={len(validation_examples)}",
        flush=True,
    )
    print(
        f"[top-layer] blocks trainable={trainable_info['trainable_blocks']}/"
        f"{trainable_info['total_blocks']} feature_dim={model.feature_dim} "
        f"gate_target={gate_target_name} gate_positive="
        f"{train_gate_positive}/{validation_gate_positive} "
        f"ranker_target={ranker_target_name} "
        f"positive_weights={positive_weights}",
        flush=True,
    )

    history: List[Dict[str, Any]] = []
    best_key = None
    best_state = None
    best_epoch = 0
    best_diagnostics = None
    stale_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        train_loss = _train_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            resolved_device,
            positive_weights,
            gradient_accumulation,
            gradient_clip,
        )
        probability, label, row, source, validation_loss = _predict_examples(
            torch, model, validation_loader, resolved_device
        )
        diagnostics = _validation_diagnostics(
            pairs,
            targets,
            validation_indices,
            probability,
            label,
            row,
            source,
            max_invocation_rate,
            lambda_cost,
            threshold_grid_points,
            selection_score_mode,
        )
        selection = diagnostics["selection"]
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "gate_accuracy": diagnostics["gate"]["accuracy"],
            "gate_roc_auc": diagnostics["gate"]["roc_auc"],
            "gate_average_precision": diagnostics["gate"]["average_precision"],
            "ranker_accuracy": diagnostics["ranker"]["accuracy"],
            "ranker_roc_auc": diagnostics["ranker"]["roc_auc"],
            "ranker_average_precision": diagnostics["ranker"]["average_precision"],
            "selected_accuracy": selection["accuracy"],
            "direct_accuracy": selection["direct_accuracy"],
            "selected_reward": selection["avg_reward"],
            "invocation_rate": selection["gate_invocation_rate"],
            "adoption_rate": selection["candidate_adoption_rate"],
            "fixes": selection["fixes"],
            "harms": selection["harms"],
            "net_fixes": selection["net_fixes"],
            "gate_threshold": diagnostics["gate_threshold"],
            "margin_threshold": diagnostics["margin_threshold"],
        }
        history.append(epoch_row)
        print(f"[top-layer] epoch={epoch} {epoch_row}", flush=True)
        key = (
            selection["accuracy"],
            selection["avg_reward"],
            selection["net_fixes"],
            -selection["gate_invocation_rate"],
            -validation_loss,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = _checkpoint_state(model)
            best_epoch = epoch
            best_diagnostics = copy.deepcopy(diagnostics)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                print(f"[top-layer] early stopping at epoch {epoch}", flush=True)
                break
    if best_state is None or best_diagnostics is None:
        raise RuntimeError("No valid policy checkpoint was produced")

    _plot_history(history, curve_output)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_diagnostics,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "selection_rule": (
            "maximize validation selected-answer accuracy, then reward and net fixes; "
            "test labels are never used"
        ),
        "test_labels_used": False,
    }
    save_json(metadata, history_output)
    bundle = {
        "policy_type": POLICY_TYPE,
        "candidate_sources": list(CANDIDATE_SOURCES),
        "encoder_name": encoder_name,
        "trainable_blocks": int(trainable_blocks),
        "layer_indices": list(layer_indices),
        "pooling": pooling,
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "max_length": int(max_length),
        "max_text_chars": int(max_text_chars),
        "eval_batch_size": int(eval_batch_size),
        "state_dict": best_state,
        "gate_threshold": float(best_diagnostics["gate_threshold"]),
        "margin_threshold": float(best_diagnostics["margin_threshold"]),
        "max_invocation_rate": float(max_invocation_rate),
        "lambda_cost": float(lambda_cost),
        "n_rows": len(pairs),
        "n_complete": int(np.sum(targets["complete"])),
        "gate_target_name": gate_target_name,
        "ranker_target_name": ranker_target_name,
        "selection_score_mode": selection_score_mode,
        "min_utility_gain": float(min_utility_gain),
        "n_positive": int(
            np.sum(
                targets[
                    "direct_error_target"
                    if gate_target_name == "direct_error"
                    else "gate_target"
                ]
            )
        ),
        "n_train_gate_positive": int(train_gate_positive),
        "n_validation_gate_positive": int(validation_gate_positive),
        "n_union_repair": int(np.sum(targets["gate_target"])),
        "reflect_2_train_augmentation": bool(include_reflect_2_augmentation),
        "sampled_augmentation": augmentation_stats,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "best_epoch": int(best_epoch),
        "best_validation": best_diagnostics,
        "trainable_info": trainable_info,
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "training_objective": (
            f"shared Qwen3 encoder: gate predicts {gate_target_name}; ranker predicts "
            f"{ranker_target_name}; v5 reflect_2 and stochastic v7 samples are "
            f"train-only augmentations; replacement uses {selection_score_mode}; "
            f"minimum utility gain={float(min_utility_gain):.4f}"
        ),
        "feature_policy": (
            "pure question/context/response text; no gold answer, reward, verifier score, "
            "or manually designed state features"
        ),
        "token_accounting": (
            "on invocation: Base once + v5 reflection once + v7 reflection once"
        ),
        "curve_output": curve_output,
        "history_output": history_output,
        "test_labels_used": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_pickle(bundle, output)
    return bundle


def predict_top_layer_dual_candidate(
    bundle: Dict[str, Any],
    old_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if bundle.get("policy_type") != POLICY_TYPE:
        raise ValueError(f"Unexpected policy_type={bundle.get('policy_type')}")
    torch, _, _, _ = _import_torch()
    pairs = align_dual_rows(old_rows, new_rows, require_complete=True)
    tokenizer, model, _, device = _load_model(
        bundle["encoder_name"],
        int(bundle["trainable_blocks"]),
        bundle["layer_indices"],
        bundle["pooling"],
        bundle["hidden_sizes"],
        float(bundle["dropout"]),
        bundle.get("device", "cuda"),
        state_dict=bundle["state_dict"],
    )
    examples = _build_examples(
        pairs,
        build_dual_targets(pairs),
        range(len(pairs)),
        int(bundle["max_text_chars"]),
        None,
        0,
        gate_target_name=bundle.get("gate_target_name", "direct_error"),
        ranker_target_name=bundle.get(
            "ranker_target_name", "candidate_correctness"
        ),
    )
    loader = _data_loader(
        tokenizer,
        examples,
        int(bundle["max_length"]),
        int(bundle.get("eval_batch_size", 8)),
        False,
        0,
    )
    probability, _, rows, sources, _ = _predict_examples(
        torch, model, loader, device, progress_label="offline inference"
    )
    gate_probability = np.zeros(len(pairs), dtype="float32")
    candidate_probability = np.zeros(
        (len(pairs), len(CANDIDATE_SOURCES)), dtype="float32"
    )
    for value, row_index, source_index in zip(probability, rows, sources):
        if int(source_index) == 0:
            gate_probability[int(row_index)] = float(value)
        else:
            candidate_probability[int(row_index), int(source_index)] = float(value)
    candidate_probability[:, 0] = 1.0 - gate_probability
    row_indices = np.arange(len(pairs))
    best_reflection, margin = reflection_decision_score(
        candidate_probability,
        bundle.get("selection_score_mode", "correctness_margin"),
    )
    invoked = gate_probability >= float(bundle["gate_threshold"])
    adopted = invoked & (margin >= float(bundle["margin_threshold"])
    )
    selected_source = np.where(adopted, best_reflection, 0)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pairs, {
        "gate_probability": gate_probability,
        "candidate_probability": candidate_probability,
        "best_reflection": best_reflection,
        "margin": margin,
        "invoked": invoked,
        "adopted": adopted,
        "selected_source": selected_source,
    }


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a QA policy by fine-tuning only the final encoder blocks."
    )
    parser.add_argument("--old_dataset", required=True)
    parser.add_argument("--new_dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument("--curve_output", required=True)
    parser.add_argument("--history_output", required=True)
    parser.add_argument("--trainable_blocks", type=int, default=2)
    parser.add_argument("--layer_indices", default="-2,-1")
    parser.add_argument("--pooling", choices=("mean", "last", "mean_last"), default="mean_last")
    parser.add_argument("--hidden_sizes", default="128,32")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_text_chars", type=int, default=8000)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--head_learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--negative_ratio", type=float, default=8.0)
    parser.add_argument(
        "--gate_target",
        choices=("direct_error", "union_repair", "union_utility_gain"),
        default="direct_error",
    )
    parser.add_argument(
        "--ranker_target",
        choices=(
            "candidate_correctness",
            "repair_advantage",
            "utility_improvement",
        ),
        default="candidate_correctness",
    )
    parser.add_argument("--min_utility_gain", type=float, default=0.05)
    parser.add_argument(
        "--selection_score_mode",
        choices=("correctness_margin", "repair_probability"),
        default="correctness_margin",
    )
    parser.add_argument("--no_reflect_2_augmentation", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_invocation_rate", type=float, default=0.15)
    parser.add_argument("--lambda_cost", type=float, default=0.0001)
    parser.add_argument("--threshold_grid_points", type=int, default=81)
    parser.add_argument(
        "--augmentation_dataset",
        action="append",
        default=[],
        help="Train-only sampled reflection trace; repeat for multiple passes.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    bundle = train_top_layer_policy(
        read_jsonl(args.old_dataset),
        read_jsonl(args.new_dataset),
        args.output,
        args.encoder_name,
        args.curve_output,
        args.history_output,
        trainable_blocks=args.trainable_blocks,
        layer_indices=_integers(args.layer_indices),
        pooling=args.pooling,
        hidden_sizes=_integers(args.hidden_sizes),
        dropout=args.dropout,
        max_length=args.max_length,
        max_text_chars=args.max_text_chars,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        negative_ratio=args.negative_ratio,
        include_reflect_2_augmentation=not args.no_reflect_2_augmentation,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        random_state=args.random_state,
        device=args.device,
        max_invocation_rate=args.max_invocation_rate,
        lambda_cost=args.lambda_cost,
        threshold_grid_points=args.threshold_grid_points,
        limit=args.limit,
        augmentation_datasets=[
            read_jsonl(path) for path in args.augmentation_dataset
        ],
        gate_target_name=args.gate_target,
        ranker_target_name=args.ranker_target,
        selection_score_mode=args.selection_score_mode,
        min_utility_gain=args.min_utility_gain,
    )
    print(f"Saved top-layer policy to {args.output}")
    print(
        f"Best epoch={bundle['best_epoch']} "
        f"validation={bundle['best_validation']['selection']}"
    )
    print(
        f"Trainable parameters={bundle['trainable_parameters']}/"
        f"{bundle['total_parameters']}"
    )


if __name__ == "__main__":
    main()
