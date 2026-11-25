from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from .train_policy import FEATURE_NAMES, featurize_row
from .utils import read_jsonl, save_pickle


ACTION_LABELS = ["direct", "reflect_1", "reflect_2"]
_SENTENCE_MODEL_CACHE: Dict[str, Any] = {}
_HF_MODEL_CACHE: Dict[tuple[str, str], Any] = {}
_TORCHVISION_STUB_LIBRARY: Any = None


def _qa_question_and_context(row: Dict[str, Any], state: Dict[str, Any]) -> tuple[str, str]:
    raw_question = str(row.get("raw_question") or "").strip()
    context = str(row.get("context") or "").strip()
    combined = str(row.get("question") or state.get("question") or "")
    marker = "\n\nQuestion:\n"
    if (not raw_question or not context) and combined.startswith("Context:\n") and marker in combined:
        parsed_context, parsed_question = combined[len("Context:\n"):].rsplit(marker, 1)
        context = context or parsed_context.strip()
        raw_question = raw_question or parsed_question.strip()
    return raw_question or combined, context


def _clip_context(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = "\n[... context middle omitted ...]\n"
    usable = max(0, budget - len(marker))
    head = int(usable * 0.75)
    tail = usable - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def build_policy_text(row: Dict[str, Any], max_text_chars: int = 8000, include_state: bool = True) -> str:
    """Build a policy input without exposing gold answers or candidate rewards.

    For QA, the actual question and Base answer are placed before context so
    tokenizer-side truncation cannot remove them. Only the context is shortened,
    preserving both its evidence-prioritized head and a small tail.
    """
    state = row.get("state_features", {}) or {}
    question = str(row.get("question") or state.get("question") or "")
    initial_answer = str(row.get("initial_answer") or state.get("initial_answer") or "")
    initial_pred = row.get("initial_pred")
    task_type = str(row.get("task_type") or state.get("task_type") or "math")
    state_parts = []
    if include_state:
        state_parts = [
            f"Task type: {task_type}",
            f"Difficulty: {row.get('difficulty') or state.get('difficulty') or 'unknown'}",
            f"Verifier score: {state.get('verifier_score', 0.0)}",
            f"Confidence score: {state.get('confidence_score', 0.0)}",
            f"Verifier reason: {state.get('verifier_reason', '')}",
            f"Initial prediction: {initial_pred}",
            f"Token cost: {state.get('token_cost', 0)}",
            f"Answer length: {state.get('answer_length', 0)}",
        ]

    if task_type.lower() == "qa":
        raw_question, context = _qa_question_and_context(row, state)
        fixed_parts = [*state_parts, "Question:", raw_question, "Initial answer:", initial_answer]
        fixed_text = "\n".join(fixed_parts)
        if not context:
            return fixed_text[:max_text_chars] if max_text_chars else fixed_text
        context_header = "\nContext:\n"
        context_budget = len(context)
        if max_text_chars:
            context_budget = max(0, max_text_chars - len(fixed_text) - len(context_header))
        text = fixed_text + context_header + _clip_context(context, context_budget)
        return text[:max_text_chars] if max_text_chars else text

    if include_state:
        parts = [*state_parts, "Question:", question, "Initial answer:", initial_answer]
    else:
        parts = ["Question:", question, "Initial answer:", initial_answer]
    text = "\n".join(parts)
    if max_text_chars and len(text) > max_text_chars:
        text = text[:max_text_chars]
    return text


def train_embedding_mlp(
    rows: List[Dict[str, Any]],
    backend: str = "tfidf_svd",
    encoder_name: str = "BAAI/bge-small-en-v1.5",
    embedding_dim: int = 512,
    max_features: int = 30000,
    hidden_sizes: Sequence[int] = (256, 64),
    max_iter: int = 300,
    test_size: float = 0.2,
    random_state: int = 42,
    max_text_chars: int = 8000,
    early_stopping: bool = True,
    alpha: float = 1e-3,
    max_length: int = 1024,
    batch_size: int = 8,
    device: str = "auto",
    pooling: str = "mean",
    balance_classes: bool = True,
    pure_text: bool = False,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot train neural policy: empty dataset.")
    texts = [build_policy_text(row, max_text_chars=max_text_chars, include_state=not pure_text) for row in rows]
    labels = [str(row["best_action"]) for row in rows]
    label_counts = Counter(labels)

    train_idx, val_idx = _split_indices(labels, test_size=test_size, random_state=random_state)
    numeric_scaler = None
    if backend in {"hf_hidden_mlp", "hf_hybrid_mlp"}:
        features, encoder_info = _encode_hf_hidden(
            texts,
            encoder_name=encoder_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            pooling=pooling,
        )
        vectorizer = None
        reducer = None
        # Hidden states from fp16 causal LMs can contain inf values or have
        # a scale that is too large for sklearn MLP training. Clean and
        # standardize them using the training split only.
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -1e6, 1e6)
        scaler = StandardScaler()
        scaler.fit(features[train_idx])
        features = scaler.transform(features).astype("float32")
        if backend == "hf_hybrid_mlp":
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            numeric = np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES) for row in rows],
                dtype="float32",
            )
            numeric_scaler = StandardScaler()
            numeric = numeric_scaler.fit_transform(numeric)
            features = np.hstack([features, numeric])
            encoder_info = {
                **encoder_info,
                "name": "hf_hidden_hybrid",
                "numeric_features": len(FEATURE_NAMES),
                "total_features": int(features.shape[1]),
            }
    elif backend == "tiny_structured_mlp":
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        vectorizer = None
        reducer = None
        numeric_scaler = None
        scaler = StandardScaler()
        features = scaler.fit_transform(
            np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES) for row in rows],
                dtype="float32",
            )
        )
        encoder_info = {
            "name": "tiny_structured",
            "numeric_features": len(FEATURE_NAMES),
            "total_features": int(features.shape[1]),
        }
    elif backend == "sentence_transformer":
        features, encoder_info = _encode_sentence_transformer(texts, encoder_name=encoder_name)
        vectorizer = None
        reducer = None
        scaler = None
    elif backend in {"tfidf_svd", "hybrid_tfidf_svd"}:
        features, vectorizer, reducer, scaler, encoder_info = _encode_tfidf_svd(
            texts,
            embedding_dim=embedding_dim,
            max_features=max_features,
            random_state=random_state,
        )
        if backend == "hybrid_tfidf_svd":
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            numeric = np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES) for row in rows],
                dtype="float32",
            )
            numeric_scaler = StandardScaler()
            numeric = numeric_scaler.fit_transform(numeric)
            features = np.hstack([features, numeric])
            encoder_info = {
                **encoder_info,
                "name": "hybrid_tfidf_svd",
                "numeric_features": len(FEATURE_NAMES),
                "total_features": int(features.shape[1]),
            }
    else:
        raise ValueError(f"Unknown neural policy backend: {backend}")

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    model = MLPClassifier(
        hidden_layer_sizes=tuple(int(x) for x in hidden_sizes),
        activation="relu",
        solver="adam",
        alpha=float(alpha),
        batch_size="auto",
        learning_rate="adaptive",
        early_stopping=bool(early_stopping),
        validation_fraction=0.15,
        n_iter_no_change=30,
        max_iter=int(max_iter),
        random_state=int(random_state),
    )

    sample_weight = _balanced_sample_weight([labels[i] for i in train_idx]) if balance_classes else None
    if sample_weight is None:
        model.fit(features[train_idx], encoded_labels[train_idx])
    else:
        model.fit(features[train_idx], encoded_labels[train_idx], sample_weight=sample_weight)

    train_preds = list(label_encoder.inverse_transform(model.predict(features[train_idx]).astype(int)))
    train_labels = [labels[i] for i in train_idx]
    train_acc = _accuracy(train_preds, train_labels)
    val_acc = None
    if val_idx:
        val_preds = list(label_encoder.inverse_transform(model.predict(features[val_idx]).astype(int)))
        val_labels = [labels[i] for i in val_idx]
        val_acc = _accuracy(val_preds, val_labels)

    return {
        "policy_type": "embedding_mlp",
        "backend": backend,
        "encoder_info": encoder_info,
        "encoder_name": encoder_name,
        "vectorizer": vectorizer,
        "reducer": reducer,
        "scaler": scaler,
        "numeric_scaler": numeric_scaler,
        "model": model,
        "label_encoder": label_encoder,
        "hidden_sizes": list(hidden_sizes),
        "max_text_chars": max_text_chars,
        "early_stopping": bool(early_stopping),
        "alpha": float(alpha),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "device": device,
        "pooling": pooling,
        "balance_classes": bool(balance_classes),
        "pure_text": bool(pure_text),
        "action_labels": list(label_encoder.classes_),
        "label_distribution": dict(label_counts),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }


def predict_neural_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    text = build_policy_text(row, max_text_chars=int(bundle.get("max_text_chars", 8000)), include_state=not bool(bundle.get("pure_text", False)))
    backend = bundle.get("backend", "tfidf_svd")
    if backend in {"hf_hidden_mlp", "hf_hybrid_mlp"}:
        features, _ = _encode_hf_hidden(
            [text],
            encoder_name=bundle.get("encoder_name"),
            max_length=int(bundle.get("max_length", 1024)),
            batch_size=1,
            device=bundle.get("device", "auto"),
            pooling=bundle.get("pooling", "mean"),
        )
        if bundle.get("scaler") is not None:
            features = bundle["scaler"].transform(features).astype("float32")
        if backend == "hf_hybrid_mlp":
            import numpy as np

            numeric = np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES)],
                dtype="float32",
            )
            numeric_scaler = bundle.get("numeric_scaler")
            if numeric_scaler is None:
                raise ValueError("HF hybrid neural policy bundle is missing numeric_scaler")
            numeric = numeric_scaler.transform(numeric)
            features = np.hstack([features, numeric])
    elif backend == "tiny_structured_mlp":
        import numpy as np

        features = bundle["scaler"].transform(
            np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES)],
                dtype="float32",
            )
        )
    elif backend == "sentence_transformer":
        features, _ = _encode_sentence_transformer([text], encoder_name=bundle.get("encoder_name", "BAAI/bge-small-en-v1.5"))
    elif backend in {"tfidf_svd", "hybrid_tfidf_svd"}:
        import numpy as np

        x = bundle["vectorizer"].transform([text])
        reducer = bundle.get("reducer")
        scaler = bundle.get("scaler")
        text_features = reducer.transform(x) if reducer is not None else x.toarray()
        if scaler is not None:
            text_features = scaler.transform(text_features)
        if backend == "hybrid_tfidf_svd":
            numeric = np.asarray(
                [featurize_row(row, feature_names=FEATURE_NAMES)],
                dtype="float32",
            )
            numeric_scaler = bundle.get("numeric_scaler")
            if numeric_scaler is None:
                raise ValueError("Hybrid neural policy bundle is missing numeric_scaler")
            numeric = numeric_scaler.transform(numeric)
            features = np.hstack([text_features, numeric])
        else:
            features = text_features
    else:
        raise ValueError(f"Unknown neural policy backend: {backend}")
    raw_predictions = bundle["model"].predict(features)
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        return str(label_encoder.inverse_transform(raw_predictions.astype(int))[0])
    return str(raw_predictions[0])


def train_from_jsonl(dataset_path: str, output_path: str, **kwargs: Any) -> Dict[str, Any]:
    rows = read_jsonl(dataset_path)
    bundle = train_embedding_mlp(rows, **kwargs)
    save_pickle(bundle, output_path)
    return bundle


def _encode_tfidf_svd(
    texts: List[str],
    embedding_dim: int,
    max_features: int,
    random_state: int,
) -> tuple[Any, Any, Any, Any, Dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_features=int(max_features),
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(texts)
    max_components = max(1, min(int(embedding_dim), x.shape[0] - 1, x.shape[1] - 1))
    if max_components < 2:
        features = x.toarray()
        reducer = None
    else:
        reducer = TruncatedSVD(n_components=max_components, random_state=int(random_state))
        features = reducer.fit_transform(x)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, vectorizer, reducer, scaler, {
        "name": "tfidf_svd",
        "embedding_dim": int(features.shape[1]),
        "tfidf_features": int(x.shape[1]),
        "max_features": int(max_features),
    }


def _encode_sentence_transformer(texts: List[str], encoder_name: str) -> tuple[Any, Dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence_transformers backend is unavailable in this environment. "
            "Use --backend tfidf_svd or fix/install sentence-transformers."
        ) from exc
    model = _SENTENCE_MODEL_CACHE.get(encoder_name)
    if model is None:
        model = SentenceTransformer(encoder_name)
        _SENTENCE_MODEL_CACHE[encoder_name] = model
    features = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    return features, {"name": "sentence_transformer", "encoder_name": encoder_name, "embedding_dim": int(features.shape[1])}


def _encode_hf_hidden(
    texts: List[str],
    encoder_name: str,
    max_length: int,
    batch_size: int,
    device: str,
    pooling: str = "mean",
    layer_indices: Sequence[int] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Encode policy text with a small HuggingFace causal/base LM.

    The encoder is kept outside the policy pickle. Only the small MLP head and
    tokenizer-independent metadata are saved, so the policy artifact stays
    lightweight while the encoder remains replaceable.
    """
    import numpy as np
    import torch

    if pooling not in {"mean", "last", "mean_last"}:
        raise ValueError("pooling must be one of: mean, last, mean_last")

    _ensure_torchvision_nms_compat(torch)
    from transformers import AutoModel, AutoTokenizer

    if not encoder_name:
        raise ValueError("hf_hidden_mlp requires --encoder_name or ENCODER_NAME")
    if os.path.isabs(str(encoder_name)) and not os.path.isdir(str(encoder_name)):
        raise FileNotFoundError(
            f"Local encoder directory does not exist: {encoder_name}. "
            "Download the small model first or pass a valid HuggingFace model id."
        )
    resolved_device = device
    if resolved_device in {"", "auto", None}:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (str(encoder_name), str(resolved_device))
    cached = _HF_MODEL_CACHE.get(cache_key)
    if cached is None:
        tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if str(resolved_device).startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch.float32
        model = AutoModel.from_pretrained(
            encoder_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        model.to(resolved_device)
        model.eval()
        cached = (tokenizer, model)
        _HF_MODEL_CACHE[cache_key] = cached
    tokenizer, model = cached

    embeddings = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch_texts = texts[start : start + max(1, int(batch_size))]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        batch = {key: value.to(resolved_device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(**batch, output_hidden_states=layer_indices is not None)
            mask = batch["attention_mask"]
            selected_hidden = (
                [output.hidden_states[index] for index in layer_indices]
                if layer_indices is not None
                else [output.last_hidden_state]
            )
            pooled_parts = []
            for hidden in selected_hidden:
                if pooling == "mean":
                    mask_float = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
                else:
                    lengths = mask.sum(dim=1).clamp_min(1) - 1
                    last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
                    if pooling == "last":
                        pooled = last
                    else:
                        mask_float = mask.unsqueeze(-1).to(hidden.dtype)
                        mean = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
                        pooled = torch.cat([mean, last], dim=1)
                pooled_parts.append(pooled)
            pooled = torch.cat(pooled_parts, dim=1)
        embeddings.append(pooled.float().cpu().numpy())

    if not embeddings:
        raise ValueError("Cannot encode empty text list with hf_hidden_mlp")
    features = np.concatenate(embeddings, axis=0).astype("float64")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -1e6, 1e6)
    return features, {
        "name": "hf_hidden_state",
        "encoder_name": encoder_name,
        "embedding_dim": int(features.shape[1]),
        "max_length": int(max_length),
        "device": str(resolved_device),
        "pooling": pooling,
        "layer_indices": list(layer_indices) if layer_indices is not None else [-1],
        "num_layers_used": len(layer_indices) if layer_indices is not None else 1,
    }


def _ensure_torchvision_nms_compat(torch: Any) -> None:
    """Define the missing torchvision nms schema for text-only Transformers use.

    Some environments contain a torchvision build whose compiled operators do
    not match torch. Transformers imports torchvision indirectly even for a
    text-only Qwen model, and its fake nms registration then fails. Defining
    the schema is sufficient here because this policy encoder never processes
    images.
    """
    global _TORCHVISION_STUB_LIBRARY
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError as exc:
        if "does not exist" not in str(exc):
            raise
    _TORCHVISION_STUB_LIBRARY = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_STUB_LIBRARY.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )


def _split_indices(labels: List[str], test_size: float, random_state: int) -> tuple[List[int], List[int]]:
    indices = list(range(len(labels)))
    if test_size <= 0 or len(indices) < 5:
        return indices, []
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 and len(counts) > 1 else None
    try:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=stratify,
        )
        return list(train_idx), list(val_idx)
    except Exception:
        return indices, []


def _balanced_sample_weight(labels: List[str]) -> List[float]:
    counts = Counter(labels)
    n = len(labels)
    k = max(1, len(counts))
    return [n / (k * counts[label]) for label in labels]


def _accuracy(preds: Iterable[str], labels: Iterable[str]) -> float:
    preds = list(preds)
    labels = list(labels)
    if not labels:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
