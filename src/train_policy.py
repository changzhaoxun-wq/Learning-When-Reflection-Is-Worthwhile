import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .utils import load_pickle, project_root, read_jsonl, save_pickle


ACTION_LABELS = ["direct", "reflect_1", "reflect_2"]
FEATURE_NAMES = [
    "confidence_score",
    "verifier_score",
    "answer_length",
    "round_id",
    "token_cost",
    "verifier_token_cost",
    "difficulty_score",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "task_math",
    "task_qa",
    "question_length",
    "context_length",
    "question_num_count",
    "question_entity_count",
    "wh_word_count",
    "initial_pred_missing",
    "initial_answer_has_final_marker",
]


class MajorityPolicy:
    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, xs: List[List[float]]) -> List[str]:
        return [self.label for _ in xs]


def featurize_row(row: Dict[str, Any], feature_names: List[str] | None = None) -> List[float]:
    features = _feature_dict(row)
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]


def _feature_dict(row: Dict[str, Any]) -> Dict[str, float]:
    state = row.get("state_features", {})
    question = row.get("question", state.get("question", ""))
    initial_pred = row.get("initial_pred")
    task_type = str(row.get("task_type") or state.get("task_type") or "math").lower()
    return {
        "confidence_score": float(state.get("confidence_score", 0.0)),
        "verifier_score": float(state.get("verifier_score", 0.0)),
        "answer_length": float(state.get("answer_length", 0.0)),
        "round_id": float(state.get("round_id", 0.0)),
        "token_cost": float(state.get("token_cost", 0.0)),
        "verifier_token_cost": float(state.get("verifier_token_cost", 0.0)),
        "difficulty_score": float(state.get("difficulty_score", row.get("difficulty_score", 0.0))),
        "difficulty_easy": float(state.get("difficulty_easy", 1.0 if row.get("difficulty") == "easy" else 0.0)),
        "difficulty_medium": float(state.get("difficulty_medium", 1.0 if row.get("difficulty") == "medium" else 0.0)),
        "difficulty_hard": float(state.get("difficulty_hard", 1.0 if row.get("difficulty") == "hard" else 0.0)),
        "task_math": float(state.get("task_math", 1.0 if task_type == "math" else 0.0)),
        "task_qa": float(state.get("task_qa", 1.0 if task_type == "qa" else 0.0)),
        "question_length": float(state.get("question_length", len(_strip_context_prefix(question).split()))),
        "context_length": float(state.get("context_length", _context_length(question))),
        "question_num_count": float(state.get("question_num_count", _count_numbers(_strip_context_prefix(question)))),
        "question_entity_count": float(state.get("question_entity_count", _count_capitalized_phrases(_strip_context_prefix(question)))),
        "wh_word_count": float(state.get("wh_word_count", _count_wh_words(_strip_context_prefix(question)))),
        "initial_pred_missing": float(initial_pred is None),
        "initial_answer_has_final_marker": float(state.get("initial_answer_has_final_marker", 0.0)),
    }


def train_policy(rows: List[Dict[str, Any]], model_type: str = "random_forest") -> Dict[str, Any]:
    xs = [featurize_row(row) for row in rows]
    ys = [row["best_action"] for row in rows]
    if not rows:
        raise ValueError("Cannot train policy: empty dataset.")

    try:
        if model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                class_weight="balanced_subsample",
            )
        model.fit(xs, ys)
        backend = f"sklearn:{model_type}"
    except Exception as exc:
        majority = Counter(ys).most_common(1)[0][0]
        model = MajorityPolicy(majority)
        backend = f"majority_fallback:{exc}"

    preds = list(model.predict(xs))
    train_acc = sum(int(p == y) for p, y in zip(preds, ys)) / len(ys)
    action_labels = [action for action in ACTION_LABELS if action in set(ys)]
    action_labels.extend(sorted(set(ys) - set(action_labels)))
    return {
        "model": model,
        "backend": backend,
        "feature_names": FEATURE_NAMES,
        "action_labels": action_labels,
        "train_accuracy": train_acc,
        "label_distribution": dict(Counter(ys)),
    }


def predict_action(bundle: Dict[str, Any], row: Dict[str, Any]) -> str:
    # Keep old four-action policy files loadable after verify_reflect removal.
    if bundle.get("policy_type") == "embedding_mlp":
        from .neural_policy import predict_neural_action

        action = predict_neural_action(bundle, row)
    elif bundle.get("policy_type") == "utility_mlp":
        from .utility_policy import predict_utility_action

        action = predict_utility_action(bundle, row)
    elif bundle.get("policy_type") == "actor_mlp":
        from .actor_policy import predict_actor_action

        action = predict_actor_action(bundle, row)
    elif bundle.get("policy_type") == "torch_embedding_mlp":
        from .transformer_policy import predict_transformer_action

        action = predict_transformer_action(bundle, row)
    elif bundle.get("policy_type") == "selective_advantage_mlp":
        from .selective_advantage_policy import predict_action as predict_advantage_action

        action = predict_advantage_action(bundle, row)
    else:
        model = bundle["model"]
        feature_names = bundle.get("feature_names") or FEATURE_NAMES
        action = str(model.predict([featurize_row(row, feature_names=feature_names)])[0])
    return "reflect_1" if action == "verify_reflect" else action


def _strip_context_prefix(question: str) -> str:
    text = str(question or "")
    marker = "Question:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()


def _context_length(question: str) -> int:
    text = str(question or "")
    marker = "Question:"
    if marker not in text:
        return 0
    return len(text.split(marker, 1)[0].split())


def _count_numbers(text: str) -> int:
    import re

    return len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text)))


def _count_capitalized_phrases(text: str) -> int:
    import re

    return len(re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", str(text)))


def _count_wh_words(text: str) -> int:
    import re

    wh_words = {"who", "what", "when", "where", "which", "why", "how", "whose", "whom"}
    return sum(1 for word in re.findall(r"[a-zA-Z]+", str(text).lower()) if word in wh_words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(project_root() / "outputs" / "policy_dataset.jsonl"))
    parser.add_argument("--output", default=str(project_root() / "outputs" / "policy_model.pkl"))
    parser.add_argument("--model_type", choices=["random_forest", "logistic_regression"], default="random_forest")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    bundle = train_policy(rows, model_type=args.model_type)
    save_pickle(bundle, args.output)
    print(f"Saved policy model to {args.output}")
    print(f"Backend: {bundle['backend']}")
    print(f"Train accuracy: {bundle['train_accuracy']:.4f}")
    print(f"Label distribution: {bundle['label_distribution']}")


if __name__ == "__main__":
    main()
