import argparse

from .neural_policy import train_from_jsonl
from .utils import project_root


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(',') if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description='Train an automatic-vector neural policy classifier.')
    parser.add_argument('--dataset', default=str(project_root() / 'outputs' / 'policy_dataset.jsonl'))
    parser.add_argument('--output', default=str(project_root() / 'outputs' / 'neural_policy_model.pkl'))
    parser.add_argument('--backend', choices=['tfidf_svd', 'hybrid_tfidf_svd', 'tiny_structured_mlp', 'hf_hidden_mlp', 'hf_hybrid_mlp', 'sentence_transformer'], default='tfidf_svd')
    parser.add_argument('--encoder_name', default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--max_features', type=int, default=30000)
    parser.add_argument('--hidden_sizes', default='256,64')
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--max_text_chars', type=int, default=8000)
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--pooling', choices=['mean', 'last', 'mean_last'], default='mean')
    parser.add_argument('--no_class_balance', action='store_true')
    parser.add_argument('--pure_text', action='store_true')
    args = parser.parse_args()

    bundle = train_from_jsonl(
        dataset_path=args.dataset,
        output_path=args.output,
        backend=args.backend,
        encoder_name=args.encoder_name,
        embedding_dim=args.embedding_dim,
        max_features=args.max_features,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        max_iter=args.max_iter,
        test_size=args.test_size,
        random_state=args.random_state,
        max_text_chars=args.max_text_chars,
        early_stopping=not args.no_early_stopping,
        alpha=args.alpha,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        pooling=args.pooling,
        balance_classes=not args.no_class_balance,
        pure_text=args.pure_text,
    )
    print(f"Saved neural policy model to {args.output}")
    print(f"Policy type: {bundle['policy_type']}")
    print(f"Backend: {bundle['backend']} {bundle['encoder_info']}")
    print(f"Train accuracy: {bundle['train_accuracy']:.4f}")
    if bundle.get('val_accuracy') is not None:
        print(f"Validation accuracy: {bundle['val_accuracy']:.4f}")
    print(f"Label distribution: {bundle['label_distribution']}")


if __name__ == '__main__':
    main()
