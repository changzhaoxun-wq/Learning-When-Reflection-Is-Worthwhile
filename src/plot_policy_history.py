from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy training history.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Policy training")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(Path(args.history).read_text())
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    train_acc = history.get("train_answer_accuracy", history.get("train_accuracy"))
    val_acc = history.get("val_answer_accuracy", history.get("val_accuracy"))
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="validation")
    axes[1].set_title("Answer / action accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    if "train_reward" in history:
        axes[2].plot(epochs, history["train_reward"], label="train")
        axes[2].plot(epochs, history["val_reward"], label="validation")
        axes[2].set_title("Reward")
        axes[2].legend()
    else:
        axes[2].axis("off")
    fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
