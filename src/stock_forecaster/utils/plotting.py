"""Save training curves to plots/ directory."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_metric_plots(
    history: dict[str, list[float]],
    plots_dir: Path,
) -> None:
    """Persist loss and classification metric curves."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss" in history and "val_loss" in history:
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training and validation loss")
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png")
        plt.close()

    for metric_name in ("accuracy", "precision", "f1", "recall", "roc_auc"):
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"
        if train_key not in history and val_key not in history:
            continue
        plt.figure(figsize=(8, 5))
        if train_key in history:
            plt.plot(history[train_key], label=train_key)
        if val_key in history:
            plt.plot(history[val_key], label=val_key)
        plt.xlabel("epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(metric_name)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric_name}_curve.png")
        plt.close()
