import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path


def load_loss_from_checkpoints(checkpoints_dir):
    """Load training loss from all checkpoint trainer_state.json files."""
    checkpoints_dir = Path(checkpoints_dir)
    
    if not checkpoints_dir.exists():
        print(f"  Warning: {checkpoints_dir} does not exist, skipping.")
        return []

    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )

    if not checkpoint_dirs:
        print(f"  Warning: No checkpoint dirs found in {checkpoints_dir}")
        return []

    all_logs = {}

    for ckpt_dir in checkpoint_dirs:
        state_file = ckpt_dir / "trainer_state.json"
        if not state_file.exists():
            print(f"  Warning: No trainer_state.json in {ckpt_dir.name}, skipping.")
            continue

        with open(state_file) as f:
            state = json.load(f)

        for entry in state.get("log_history", []):
            if "loss" in entry and "step" in entry:
                step = entry["step"]
                if step not in all_logs:
                    all_logs[step] = entry["loss"]

    # Sort by step
    steps = sorted(all_logs.keys())
    losses = [all_logs[s] for s in steps]
    return list(zip(steps, losses))


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from LoRA checkpoints")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to run directories containing a 'checkpoints/' subfolder. E.g. test0 test2 test3"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0,
        help="EMA smoothing factor (0 = no smoothing, 0.99 = very smooth). Default: 0"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loss_plot.png",
        help="Output plot filename. Default: loss_plot.png"
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10(np.linspace(0, 1, len(args.runs)))

    for run_path, color in zip(args.runs, colors):
        run_path = Path(run_path)
        label = run_path.name
        checkpoints_dir = run_path / "checkpoints"

        print(f"Loading: {checkpoints_dir}")
        data = load_loss_from_checkpoints(checkpoints_dir)

        if not data:
            print(f"  No data found for {label}, skipping.")
            continue

        steps, losses = zip(*data)

        # Raw loss (faint)
        ax.plot(steps, losses, color=color, alpha=0.2, linewidth=0.8)

        # Smoothed loss
        smoothed = smooth(list(losses), weight=args.smoothing)
        ax.plot(steps, smoothed, color=color, linewidth=2, label=label)

        print(f"  {label}: {len(steps)} steps, final loss = {losses[-1]:.4f}")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss by Run", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output = Path(args.output)
    fig.savefig(output, dpi=150)
    print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    main()