"""
Simple training curve visualizer.

Usage:
    python apps/plot_training.py logs/train.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import oshttps://www.youtube.com/watch?v=XjHwy07aHsU


def main(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] No data in {csv_path}")
        return

    # Columns logged by Callbacks: t, episode, loss, policy_loss, value_loss, entropy
    plt.figure(figsize=(10, 6))
    plt.suptitle("Training Metrics")

    # Loss curves
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df["episode"], df["loss"], label="total loss")
    ax1.plot(df["episode"], df["policy_loss"], label="policy loss", alpha=0.7)
    ax1.plot(df["episode"], df["value_loss"], label="value loss", alpha=0.7)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Entropy
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df["episode"], df["entropy"], color="tab:orange", label="entropy")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Entropy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Determine output path
    out_dir = os.path.dirname(csv_path)
    out_name = os.path.splitext(os.path.basename(csv_path))[0] + "_plot.png"
    out_path = os.path.join(out_dir, out_name)

    # Save plot
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved training plot to {out_path}")

    # Optionally display
    try:
        plt.show()
    except Exception:
        print("[INFO] Non-interactive backend — skipping display.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="logs/train.csv", help="Path to training CSV log")
    args = ap.parse_args()
    main(args.csv)
