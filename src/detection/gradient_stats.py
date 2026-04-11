"""
Gradient Statistics Analysis — BW3 starting point.

Collects per-client gradient L2 norms across FL rounds.
Splits honest vs malicious clients to show norm distributions.
This data feeds into anomaly detection design in BW4.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict


def compute_gradient_norm(
    global_params: List[np.ndarray],
    local_params: List[np.ndarray],
) -> float:
    """
    Compute L2 norm of gradient update (local - global) for a single client.
    
    Args:
        global_params : model weights before the round (list of np.ndarrays)
        local_params  : model weights after local training (list of np.ndarrays)
    Returns:
        l2_norm : scalar float — the L2 norm of the gradient delta
    """
    deltas = [l - g for l, g in zip(local_params, global_params)]
    flat   = np.concatenate([d.flatten() for d in deltas])
    return float(np.linalg.norm(flat))


def collect_norm_stats(
    norm_logs: List[Dict],
    out_csv: str = "experiments/results/gradient_norms.csv",
) -> pd.DataFrame:
    """
    Save collected norm logs to CSV.
    
    norm_logs: list of dicts, each: 
        {"round": int, "client_id": int, "is_malicious": bool, "l2_norm": float}
    """
    df = pd.DataFrame(norm_logs)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✓ Gradient norm stats saved → {out_csv}")
    return df


def plot_norm_distributions(
    df: pd.DataFrame,
    attack_name: str = "Gradient Scaling λ=10",
    out_path: str = "report/figures/fig9_grad_norm_dist.png",
):
    """
    Plot L2 norm distributions: honest vs malicious clients across all rounds.
    Creates two subplots:
      Left  — Box plots per round (shows temporal pattern)
      Right — Distribution histogram (honest vs malicious overlap)
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    honest_df    = df[df["is_malicious"] == False]
    malicious_df = df[df["is_malicious"] == True]

    # ── Left: Per-round mean norms ──────────────────────────────────────────
    ax = axes[0]
    h_means = honest_df.groupby("round")["l2_norm"].mean()
    m_means = malicious_df.groupby("round")["l2_norm"].mean()

    ax.plot(h_means.index, h_means.values, color="#4fe8a0", lw=2, label="Honest clients (mean)")
    ax.fill_between(
        h_means.index,
        honest_df.groupby("round")["l2_norm"].min(),
        honest_df.groupby("round")["l2_norm"].max(),
        alpha=0.2, color="#4fe8a0", label="Honest (range)"
    )
    if len(malicious_df) > 0:
        ax.plot(m_means.index, m_means.values, color="#ff6b6b", lw=2, ls="--", label="Malicious clients (mean)")

    ax.set_xlabel("Round"); ax.set_ylabel("L2 Norm of Gradient Update")
    ax.set_title(f"Gradient Norms per Round\n({attack_name})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # ── Right: Distribution histogram ───────────────────────────────────────
    ax = axes[1]
    ax.hist(honest_df["l2_norm"],    bins=40, alpha=0.7, color="#4fe8a0", label="Honest", density=True)
    if len(malicious_df) > 0:
        ax.hist(malicious_df["l2_norm"], bins=40, alpha=0.7, color="#ff6b6b", label="Malicious", density=True)

    # Add µ ± 2σ threshold line for honest clients
    mu    = honest_df["l2_norm"].mean()
    sigma = honest_df["l2_norm"].std()
    ax.axvline(mu + 2 * sigma, color="#ffd166", ls="--", lw=1.5, label=f"µ+2σ = {mu+2*sigma:.2f}")

    ax.set_xlabel("L2 Norm"); ax.set_ylabel("Density")
    ax.set_title(f"Norm Distribution: Honest vs Malicious\n({attack_name})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    fig.suptitle("Fig 9 — Gradient L2 Norm Statistics (BW3 Analysis)", fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ {out_path} saved")


if __name__ == "__main__":
    # ── Synthetic demo: simulate norms to verify plotting works ──────────────
    # In real use, this data comes from your FL simulation loop
    rng = np.random.default_rng(42)
    logs = []
    for r in range(1, 21):
        for c in range(8):   # 8 honest
            logs.append({"round": r, "client_id": c, "is_malicious": False,
                          "l2_norm": rng.normal(0.5, 0.08)})
        for c in range(2):   # 2 malicious (gradient scale λ=10 → ~10× norms)
            logs.append({"round": r, "client_id": 8+c, "is_malicious": True,
                          "l2_norm": rng.normal(5.0, 0.3)})

    df = collect_norm_stats(logs)
    plot_norm_distributions(df, attack_name="Gradient Scaling λ=10 (20% malicious)")
    print("✓ Gradient stats demo complete")