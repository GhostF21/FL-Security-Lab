"""
BW3 Figures — fixed to match actual filenames in experiments/results/

Generates:
  fig7 — 2-panel: Label-Flip and Gradient Scale curves (FedAvg vs Krum vs TrimMean)
          using f=20% (middle fractions) as the representative case
  fig8 — grouped bar chart: final accuracy across all fractions (10/20/30%)
          for both attack types and all three defenses
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "experiments/results/"
FIGURES = "report/figures/"
os.makedirs(FIGURES, exist_ok=True)

# ── Load CSVs using your actual filenames ────────────────────────────────────
baseline      = pd.read_csv(RESULTS + "baseline_mnist.csv")

# Label-flip at 20% (f=2) — representative for fig7 curves
lf_fedavg_f20 = pd.read_csv(RESULTS + "label_flip_fedavg_f20.csv")
lf_krum_f20   = pd.read_csv(RESULTS + "label_flip_krum_f20.csv")
lf_trim_f20   = pd.read_csv(RESULTS + "label_flip_trimmean_f20.csv")

# Gradient scale at 20% (f=2) — representative for fig7 curves
gs_fedavg_f20 = pd.read_csv(RESULTS + "grad_scale_10_fedavg_f20.csv")
gs_krum_f20   = pd.read_csv(RESULTS + "grad_scale_10_krum_f20.csv")
gs_trim_f20   = pd.read_csv(RESULTS + "grad_scale_10_trimmean_f20.csv")

# All fractions for fig8 bar chart
lf_fedavg_f10  = pd.read_csv(RESULTS + "label_flip_fedavg_f10.csv")
lf_fedavg_f30  = pd.read_csv(RESULTS + "label_flip_fedavg_f30.csv")
lf_krum_f10    = pd.read_csv(RESULTS + "label_flip_krum_f10.csv")
lf_krum_f30    = pd.read_csv(RESULTS + "label_flip_krum_f30.csv")
lf_trim_f10    = pd.read_csv(RESULTS + "label_flip_trimmean_f10.csv")
lf_trim_f30    = pd.read_csv(RESULTS + "label_flip_trimmean_f30.csv")

gs_fedavg_f10  = pd.read_csv(RESULTS + "grad_scale_10_fedavg_f10.csv")
gs_fedavg_f30  = pd.read_csv(RESULTS + "grad_scale_10_fedavg_f30.csv")
gs_krum_f10    = pd.read_csv(RESULTS + "grad_scale_10_krum_f10.csv")
gs_krum_f30    = pd.read_csv(RESULTS + "grad_scale_10_krum_f30.csv")
gs_trim_f10    = pd.read_csv(RESULTS + "grad_scale_10_trimmean_f10.csv")
gs_trim_f30    = pd.read_csv(RESULTS + "grad_scale_10_trimmean_f30.csv")

# ── Helper: extract final accuracy % from a dataframe ────────────────────────
def final_acc(df):
    return df["accuracy"].iloc[-1] * 100

def acc(df):
    return df["accuracy"] * 100

def rnd(df):
    return df["round"]

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "baseline" : "#888888",
    "fedavg"   : "#ff6b6b",
    "krum"     : "#4f9cf9",
    "trim"     : "#4fe8a0",
}

plt.style.use("dark_background")

# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — Accuracy curves: FedAvg vs Krum vs TrimMean (20% malicious)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ── Left panel: Label-Flip ────────────────────────────────────────────────
ax = axes[0]
ax.plot(rnd(baseline),      acc(baseline),      color=C["baseline"], lw=1.5, ls=":",  label="Clean FedAvg (baseline)")
ax.plot(rnd(lf_fedavg_f20), acc(lf_fedavg_f20), color=C["fedavg"],   lw=2,   ls="--", label="FedAvg + Label-Flip 20%")
ax.plot(rnd(lf_krum_f20),   acc(lf_krum_f20),   color=C["krum"],     lw=2,   ls="-",  label="Multi-Krum (f=2)")
ax.plot(rnd(lf_trim_f20),   acc(lf_trim_f20),   color=C["trim"],     lw=2,   ls="-",  label="Trimmed Mean (β=2)")
ax.set_title("Label-Flip Attack — 20% Malicious", fontsize=11)
ax.set_xlabel("Round")
ax.set_ylabel("Test Accuracy (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)
ax.set_ylim(0, 105)

# ── Right panel: Gradient Scaling ────────────────────────────────────────
ax = axes[1]
ax.plot(rnd(baseline),      acc(baseline),      color=C["baseline"], lw=1.5, ls=":",  label="Clean FedAvg (baseline)")
ax.plot(rnd(gs_fedavg_f20), acc(gs_fedavg_f20), color=C["fedavg"],   lw=2,   ls="--", label="FedAvg + GradScale λ=10, 20%")
ax.plot(rnd(gs_krum_f20),   acc(gs_krum_f20),   color=C["krum"],     lw=2,   ls="-",  label="Multi-Krum (f=2)")
ax.plot(rnd(gs_trim_f20),   acc(gs_trim_f20),   color=C["trim"],     lw=2,   ls="-",  label="Trimmed Mean (β=2)")
ax.set_title("Gradient Scale Attack (λ=10) — 20% Malicious", fontsize=11)
ax.set_xlabel("Round")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)
ax.set_ylim(0, 105)

fig.suptitle(
    "Fig 7 — Defense Comparison: FedAvg vs Krum vs Trimmed Mean (MNIST non-IID, 20% malicious)",
    fontsize=12, y=1.02
)
plt.tight_layout()
fig7_path = FIGURES + "fig7_defense_curves.png"
plt.savefig(fig7_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {fig7_path} saved")


# ════════════════════════════════════════════════════════════════════════════
# Fig 8 — Bar chart: final accuracy across all fractions and defenses
# ════════════════════════════════════════════════════════════════════════════
#
# Layout: 6 groups (LF-10%, LF-20%, LF-30%, GS-10%, GS-20%, GS-30%)
#         3 bars per group (FedAvg, Krum, TrimMean)
#
baseline_acc = final_acc(baseline)

groups = [
    ("LF 10%",  lf_fedavg_f10,  lf_krum_f10,  lf_trim_f10),
    ("LF 20%",  lf_fedavg_f20,  lf_krum_f20,  lf_trim_f20),
    ("LF 30%",  lf_fedavg_f30,  lf_krum_f30,  lf_trim_f30),
    ("GS 10%",  gs_fedavg_f10,  gs_krum_f10,  gs_trim_f10),
    ("GS 20%",  gs_fedavg_f20,  gs_krum_f20,  gs_trim_f20),
    ("GS 30%",  gs_fedavg_f30,  gs_krum_f30,  gs_trim_f30),
]

labels      = [g[0] for g in groups]
fedavg_vals = [final_acc(g[1]) for g in groups]
krum_vals   = [final_acc(g[2]) for g in groups]
trim_vals   = [final_acc(g[3]) for g in groups]

x     = np.arange(len(labels))
width = 0.25

fig2, ax2 = plt.subplots(figsize=(13, 5))

bars_f = ax2.bar(x - width, fedavg_vals, width, label="FedAvg (no defense)", color=C["fedavg"],   alpha=0.85)
bars_k = ax2.bar(x,         krum_vals,   width, label="Multi-Krum",          color=C["krum"],     alpha=0.85)
bars_t = ax2.bar(x + width, trim_vals,   width, label="Trimmed Mean",        color=C["trim"],     alpha=0.85)

# Baseline reference line
ax2.axhline(baseline_acc, color=C["baseline"], lw=1.5, ls=":", label=f"Clean baseline ({baseline_acc:.1f}%)")

# Value labels on bars
for bars in [bars_f, bars_k, bars_t]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.4,
            f"{h:.1f}",
            ha="center", va="bottom", fontsize=7.5, color="white"
        )

# Vertical separator between LF and GS groups
ax2.axvline(2.5, color="#555", lw=1, ls="--")
ax2.text(0.95, 1.01, "← Label-Flip", transform=ax2.transAxes,
         ha="right", fontsize=9, color="#aaa")
ax2.text(0.97, 1.01, "Grad Scale →", transform=ax2.transAxes,
         ha="left", fontsize=9, color="#aaa")

ax2.set_xlabel("Attack Type & Malicious Fraction")
ax2.set_ylabel("Final Test Accuracy (%)")
ax2.set_title("Fig 8 — Defense Comparison Matrix: Final Accuracy per Attack & Fraction (MNIST non-IID)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 110)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.12, axis="y")

plt.tight_layout()
fig8_path = FIGURES + "fig8_defense_matrix_bar.png"
plt.savefig(fig8_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {fig8_path} saved")

# ════════════════════════════════════════════════════════════════════════════
# Print summary table for quick sanity check
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  FINAL ACCURACY SUMMARY")
print(f"{'='*70}")
print(f"  {'Group':<10} │ {'FedAvg':>8} │ {'Krum':>8} │ {'TrimMean':>9} │ {'Best defense'}")
print(f"  {'─'*70}")
for label, f_df, k_df, t_df in groups:
    fa = final_acc(f_df)
    ka = final_acc(k_df)
    ta = final_acc(t_df)
    best = "Krum" if ka >= ta else "TrimMean"
    print(f"  {label:<10} │ {fa:>7.2f}% │ {ka:>7.2f}% │ {ta:>8.2f}% │ {best}")

print(f"\n  Clean baseline: {baseline_acc:.2f}%")
print(f"  Figures saved → {FIGURES}")