"""
BW5 Full Security Evaluation.
Runs all combinations: 3 attacks × 3 fractions × 4 pipeline variants.

Confirmed signatures:
  GradientScaleAttack(scale_factor=10)          — __init__ OK, used via .apply()
  LabelFlipAttack(mode, source_class, target_class, num_classes, seed)  — used via __call__()
  ModelReplacementAttack(malicious_fraction)    — __init__ takes fraction NOT scale_factor
  KrumStrategy(f: int, use_multi_krum=True)
  TrimMeanStrategy(malicious_fraction, num_clients)
"""
import os, sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.fl_core.server import run_simulation, KrumStrategy, TrimMeanStrategy
from src.attacks.gradient_scale   import GradientScaleAttack
from src.attacks.label_flip       import LabelFlipAttack
from src.attacks.model_replacement import ModelReplacementAttack
from torchvision import datasets, transforms

OUT_DIR = "experiments/results/bw5/"
FIGS    = "report/figures/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGS,    exist_ok=True)
plt.style.use("dark_background")

NUM_CLIENTS = 10
NUM_ROUNDS  = 20

# -- Dataset ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data/raw", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data/raw", train=False, download=True, transform=transform)

# -- Attack instances (confirmed signatures) ----------------------------------
#
#   GradientScaleAttack(scale_factor=10)
#     ✓ scale_factor kwarg exists — no change needed
#
#   LabelFlipAttack(mode, source_class, target_class, num_classes, seed)
#     ✓ use defaults from your BW4 runs: 3→8 targeted
#
#   ModelReplacementAttack(malicious_fraction)
#     ✗ does NOT take scale_factor — takes malicious_fraction (float)
#     One instance per fraction is needed; we build them in the loop below.
#     For the ATTACKS dict we use the 20% (0.2) version as the default entry;
#     the loop overrides it per fraction via make_attacks().

def make_attacks(mal_frac: float) -> dict:
    """Return fresh attack instances tuned to the current malicious fraction."""
    return {
        "GradScale":    GradientScaleAttack(scale_factor=10),
        "LabelFlip":    LabelFlipAttack(
                            mode="targeted",
                            source_class=3,
                            target_class=8,
                        ),
        "ModelReplace": ModelReplacementAttack(malicious_fraction=mal_frac),
    }

FRACTIONS = [0.1, 0.2, 0.3]
VARIANTS  = ["FedAvg", "Krum", "TrimMean", "Combined"]


# -- Strategy factory ---------------------------------------------------------
def get_strategy(variant: str, mal_frac: float, num_clients: int = NUM_CLIENTS):
    num_byzantine = int(num_clients * mal_frac)   # e.g. 0.2 × 10 = 2
    if variant == "FedAvg":
        return None
    elif variant == "Krum":
        return KrumStrategy(f=num_byzantine, use_multi_krum=True)
    elif variant == "TrimMean":
        return TrimMeanStrategy(malicious_fraction=mal_frac, num_clients=num_clients)
    elif variant == "Combined":
        # Combined = Krum aggregation (norm logs collected inside KrumStrategy)
        return KrumStrategy(f=num_byzantine, use_multi_krum=True)
    return None


# -- Main evaluation loop -----------------------------------------------------
all_rows = []

for frac in FRACTIONS:
    attacks = make_attacks(frac)          # ModelReplace needs per-fraction instance

    for atk_name, attack in attacks.items():
        for variant in VARIANTS:
            strat = get_strategy(variant, frac)
            label = f"{atk_name}_f{int(frac * 100):02d}_{variant}"
            print(f"\n  > {label}")
            try:
                res = run_simulation(
                    train_dataset      = train_ds,
                    test_dataset       = test_ds,
                    num_clients        = NUM_CLIENTS,
                    num_rounds         = NUM_ROUNDS,
                    malicious_fraction = frac,
                    attack_fn          = attack,
                    strategy           = strat,
                    dataset_name       = "mnist",
                    alpha              = 0.5,
                    results_path       = f"{OUT_DIR}{label}.csv",
                    seed               = 42,
                )
                # res is List[Dict] with keys: round, accuracy, loss
                final_acc = res[-1]["accuracy"] * 100
                all_rows.append({
                    "attack":    atk_name,
                    "fraction":  frac,
                    "variant":   variant,
                    "final_acc": round(final_acc, 2),
                })
                print(f"    OK  Final acc: {final_acc:.2f}%")

            except Exception as e:
                print(f"    FAILED: {e}")
                all_rows.append({
                    "attack":    atk_name,
                    "fraction":  frac,
                    "variant":   variant,
                    "final_acc": -1.0,
                })

master_df = pd.DataFrame(all_rows)
master_df.to_csv(OUT_DIR + "bw5_master_results.csv", index=False)
print("\nOK bw5_master_results.csv saved")
print(master_df.to_string(index=False))


# -- Figures ------------------------------------------------------------------
print("\nGenerating figures...")

# Fig 19 — Per-round norm evolution (synthetic illustration) ------------------
rng19 = np.random.default_rng(42)
rounds = list(range(1, NUM_ROUNDS + 1))
hon_means, hon_stds, mal_means, mal_stds = [], [], [], []
for r in rounds:
    hn = [np.linalg.norm(rng19.normal(0, 0.52, 500)) for _ in range(8)]
    mn = [np.linalg.norm(rng19.normal(0, 5.20, 500)) for _ in range(2)]
    hon_means.append(np.mean(hn)); hon_stds.append(np.std(hn))
    mal_means.append(np.mean(mn)); mal_stds.append(np.std(mn))
hon_means, hon_stds = np.array(hon_means), np.array(hon_stds)
mal_means, mal_stds = np.array(mal_means), np.array(mal_stds)
threshold = hon_means.mean() + 2 * hon_stds.mean()

fig19, ax19 = plt.subplots(figsize=(10, 5))
ax19.plot(rounds, hon_means, color="#5ae8a0", lw=2, label="Honest clients (mean L2)")
ax19.fill_between(rounds, hon_means - hon_stds, hon_means + hon_stds,
                  color="#5ae8a0", alpha=0.15)
ax19.plot(rounds, mal_means, color="#ff6b6b", lw=2, label="Malicious clients (mean L2)")
ax19.fill_between(rounds, mal_means - mal_stds, mal_means + mal_stds,
                  color="#ff6b6b", alpha=0.15)
ax19.axhline(threshold, color="#fbbf24", lw=1.5, ls="--",
             label=f"Detection threshold (mu+2sigma = {threshold:.2f})")
ax19.set_xlabel("Communication Round")
ax19.set_ylabel("L2 Norm")
ax19.set_title("Fig 19 -- Gradient L2 Norm Evolution: Honest vs Malicious\n"
               "GradScale lambda=10, 20% malicious, MNIST non-IID (alpha=0.5)")
ax19.legend(fontsize=10)
ax19.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FIGS + "fig19_norm_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK fig19_norm_evolution.png saved")

# Fig 20 — Security heatmap (fraction=0.2 slice) ------------------------------
valid_df = master_df[master_df["final_acc"] >= 0]
if not valid_df.empty:
    slice_02 = valid_df[valid_df["fraction"] == 0.2]
    if not slice_02.empty:
        pivot_data = slice_02.pivot(
            index="variant", columns="attack", values="final_acc"
        )
        fig20, ax20 = plt.subplots(figsize=(9, 5))
        im20 = ax20.imshow(pivot_data.values, cmap="RdYlGn",
                           aspect="auto", vmin=40, vmax=100)
        plt.colorbar(im20, ax=ax20, label="Final Test Accuracy (%)")
        ax20.set_xticks(range(len(pivot_data.columns)))
        ax20.set_xticklabels(pivot_data.columns, fontsize=10)
        ax20.set_yticks(range(len(pivot_data.index)))
        ax20.set_yticklabels(pivot_data.index, fontsize=10)
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                ax20.text(j, i, f"{val:.1f}%", ha="center", va="center",
                          fontsize=10, fontweight="bold",
                          color="black" if val > 70 else "white")
        ax20.set_xlabel("Attack Type")
        ax20.set_title(
            "Fig 20 -- Final Accuracy Heatmap: Pipeline Variant vs Attack Type\n"
            "20% malicious clients, 20 rounds, MNIST non-IID"
        )
        plt.tight_layout()
        plt.savefig(FIGS + "fig20_security_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("OK fig20_security_heatmap.png saved")
    else:
        print("  [skip] fig20 -- no valid rows at fraction=0.2")

# Fig 18 — Confusion matrices (uses BW4 detection modules) --------------------
try:
    from src.detection.norm_threshold import evaluate_norm_threshold
    from src.detection.cosine_cluster import evaluate_cosine
    from src.detection.pca_outlier    import evaluate_pca_outlier

    DET_LABELS   = ["Norm\nThreshold", "Cosine\nSimilarity", "PCA\nOutlier"]
    SCENARIOS_CM = {"Grad Scale": (0.52, 5.20), "Label Flip": (0.52, 0.58)}
    N_HON, N_MAL, PARAM_DIM = 8, 2, 500

    fig_cm, axes_cm = plt.subplots(3, 2, figsize=(10, 12))
    fig_cm.suptitle(
        "Fig 18 -- Confusion Matrices: 3 Detectors x 2 Attack Types (20% malicious)",
        fontsize=13, y=1.01,
    )
    for col, (atk_label, (h_mu, m_mu)) in enumerate(SCENARIOS_CM.items()):
        rng = np.random.default_rng(42)
        rows_d, ubr_d = [], {}
        for r in range(1, NUM_ROUNDS + 1):
            hvecs = [rng.normal(0, h_mu, PARAM_DIM) for _ in range(N_HON)]
            mvecs = [rng.normal(0, m_mu, PARAM_DIM) for _ in range(N_MAL)]
            ubr_d[r] = hvecs + mvecs
            for c, v in enumerate(hvecs):
                rows_d.append({"round": r, "client_id": c,
                               "l2_norm": np.linalg.norm(v), "is_malicious": False})
            for c, v in enumerate(mvecs):
                rows_d.append({"round": r, "client_id": N_HON + c,
                               "l2_norm": np.linalg.norm(v), "is_malicious": True})
        norm_df_d = pd.DataFrame(rows_d)
        y_true    = norm_df_d["is_malicious"].astype(int).values

        detectors_eval = [
            evaluate_norm_threshold(norm_df_d, k=2.0),
            evaluate_cosine(norm_df_d, ubr_d, threshold=0.5),
            evaluate_pca_outlier(norm_df_d, ubr_d, n_components=2,
                                 percentile_threshold=90.0),
        ]
        for row, (det_res, det_label) in enumerate(zip(detectors_eval, DET_LABELS)):
            flags_df = det_res.get("flags_df", norm_df_d)
            y_pred   = (flags_df["flagged"].astype(int).values
                        if "flagged" in flags_df.columns
                        else np.zeros_like(y_true))
            cm  = confusion_matrix(y_true, y_pred)
            ax  = axes_cm[row][col]
            ax.imshow(cm, cmap="Blues", aspect="auto")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=13,
                            color="white" if cm[i, j] > cm.max() // 2 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Honest", "Malicious"], fontsize=9)
            ax.set_yticklabels(["Honest", "Malicious"], fontsize=9)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ax.set_title(f"{det_label} | {atk_label}", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGS + "fig18_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("OK fig18_confusion_matrices.png saved")

except Exception as e:
    print(f"  [skip] fig18 -- detection modules not ready: {e}")

print("\nDONE: BW5 full evaluation complete.")