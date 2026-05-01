"""
BW4 Detection Evaluation Script.

Evaluates all 3 anomaly detectors across both attack types and
malicious fractions (10%, 20%, 30%).

Outputs:
  experiments/results/detection/detection_summary_bw4.csv  — F1 table
  report/figures/fig13_detection_heatmap.png               — F1 heatmap
  report/figures/fig14_roc_curves.png                      — ROC curves
  report/figures/fig15_pca_scatter.png                     — PCA 2D view
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.detection.norm_threshold import evaluate_norm_threshold
from src.detection.cosine_cluster import evaluate_cosine
from src.detection.pca_outlier    import evaluate_pca_outlier

OUT_DIR = "experiments/results/detection/"
FIGURES = "report/figures/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
plt.style.use("dark_background")

NUM_ROUNDS  = 20
NUM_CLIENTS = 10
PARAM_DIM   = 500   # flattened gradient dimension (use 500 for speed)

# ── Scenario definitions ─────────────────────────────────────────────────────
SCENARIOS = [
    # (scenario_name, attack_type, mal_frac, honest_norm_mu, mal_norm_mu)
    ("GS_f10", "grad_scale", 0.1, 0.52, 5.20),
    ("GS_f20", "grad_scale", 0.2, 0.52, 5.20),
    ("GS_f30", "grad_scale", 0.3, 0.52, 5.20),
    ("LF_f10", "label_flip", 0.1, 0.52, 0.58),   # LF norms close to honest
    ("LF_f20", "label_flip", 0.2, 0.52, 0.58),
    ("LF_f30", "label_flip", 0.3, 0.52, 0.60),
]

all_results = []

for scenario, attack_type, mal_frac, h_mu, m_mu in SCENARIOS:
    rng          = np.random.default_rng(42)
    n_mal        = int(NUM_CLIENTS * mal_frac)
    n_hon        = NUM_CLIENTS - n_mal

    print(f"\n{'─'*50}")
    print(f"  Scenario: {scenario}  |  attack={attack_type}  |  mal={mal_frac:.0%}")

    rows             = []
    updates_by_round = {}

    for r in range(1, NUM_ROUNDS + 1):
        # Honest gradient vectors
        h_vecs = [rng.normal(0, h_mu, PARAM_DIM) for _ in range(n_hon)]
        # Malicious gradient vectors
        if attack_type == "grad_scale":
            # Same direction, much larger magnitude
            m_vecs = [rng.normal(0, m_mu, PARAM_DIM) for _ in range(n_mal)]
        else:
            # Label-flip: similar magnitude, slightly different direction
            m_vecs = [rng.normal(0.05, m_mu, PARAM_DIM) for _ in range(n_mal)]

        updates_by_round[r] = h_vecs + m_vecs

        for c, v in enumerate(h_vecs):
            rows.append({"round":r,"client_id":c,"l2_norm":np.linalg.norm(v),"is_malicious":False})
        for c, v in enumerate(m_vecs):
            rows.append({"round":r,"client_id":n_hon+c,"l2_norm":np.linalg.norm(v),"is_malicious":True})

    norm_df = pd.DataFrame(rows)
    norm_df.to_csv(f"{OUT_DIR}{scenario}_norms.csv", index=False)

    # ── Run all 3 detectors ───────────────────────────────────────────────────
    r1 = evaluate_norm_threshold(norm_df, k=2.0)
    r2 = evaluate_cosine(norm_df, updates_by_round, threshold=0.5)
    r3 = evaluate_pca_outlier(norm_df, updates_by_round, n_components=2, percentile_threshold=90.0)

    for res in [r1, r2, r3]:
        all_results.append({
            "scenario"   : scenario,
            "attack"     : attack_type,
            "mal_frac"   : mal_frac,
            "detector"   : res["detector"],
            "precision"  : res["precision"],
            "recall"     : res["recall"],
            "f1"         : res["f1"],
            "fpr"        : res["fpr"],
        })
        print(f"  {res['detector']:<35} F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}  FPR={res['fpr']:.4f}")

# ── Save summary CSV ──────────────────────────────────────────────────────────
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(OUT_DIR + "detection_summary_bw4.csv", index=False)
print(f"\n✓ detection_summary_bw4.csv saved → {OUT_DIR}")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 13 — F1 Score Heatmap
# ═══════════════════════════════════════════════════════════════════════════
detectors = ["NormThreshold(k=2.0)", "CosineSimilarity(thr=0.5)", "PCAOutlier(k=2, pct=90.0)"]
scenarios_ordered = [s[0] for s in SCENARIOS]

heatmap_data = np.zeros((len(detectors), len(scenarios_ordered)))
for i, det in enumerate(detectors):
    for j, scen in enumerate(scenarios_ordered):
        row = summary_df[(summary_df["detector"]==det) & (summary_df["scenario"]==scen)]
        if len(row) > 0:
            heatmap_data[i, j] = row["f1"].values[0]

fig1, ax1 = plt.subplots(figsize=(11, 4))
im = ax1.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax1, label="F1 Score")

short_det = ["Norm\nThreshold", "Cosine\nSimilarity", "PCA\nOutlier"]
ax1.set_yticks(range(len(detectors))); ax1.set_yticklabels(short_det, fontsize=10)
ax1.set_xticks(range(len(scenarios_ordered))); ax1.set_xticklabels(scenarios_ordered, fontsize=9)
ax1.set_xlabel("Scenario (attack_fraction)"); ax1.set_title("Fig 13 — Detection F1 Score Heatmap (BW4)", fontsize=12)

for i in range(len(detectors)):
    for j in range(len(scenarios_ordered)):
        val = heatmap_data[i, j]
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="black" if val > 0.5 else "white", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURES + "fig13_detection_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print("✓ fig13_detection_heatmap.png saved")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 14 — ROC Curves (GS_f20 scenario, all 3 detectors)
# ═══════════════════════════════════════════════════════════════════════════
# Re-generate GS_f20 data for ROC (need continuous scores not just flags)
rng = np.random.default_rng(42)
n_mal, n_hon = 2, 8
rows2, ubr2 = [], {}
for r in range(1, NUM_ROUNDS+1):
    hvecs = [rng.normal(0, 0.52, PARAM_DIM) for _ in range(n_hon)]
    mvecs = [rng.normal(0, 5.20, PARAM_DIM) for _ in range(n_mal)]
    ubr2[r] = hvecs + mvecs
    for c,v in enumerate(hvecs): rows2.append({"round":r,"client_id":c,"l2_norm":np.linalg.norm(v),"is_malicious":False})
    for c,v in enumerate(mvecs): rows2.append({"round":r,"client_id":n_hon+c,"l2_norm":np.linalg.norm(v),"is_malicious":True})

df2    = pd.DataFrame(rows2)
y_true = df2["is_malicious"].astype(int).values
y_norm = df2["l2_norm"].values   # score for NormThreshold

# Cosine scores: use (1 - cosine_sim) as the anomaly score
from sklearn.preprocessing import normalize
cos_scores = []
for r, grp in df2.groupby("round"):
    vecs = ubr2[r]
    mat  = np.stack(vecs, axis=0)
    mn   = normalize(mat, norm="l2", axis=1)
    mu   = mn.mean(axis=0, keepdims=True)
    mu   = normalize(mu, norm="l2", axis=1)
    sims = (mn @ mu.T).flatten()
    cos_scores.extend((1 - sims).tolist())   # higher = more anomalous

# PCA Mahalanobis distances as scores
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
pca_scores = []
for r, grp in df2.groupby("round"):
    vecs = ubr2[r]
    mat  = StandardScaler().fit_transform(np.stack(vecs, axis=0))
    proj = PCA(n_components=2, random_state=42).fit_transform(mat)
    mu   = proj.mean(axis=0)
    cov  = np.cov(proj.T)
    try:
        ci   = np.linalg.inv(cov)
        dsts = [mahalanobis(p, mu, ci) for p in proj]
    except:
        dsts = [np.linalg.norm(p - mu) for p in proj]
    pca_scores.extend(dsts)

fig2, ax2 = plt.subplots(figsize=(8, 6))
for scores, label, color in [
    (y_norm,                 "Norm Threshold",      "#ff6b6b"),
    (np.array(cos_scores),   "Cosine Similarity",   "#5ab4e8"),
    (np.array(pca_scores),   "PCA Outlier",         "#5ae8a0"),
]:
    fpr_r, tpr_r, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr_r, tpr_r)
    ax2.plot(fpr_r, tpr_r, color=color, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

ax2.plot([0,1],[0,1], color="#555", lw=1, ls="--", label="Random (AUC=0.5)")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("Fig 14 — ROC Curves: All Detectors vs Grad Scale (20% malicious)")
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.15); ax2.set_xlim(0,1); ax2.set_ylim(0,1.02)
plt.tight_layout()
plt.savefig(FIGURES + "fig14_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(); print("✓ fig14_roc_curves.png saved")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 15 — PCA 2D scatter (GS_f20, round 1)
# ═══════════════════════════════════════════════════════════════════════════
r1_updates = ubr2[1]
mat_s      = StandardScaler().fit_transform(np.stack(r1_updates, axis=0))
proj_2d    = PCA(n_components=2, random_state=42).fit_transform(mat_s)
labels_r1  = [False]*n_hon + [True]*n_mal

fig3, ax3 = plt.subplots(figsize=(7, 6))
for is_mal, color, marker, lab in [
    (False, "#5ae8a0", "o", "Honest clients"),
    (True,  "#ff6b6b", "X", "Malicious clients"),
]:
    idxs = [i for i,m in enumerate(labels_r1) if m==is_mal]
    ax3.scatter(proj_2d[idxs,0], proj_2d[idxs,1], c=color, marker=marker,
                s=120, label=lab, alpha=0.9, edgecolors="white", linewidths=0.5)

ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")
ax3.set_title("Fig 15 — PCA 2D Projection: Honest vs Malicious Clients\nGrad Scale λ=10, 20% malicious (Round 1)")
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(FIGURES + "fig15_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close(); print("✓ fig15_pca_scatter.png saved")

# ── Print full summary table ───────────────────────────────────────────────
print(f"\n{'='*70}")
print("  BW4 DETECTION SUMMARY TABLE")
print(f"{'='*70}")
print(f"  {'Scenario':<10} {'Detector':<30} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}")
print(f"  {'─'*70}")
for _, row in summary_df.iterrows():
    print(f"  {row['scenario']:<10} {row['detector']:<30} {row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f} {row['fpr']:>7.4f}")