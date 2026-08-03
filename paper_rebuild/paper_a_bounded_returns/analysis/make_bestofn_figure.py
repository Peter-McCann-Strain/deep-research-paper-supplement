#!/usr/bin/env python
"""Best-of-k selection figure (the winner's-curse control).

Reads canonical_numbers.json['best_of_n']. Plots the oracle (judge-max) best-of-k curve
against the mean-of-k flat line and the orchestrated-cluster mean, and overlays the
pure-noise prediction (max-order-statistic of N(flat_mean, sigma_within)). The observed
best-of-k tracks the pure-noise curve almost exactly: most of the apparent best-of-k
'gain' over a single P0 sample is judge-noise capitalisation, not real quality.
Even the judge-max upper bound only reaches the cluster mean around k~7, at cluster-level cost.

CPU-only; no API.
"""
import json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = "."
B = json.load(open(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/analysis/canonical_numbers.json"))["best_of_n"]

ks = sorted(int(k) for k in B["curve"].keys())
best = [B["curve"][str(k)]["best_of_k"] for k in ks]
meank = [B["curve"][str(k)]["mean_of_k"] for k in ks]
noise = [B["pure_noise"]["prediction"][str(k)]["predicted_best_of_k"] for k in ks]
cluster = B["cluster_mean"]
# Decoupled (winner's-curse-corrected) split-half curve: selection on half A, scoring on half B.
dec = B["decoupled"]
dks = sorted(int(k) for k in dec["curve"].keys())
dbest = [dec["curve"][str(k)]["best_of_k_decoupled"] for k in dks]
cluster_B = dec["cluster_mean_half_B"]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.axhline(cluster, color="#1f4e79", ls="--", lw=1.4, zorder=2,
           label=f"orchestrated cluster mean ({cluster:.3f})")
ax.plot(ks, noise, "-", color="#c0504d", lw=1.6, alpha=0.85, zorder=3,
        label="pure-noise prediction (max order stat.)")
ax.plot(ks, best, "o-", color="#1f4e79", lw=1.8, ms=6, zorder=4,
        label="observed best-of-$k$ (judge-max)")
ax.plot(ks, meank, "s-", color="#7f9bb5", lw=1.4, ms=4, zorder=3,
        label="mean-of-$k$ (single sample)")
ax.plot(dks, dbest, "^-", color="#2e7d32", lw=1.8, ms=5, zorder=4,
        label="decoupled best-of-$k$ (winner's-curse-corrected)")
# Mark where the DECOUPLED curve first reaches the held-out cluster line (the honest reach).
# The coupled curve's earlier crossing is winner's-curse inflation, so the annotation uses the
# decoupled split-half curve against cluster_mean_half_B -> k~7 (matches prose/abstract).
reach = next((k for k, v in zip(dks, dbest) if v >= cluster_B), None)
if reach is not None:
    ax.axvline(reach, color="#888", ls=":", lw=1.0, zorder=1)
    ax.annotate(f"decoupled reaches\ncluster at $k\\approx{reach}$", xy=(reach, cluster_B),
                xytext=(reach + 0.4, cluster_B - 0.03), fontsize=8.4, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
ax.set_xlabel("$k$ (samples drawn from a single P0 architecture)")
ax.set_ylabel("Overall score (GPT-5.2, variance set)")
ax.set_xticks(ks)
ax.set_ylim(0.41, 0.51)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.legend(loc="lower right", frameon=False, fontsize=8.6)
ax.set_title("Best-of-$k$ gains track judge-noise capitalisation, not quality",
             fontsize=10.5, loc="left")
plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"{ROOT}/paper_rebuild/paper_a_bounded_returns/figures/fig_bestofn.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote fig_bestofn; best-of-N={B['best_of_N']:.4f} vs cluster {cluster:.4f} "
      f"(gap {B['gap_best_of_N_to_cluster']:+.4f}); sigma_within={B['pure_noise']['sigma_within_query']:.4f}; "
      f"best-of-N cost ${B['best_of_N_cost_usd']:.2f} vs cluster ${B['cluster_cost_usd']:.2f}")
