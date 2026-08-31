#!/usr/bin/env python3
"""Figures for T1/T2/T3."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "data"
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
plt.rcParams.update({"font.size": 9, "figure.dpi": 140})

# ---------------- T1 figure: R distribution + ceiling + ratio ----------------
R = pd.read_csv(f"{OUT}/tables/t1b_R_per_size_recipe.csv")
S = pd.read_csv(f"{OUT}/tables/t1b_summary_by_size.csv")
C = pd.read_parquet(f"{OUT}/analysis/t1b_cells.parquet")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axes[0]
data = [R[R.params==p].R_log.dropna() for p in SIZES]
bp = ax.boxplot(data, tick_labels=SIZES, showfliers=False, widths=0.6)
ceil = [R[R.params==p].ceil_med.median() for p in SIZES]
null = [R[R.params==p].null_hi.median() for p in SIZES]
ax.plot(range(1,15), ceil, "g--", lw=1.5, label="MC ceiling (proxy perfect,\n3-seed estimation limit)")
ax.plot(range(1,15), null, "r:", lw=1.5, label="null 95% (no relation)")
ax.axhline(0.9, color="k", lw=0.8, alpha=0.5); ax.text(0.3, 0.905, "R=0.9 (S&N)", fontsize=7)
ax.set_ylabel("Pearson R (log space), 10 OLMES tasks"); ax.set_xlabel("model size")
ax.set_title("T1: step-noise vs 3-seed noise correlation\nper (size, recipe); 25 recipes/box")
ax.tick_params(axis='x', rotation=60); ax.legend(fontsize=7, loc="lower left")

ax = axes[1]
med = [ (C[(C.params==p)].y/C[(C.params==p)].x).median() for p in SIZES]
q25 = [ (C[(C.params==p)].y/C[(C.params==p)].x).quantile(.25) for p in SIZES]
q75 = [ (C[(C.params==p)].y/C[(C.params==p)].x).quantile(.75) for p in SIZES]
ax.errorbar(range(14), med, yerr=[np.array(med)-np.array(q25), np.array(q75)-np.array(med)],
            fmt="o-", capsize=3, color="#1f6fb4")
ax.axhline(1.0, color="k", lw=0.8); ax.axhline(np.sqrt(2), color="orange", lw=0.8, ls="--")
ax.text(0.1, 1.435, r"$\sqrt{2}$ (two-source bundling)", fontsize=7, color="orange")
ax.set_xticks(range(14)); ax.set_xticklabels(SIZES, rotation=60)
ax.set_ylabel("median ratio  seed-noise / step-noise")
ax.set_title("T1: proxy calibration (accuracy metrics)\nmedian over 250 task-recipe cells")

ax = axes[2]
P = pd.read_csv(f"{OUT}/tables/t1c_ppl_summary_by_size.csv")
ax.plot(range(len(P)), P.med_ratio, "s-", color="#b02a2a", label="raw proxy (as published)")
ax.plot(range(len(P)), P.med_ratio_dt, "^-", color="#2a7a2a", label="detrended proxy")
ax.axhline(1.0, color="k", lw=0.8)
ax.set_xticks(range(len(P))); ax.set_xticklabels(P.params, rotation=60)
ax.set_yscale("log"); ax.set_ylabel("median ratio  seed / step noise (log ppl)")
ax.set_title("T1: continuous metric (perplexity)\ncoarse released grid")
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/figs/T1_proxy.png", bbox_inches="tight"); plt.close()

# ---------------- T2 figure: forest of init/order ratios ----------------
T = pd.read_csv(f"{OUT}/tables/t2_source_equivalence.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
for ax, mm, ttl in [(axes[0], "primary_like", "primary metrics (9 tasks)"),
                    (axes[1], "bits_per_byte", "bits-per-byte (18 tasks)")]:
    s = T[T.metric_mode==mm].sort_values("ratio").reset_index(drop=True)
    y = np.arange(len(s))
    ax.errorbar(s.ratio, y, xerr=[s.ratio-s.ci_lo_F, s.ci_hi_F-s.ratio],
                fmt="o", capsize=2, color="#1f6fb4", ms=4)
    ax.axvline(1, color="k", lw=0.9)
    ax.axvspan(0.5, 2.0, color="grey", alpha=0.12)
    ax.set_yticks(y); ax.set_yticklabels(s.task, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel(r"$\sigma_{init}/\sigma_{order}$ (final score, 10 vs 9 runs)")
    ax.set_title(ttl, fontsize=9)
fig.suptitle("T2: init-seed vs data-order noise, S&N 1B suite (grey = 2x equivalence band)", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/figs/T2_source_equivalence.png", bbox_inches="tight"); plt.close()

# ---------------- T3 figure: accuracy curve + error decomposition ----------------
R3 = pd.read_csv(f"{OUT}/tables/t3_decision_replay.csv")
macro = R3[R3.task=="olmes_10_macro_avg"].copy()
macro["order"] = macro["size"].map({s:i for i,s in enumerate(SIZES)})
macro = macro.sort_values("order")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax = axes[0]
ax.plot(macro["size"], macro.acc_seedmean, "o-", label="3-seed mean decisions")
ax.plot(macro["size"], macro.acc_singleseed, "s--", label="single-seed decisions (avg)")
ax.axhline(0.8, color="k", lw=0.8, alpha=0.6); ax.text(0.2, 0.805, "80% (paper)", fontsize=7)
ax.set_ylabel("decision accuracy vs 1B consensus"); ax.set_xlabel("decision size")
ax.tick_params(axis='x', rotation=60); ax.legend(fontsize=8)
ax.set_title("T3: OLMES-macro pairwise decision accuracy\n(300 recipe pairs; target = 1B 3-seed mean)")
ax = axes[1]
b1 = macro.err_stable
b2 = macro.err_tgt_unstable
b3 = macro.err_small_unstable
# note: small & tgt unstable can overlap; use any-decomposition: stable / any-unstable
b_any = macro.err_any_unstable
ax.bar(macro["size"], b_any, label="error on seed-unstable pair\n(small or 1B side flips across seeds)", color="#d08770")
ax.bar(macro["size"], macro.err - b_any, bottom=b_any, label="error on seed-stable pair", color="#5e81ac")
ax.set_ylabel("# wrong decisions (of 300 pairs)"); ax.tick_params(axis='x', rotation=60)
ax.legend(fontsize=7); ax.set_title("T3: error decomposition by seed stability")
plt.tight_layout(); plt.savefig(f"{OUT}/figs/T3_decisions.png", bbox_inches="tight"); plt.close()

# enrichment: instability among wrong vs all pairs (macro, 150M & 90M-750M avg)
Pp = pd.read_parquet(f"{OUT}/analysis/t3_pairs.parquet")
for size in ["90M","150M","300M"]:
    s = Pp[(Pp.task=="olmes_10_macro_avg") & (Pp["size"]==size)]
    any_uns = (s.small_unstable | s.tgt_unstable)
    print(f"macro {size}: P(unstable)={any_uns.mean():.2f}  P(unstable|wrong)={any_uns[~s.correct].mean():.2f} "
          f" P(unstable|correct)={any_uns[s.correct].mean():.2f}  enrich={(any_uns[~s.correct].mean()/any_uns.mean()):.2f}")
print("figs saved")
