#!/usr/bin/env python3
"""App E atlas (redesign of F6): noise is a function of the recipe.

Left:  the sigma atlas -- q10-q90 / q25-q75 bands and the median of per-cell
       3-seed final log-ppl SD across 275 cells per scale (log-log).
Right: recipe-only spread at fixed size (c4_en, 25 recipes, jittered rings)
       against the +/-1 SD band expected from n=3 sampling under homogeneity;
       the spread factor per group rides on top.
No in-axes legends: bands and lines carry right-rail labels.
Data: zero-gpu/analysis/t1c_ppl_cells.parquet. Paper numbers asserted.

Output: out/appe_atlas.pdf (4.9 x 2.2 in, appendix row).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PQ = os.path.join(os.environ.get("NFT_R", os.path.join(HERE, "..", "..", "data", "analysis")), "t1c_ppl_cells.parquet")

for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC", "font.size": 7.0,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.labelsize": 7.5, "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelcolor": "#333333", "ytick.labelcolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "pdf.fonttype": 3, "savefig.dpi": 300,
})

C_MAIN = "#003049"; C_REF = "#666666"; C_WARN = "#e76f51"; C_OK = "#2a9d8f"
C_STRIP = "#dcdcdc"  # PP palette: petrol / PP-orange / teal

df = pd.read_parquet(PQ)
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
SIZE_M = np.array([4,6,8,10,14,16,20,60,90,150,300,530,750,1000], dtype=float)

q = df.groupby("params")["y"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack().reindex(ORDER)
q10, q25, med, q75, q90 = [q[c].values for c in q.columns]

# ---- gates: tab:sigma + headline medians ------------------------------
assert abs(med[0] - 0.0160) < 5e-4 and abs(med[6] - 0.0288) < 5e-4
assert abs(med[9] - 0.0043) < 5e-4

# recipe-spread factors on c4_en (estimator: exp(sqrt(max(Var(log y)-0.411,0))))
SD_HOM = np.sqrt(0.411)
sub = df[df["dom"] == "c4_en"]
SIZES4 = ["4M", "20M", "150M", "1B"]
groups, factors = [], []
for sz in SIZES4:
    v = sub[sub["params"] == sz].sort_values("data")["y"].values
    assert len(v) == 25, (sz, len(v))
    groups.append(v)
    factors.append(float(np.exp(np.sqrt(max(np.var(np.log(v), ddof=1) - 0.411, 0.0)))))
assert abs(factors[0] - 2.29) < 0.03 and abs(factors[1] - 1.78) < 0.03
assert abs(factors[2] - 1.65) < 0.03 and factors[3] == 1.0
print("atlas gates pass: medians 0.0160/0.0288/0.0043; c4_en factors",
      [round(f, 2) for f in factors])

FIGW, FIGH = 4.9, 2.2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.095, right=0.885, top=0.83, bottom=0.19,
                    wspace=0.42)

# ---------------- left: quantile bands ----------------------------------
x = np.log10(SIZE_M)
a1.fill_between(x, q10, q90, color=C_MAIN, alpha=0.15, zorder=1, lw=0)
a1.fill_between(x, q25, q75, color=C_MAIN, alpha=0.32, zorder=2, lw=0)
a1.plot(x, med, color=C_MAIN, lw=1.2, zorder=3)
a1.scatter(x, med, s=7, c=C_MAIN, edgecolors="none", zorder=4)
a1.scatter([x[-1]], [med[-1]], s=18, c=C_MAIN, edgecolors="white",
           linewidths=0.5, zorder=5)
for yref, lab in [(0.005, "0.005"), (0.002, "0.002")]:
    a1.axhline(yref, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
    a1.annotate(lab, xy=(1.02, yref), xycoords=("axes fraction", "data"),
                fontsize=6.5, color=C_REF, ha="left", va="center",
                annotation_clip=False)
a1.set_yscale("log")
a1.set_ylim(5e-4, 1.1e-1)
a1.set_yticks([1e-3, 1e-2, 1e-1])
a1.set_yticklabels(["0.001", "0.01", "0.1"])
xt = [np.log10(v) for v in [4, 20, 90, 300, 1000]]
a1.set_xticks(xt)
a1.set_xticklabels(["4M", "20M", "90M", "300M", "1B"], rotation=28,
                   ha="right", rotation_mode="anchor")
a1.minorticks_off()
a1.set_xlabel("model scale")
a1.set_ylabel("3-seed SD of log-ppl (nat)")
from matplotlib.patches import Rectangle
a1.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=a1.transAxes,
                       facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                       clip_on=False, zorder=1))
a1.text(0.5, 1.080, "σ atlas (275 cells / scale)", transform=a1.transAxes,
        fontsize=7.5, fontweight="bold", color="#1a1a1a",
        ha="center", va="center")

# ---------------- right: recipe-only spread, c4_en ----------------------
rng = np.random.default_rng(0)
for i, (sz, v, f) in enumerate(zip(SIZES4, groups, factors)):
    xpos = i + 1
    medv = np.median(v)
    a2.fill_between([xpos - 0.32, xpos + 0.32],
                    medv * np.exp(-SD_HOM), medv * np.exp(SD_HOM),
                    color="0.45", alpha=0.25, zorder=1, lw=0)
    xj = xpos + rng.uniform(-0.17, 0.17, size=len(v))
    a2.scatter(xj, v, s=8, c=C_WARN if f > 1 else C_OK, edgecolors="none",
               alpha=0.9, zorder=3)
    a2.plot([xpos - 0.24, xpos + 0.24], [medv, medv], color="#1a1a1a",
            lw=1.4, zorder=4)
    a2.text(xpos, 0.30, f"{f:.2f}×", ha="center", va="top", fontsize=6.5,
            fontweight="bold", color=C_WARN if f > 1 else C_OK)
a2.text(4, 0.215, "homogeneity-\ncompatible", ha="center", va="top",
        fontsize=6.0, color=C_OK, linespacing=1.1)
a2.text(1, 0.215, "band: n=3\nsampling ±1σ", ha="center", va="top",
        fontsize=6.0, color="#666666", linespacing=1.1)
a2.set_yscale("log")
a2.set_ylim(1.2e-3, 4.5e-1)
a2.set_yticks([1e-2, 1e-1])
a2.set_yticklabels(["0.01", "0.1"])
a2.set_xlim(0.45, 4.55)
a2.set_xticks([1, 2, 3, 4])
a2.set_xticklabels(SIZES4)
a2.minorticks_off()
a2.set_xlabel("model scale (domain: c4_en)")
a2.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=a2.transAxes,
                       facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                       clip_on=False, zorder=1))
a2.text(0.5, 1.080, "recipe spread at fixed size", transform=a2.transAxes,
        fontsize=7.5, fontweight="bold", color="#1a1a1a",
        ha="center", va="center")

for ax in (a1, a2):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "appe_atlas.pdf"))
plt.close(fig)
print("wrote appe_atlas.pdf")
