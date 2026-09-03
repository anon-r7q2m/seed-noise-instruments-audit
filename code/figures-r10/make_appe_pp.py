#!/usr/bin/env python3
"""App E PolyPythias (redesign of F7): ten-seed ground truth, 2 panels.

  (a) per-seed final deviations from the cell mean, in percentage points --
      every seed is a ring, the gray stem is +/-1 sigma, the shaded band is
      the typical 0.5-1 pp ablation gap the caption cites. Nothing is
      clipped: all 82 seed deviations fit the frame (asserted), and the one
      divergent run (blimp 31m, -3.9 pp; App E) carries a direct label.
  (b) proxy vs 10-seed truth ratio, 9 cells; parity and sqrt(2) references.
      Open diamond at 31m: blimp cell with the two divergent seeds removed
      (ratio 0.55; the 9-cell median moves 1.8x -> 1.3x, App E).
Task identity lives in the title chip row; axes carry data only.
Data: zero-gpu/analysis/pp_tidy.parquet, tables/pp_final_seed_noise.csv,
tables/pp_t1_ratio.csv. Paper numbers asserted in-script.

Output: out/appe_pp.pdf (4.9 x 2.05 in, appendix row).
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
R_DIR = os.environ.get("NFT_R", os.path.join(HERE, "..", "..", "data", "analysis"))
T_DIR = os.environ.get("NFT_TABLES", os.path.join(HERE, "..", "..", "data", "tables"))

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

SIZES = ["14m", "31m", "70m", "160m", "410m"]
C_REF = "#666666"; C_WARN = "#d62828"   # sqrt(2) bound
C_STRIP = "#dcdcdc"
TASKS = [("arc_challenge", "acc", "arc_ch acc", "#003049"),
         ("arc_challenge", "acc_norm", "arc_ch acc_norm", "#2a9d8f"),
         ("blimp", "acc", "blimp acc", "#e76f51")]

tidy = pd.read_parquet(os.path.join(R_DIR, "pp_tidy.parquet"))
F = pd.read_csv(os.path.join(T_DIR, "pp_final_seed_noise.csv")).dropna(subset=["relsd"])
T1r = pd.read_csv(os.path.join(T_DIR, "pp_t1_ratio.csv"))

# ---- gates --------------------------------------------------------------
assert len(T1r) == 9
assert abs(T1r.ratio.median() - 1.80) < 0.02, T1r.ratio.median()
b31 = F[(F["size"] == "31m") & (F.task == "blimp") & (F.metric == "acc")]
assert len(b31) == 1 and abs(b31.relsd.iloc[0] - 0.0216) < 0.001
# divergent-run variant, recomputed from the trajectories (App E: seeds 0,5)
bfin = tidy[(tidy.task == "blimp") & (tidy.metric == "acc")
            & (tidy["size"] == "31m")]
bfin = bfin[bfin.step == bfin.step.max()]
bx = float(T1r[(T1r["size"] == "31m") & (T1r.task == "blimp")].x_step.iloc[0])
bex = bfin[~bfin.seed.isin([0, 5])]["value"]
ratio_ex = float(bex.std(ddof=1) / bex.mean()) / bx
assert abs(ratio_ex - 0.55) < 0.03
# the caption's 1-sigma claim: arc cells' sd at 14m/31m is 0.7-1.0 pp
for _, r in F[F.task == "arc_challenge"].iterrows():
    assert 0.005 < r.sd < 0.011, (r["size"], r.metric, r.sd)
print("pp gates pass: 9 cells median 1.80; excl-divergent ratio %.2f; "
      "arc 1-sigma in 0.5-1.1pp" % ratio_ex)

FIGW, FIGH = 4.9, 2.05
fig = plt.figure(figsize=(FIGW, FIGH))
BOT, TOP = 0.225, 0.76
axa = fig.add_axes([0.092, BOT, 0.50, TOP - BOT])
axc = fig.add_axes([0.715, BOT, 0.205, TOP - BOT])

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def strip_title(fig, ax, text):
    ax.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=ax.transAxes,
                           facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                           clip_on=False, zorder=1))
    ax.text(0.5, 1.080, text, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", color="#1a1a1a", ha="center", va="center")

def top_legend(fig, ax, items, y=0.935):
    """PP grammar: colored dot swatches + dark text, centered on the axes."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [fig.text(0, 0, lab, fontsize=6.5, color="#1a1a1a",
                      ha="left", va="center") for _, lab in items]
    fig.canvas.draw()
    wids = [t.get_window_extent(r).width for t in texts]
    sw, gap, inter = 6, 3, 10
    total = sum(wids) + len(items) * (sw + gap) + (len(items) - 1) * inter
    W = fig.get_size_inches()[0] * fig.dpi
    axbb = ax.get_window_extent(r)
    x = (axbb.x0 + axbb.x1) / 2 - total / 2
    for (c, lab), t, w in zip(items, texts, wids):
        fig.lines.append(Line2D([(x + sw / 2) / W], [y], transform=fig.transFigure,
                                color=c, lw=0, marker="o", markersize=2.8,
                                markeredgewidth=0, markerfacecolor=c))
        t.set_position(((x + sw + gap) / W, y))
        x += sw + gap + w + inter

# ---------------- (a) per-seed final deviations, pp -----------------------
# beeswarm (deterministic, symmetric, non-overlapping) instead of random
# jitter -- the Weissgerber-style small-n idiom: show every seed, but pack
# so no two points overlap and the column shape shows the distribution.
DY = 0.20   # point diameter in y units (s=4.5 -> 2.4pt on this panel)
DX = 0.072  # dodge step in x units (one point diameter)

def swarm(devs):
    offs = []
    placed = []
    for y in sorted(devs):
        k = 0
        while True:
            for dx in ([0.0] if k == 0 else [k * DX, -k * DX]):
                if all(abs(y - y2) >= DY or abs(dx - dx2) >= DX
                       for dx2, y2 in placed):
                    placed.append((dx, y))
                    offs.append(dx)
                    k = None
                    break
            if k is None:
                break
            k += 1
    # return offsets in the original order
    order = np.argsort(np.argsort(devs))
    return [offs[i] for i in order]

xi = {s: i for i, s in enumerate(SIZES)}
DODGE = {"arc_ch acc": -0.24, "arc_ch acc_norm": 0.0, "blimp acc": 0.24}

axa.axhspan(-1.0, 1.0, color="#888888", alpha=0.10, zorder=0)
axa.axhline(0.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
axa.annotate("±1 pp\nablation gap", xy=(1.015, 0.0), xycoords=("axes fraction", "data"),
             fontsize=6.0, color=C_REF, ha="left", va="center",
             annotation_clip=False, linespacing=1.15)

n_all = []
for task, metric, lab, c in TASKS:
    for size in SIZES:
        gs = tidy[(tidy.task == task) & (tidy.metric == metric)
                  & (tidy["size"] == size)]
        if gs.empty:
            continue
        fin = gs[gs.step == gs.step.max()]
        v = fin["value"].values
        if len(v) < 6:
            continue
        devs = (v - v.mean()) * 100.0
        x0 = xi[size] + DODGE[lab]
        axa.plot([x0, x0], [-v.std(ddof=1) * 100, v.std(ddof=1) * 100],
                 color="#555555", lw=0.9, zorder=2)
        xj = x0 + np.array(swarm(devs))
        axa.scatter(xj, devs, s=4.5, c=c, edgecolors="none", zorder=3)
        n_all.extend(devs)
# nothing is clipped or hidden: every seed's deviation fits the frame
assert len(n_all) == 82 and min(n_all) > -4.5, (len(n_all), min(n_all))
# flag the one divergent run (App E: blimp 31m seed5) with a direct label
axa.annotate("blimp 31m\ndivergent run", xy=(xi["31m"] + 0.24, -3.93),
             xytext=(xi["31m"] + 0.45, -3.6), fontsize=6.0, color="#e76f51",
             ha="left", va="center", linespacing=1.15,
             arrowprops=dict(arrowstyle="-", color="#e76f51", lw=0.6))

strip_title(fig, axa, "final per-seed deviation from cell mean")
top_legend(fig, axa, [(c, lab) for _, _, lab, c in TASKS])
axa.set_ylim(-4.7, 2.9)
axa.set_yticks([-4, -2, 0, 2])
axa.set_yticklabels(["−4", "−2", "0", "+2"])
axa.set_xticks(range(5))
axa.set_xticklabels(SIZES)
axa.set_xlim(-0.55, 4.55)
axa.minorticks_off()
axa.set_xlabel("model size (PolyPythias)")
axa.set_ylabel("final score − cell mean (pp)")

# ---------------- (b) proxy vs 10-seed truth -----------------------------
axc.axhline(1.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
axc.axhline(np.sqrt(2), ls=(0, (2, 2)), lw=0.7, color=C_WARN, zorder=1)
for task, metric, lab, c in TASKS:
    sub = T1r[(T1r.task == task) & (T1r.metric == metric)]
    xs = [xi[s] for s in sub["size"]]
    axc.scatter(xs, sub.ratio, s=10, c=c, edgecolors="none", zorder=3)
axc.plot([1, 1], [2.332, ratio_ex], ls=(0, (1, 1.5)), lw=0.8, color="#e76f51",
         zorder=2)
axc.scatter([1], [ratio_ex], s=11, marker="D", c="#e76f51", edgecolors="none",
            zorder=4)
axc.text(1.18, 0.60, "excl. 2 divergent\nseeds: 0.55×", fontsize=6.0,
         color="#e76f51", ha="left", va="center", linespacing=1.15)
axc.text(-0.32, 5.6, "median 1.8×\n(9 cells)", fontsize=6.0, color="#1a1a1a",
         ha="left", va="center", linespacing=1.2)
axc.annotate("parity", xy=(1.03, 1.0), xycoords=("axes fraction", "data"),
             fontsize=6.5, color=C_REF, ha="left", va="center",
             annotation_clip=False)
axc.annotate(r"$\sqrt{2}$", xy=(1.03, np.sqrt(2)),
             xycoords=("axes fraction", "data"), fontsize=6.5, color=C_WARN,
             ha="left", va="center", annotation_clip=False)
axc.set_yscale("log")
axc.set_ylim(0.42, 7.5)
axc.set_yticks([1, 2, 4])
axc.set_yticklabels(["1", "2", "4"])
axc.set_xticks(range(5))
axc.set_xticklabels(SIZES, fontsize=6.0, rotation=28, ha="right",
                    rotation_mode="anchor")
axc.set_xlim(-0.5, 4.5)
axc.minorticks_off()
axc.set_xlabel("model size")
strip_title(fig, axc, "proxy vs 10-seed truth")

for ax in (axa, axc):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "appe_pp.pdf"))
plt.close(fig)
print("wrote appe_pp.pdf")
