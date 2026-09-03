#!/usr/bin/env python3
"""App C forest plot (redesign): source equivalence, task by task.

Per-task sigma_init/sigma_order with F-interval CIs vs the [0.5, 2] band.
Single column of 27 rows (grouped: 9 primary, gap, 18 bpb) -- two side-by-side
panels were tried and rejected: the minerva_math_* labels are ~34 chars and
physically cannot fit between or beside panels without occlusion.
rliable idiom: CI = horizontal bar, point estimate = black tick (no errorbar
caps). The two tasks whose CI excludes 1 (hellaswag below, winogrande above)
are highlighted in vermillion -- the caption's "opposite directions" claim is
visible at a glance. Data: zero-gpu/tables/t2_source_equivalence.csv; the
0/27 + 2-excluders claims are asserted in-script.

Output: out/appc_forest.pdf (4.2 x 5.2 in, appendix tall figure).
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
CSV = os.environ.get("NFT_T2", os.path.join(os.path.join(HERE, "..", "..", "data", "tables"), "t2_source_equivalence.csv"))

for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC", "font.size": 7.0,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.labelsize": 7.5, "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 7.0, "ytick.labelsize": 6.5,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelcolor": "#333333", "ytick.labelcolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 0.0,
    "xtick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.5,
    "pdf.fonttype": 3, "savefig.dpi": 300,
})

C_PRI = "#003049"; C_BPB = "#2a9d8f"; C_WARN = "#d62828"  # PP petrol/teal/red
C_REF = "#666666"; C_BAND = "#888888"
C_GRID = "#d9dde2"; C_STRIP = "#dcdcdc"

T = pd.read_csv(CSV)

# ---- gates: the caption's three claims -------------------------------
assert len(T) == 27
assert ((T.ci_lo_F >= 0.5) & (T.ci_hi_F <= 2.0)).sum() == 0, "0/27 inside band"
excl_lo = T[T.ci_hi_F < 1.0].task.tolist()   # CI entirely below 1
excl_hi = T[T.ci_lo_F > 1.0].task.tolist()   # CI entirely above 1
assert excl_lo == ["hellaswag"] and excl_hi == ["winogrande"], (excl_lo, excl_hi)
print("0/27 inside [0.5,2]; excluders:", excl_lo, "(below) /", excl_hi, "(above)")

# ---- single column: primary group on top, then a gap, then bpb -------
groups = [("primary_like", "primary (9 tasks)", C_PRI),
          ("bits_per_byte", "bpb (18 tasks)", C_BPB)]
rows = []   # (y, ratio, lo, hi, color)
ticks, labels, label_colors = [], [], []
headers = []  # (y, text, color) -- each in its own dedicated empty slot
ypos = 0.0
for mm, ttl, cc in groups:
    s = T[T.metric_mode == mm].sort_values("ratio", ascending=False)
    headers.append((ypos, ttl, cc))
    ypos -= 0.9   # header slot: no data row here, so no occlusion possible
    for _, row in s.iterrows():
        excl = row.ci_hi_F < 1.0 or row.ci_lo_F > 1.0
        c = C_WARN if excl else cc
        rows.append((ypos, row.ratio, row.ci_lo_F, row.ci_hi_F, c))
        ticks.append(ypos)
        labels.append(row.task)
        label_colors.append(C_WARN if excl else "#333333")
        ypos -= 1.0
    ypos -= 0.6  # group gap

FIGW, FIGH = 4.2, 5.2
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.435, right=0.975, top=0.975, bottom=0.065)

ax.axvspan(0.5, 2.0, color=C_BAND, alpha=0.08, zorder=0)
ax.axvline(1.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
ax.set_axisbelow(True)
ax.grid(True, axis="x", color=C_GRID, lw=0.5)
for yy, ratio, lo, hi, c in rows:
    ax.barh(yy, hi - lo, left=lo, height=0.62, color=c,
            alpha=0.42 if c == C_WARN else 0.28, zorder=2, lw=0)
    ax.plot([ratio, ratio], [yy - 0.31, yy + 0.31], color="#1a1a1a",
            lw=1.1, zorder=3)

ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=6.5)
for lbl, c in zip(ax.get_yticklabels(), label_colors):
    lbl.set_color(c)

# group headers as ggplot/PP-style gray strip rows spanning the panel width
# (each occupies its own dedicated empty slot -- no data row there by construction)
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
for yy, ttl, cc in headers:
    ax.add_patch(Rectangle((0.0, yy - 0.45), 1.0, 0.9, transform=trans,
                           facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                           zorder=4, clip_on=False))
    ax.text(0.5, yy, ttl, transform=trans, fontsize=7.0, fontweight="bold",
            color="#1a1a1a", ha="center", va="center", zorder=5)

ax.set_xscale("log")
ax.set_xlim(0.12, 6.0)
ax.set_xticks([0.2, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
ax.minorticks_off()
ax.set_ylim(ypos + 0.5, 0.7)
ax.set_xlabel(r"$\sigma_{\rm init}\,/\,\sigma_{\rm order}$"
              "   (shaded: equivalence band [0.5, 2])")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "appc_forest.pdf"))
plt.close(fig)
print("wrote appc_forest.pdf")
