#!/usr/bin/env python3
"""App D inversion figure, alluvial variant (candidate vs the slopegraph).

Each recipe is a unit-width band flowing from its 4M rank slot (left ladder) to
its 1B rank slot (right ladder); smoothstep easing. Mass = gray-blue hairline
bands; the two extreme crossers (Dolma 1.7: 1->24; DCLM QC-7% FW2: 24->3) are
filled vermillion/green with end labels. Same data source as the scatter pair
(out/fig2bc_inversion_data.csv, gate-locked).

Output: out/appd_inversion_alluvial.pdf (4.9 x 2.4 in, appendix row).

(2026-09-03: PI 裁决——本图保留此版式，不做 PP 化；新版备份在 /tmp/appd_alluvial_new_backup.py)
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
    "pdf.fonttype": 3, "savefig.dpi": 300,
})

C_WARN = "#D55E00"; C_CTRL = "#009E73"; C_MASS = "#9aa5b1"; C_GRID = "#e8e8e8"

df = pd.read_csv(os.path.join(OUT, "fig2bc_inversion_data.csv"))
macro = df[df.readout == "macro"].reset_index(drop=True)
bpb = df[df.readout == "bpb"].reset_index(drop=True)

HILITE = {"Dolma1.7": "Dolma 1.7", "DCLM-Baseline (QC 7%, FW2)": "DCLM QC-7% FW2"}

T = np.linspace(0, 1, 60)
S = 3 * T**2 - 2 * T**3          # smoothstep easing


def alluvial_panel(ax, df, color, title, r_text):
    for rk in [1, 5, 10, 15, 20, 25]:
        ax.axhline(rk, color=C_GRID, lw=0.4, zorder=0)
    # mass bands (skip highlighted)
    for _, r in df.iterrows():
        if r.recipe in HILITE:
            continue
        y0, y1 = r.rank_4M, r.rank_1B
        lo = (y0 - 0.42) + ((y1 - 0.42) - (y0 - 0.42)) * S
        hi = (y0 + 0.42) + ((y1 + 0.42) - (y0 + 0.42)) * S
        ax.fill_between(T, lo, hi, color=C_MASS, alpha=0.22, lw=0, zorder=2)
        ax.plot(T, (y0 + (y1 - y0) * S), color=C_MASS, lw=0.4, alpha=0.5, zorder=2)
    # ladder bars anchoring the band ends
    for xx in (0.0, 1.0):
        ax.plot([xx, xx], [0.6, 25.4], color="#333333", lw=1.0, zorder=3,
                solid_capstyle="butt")
    # highlighted bands
    for rec, short in HILITE.items():
        r = df[df.recipe == rec]
        if r.empty:
            continue
        r = r.iloc[0]
        y0, y1 = r.rank_4M, r.rank_1B
        lo = (y0 - 0.42) + ((y1 - 0.42) - (y0 - 0.42)) * S
        hi = (y0 + 0.42) + ((y1 + 0.42) - (y0 + 0.42)) * S
        ax.fill_between(T, lo, hi, facecolor=color, edgecolor="white",
                        linewidth=0.5, alpha=0.9, zorder=4)
        ax.scatter([0, 1], [y0, y1], s=13, facecolors="white",
                   edgecolors=color, linewidths=0.9, zorder=5)
        lbl_kw = dict(fontsize=7.0, color=color, fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=1.0,
                                pad=0.6))
        n4 = 5 if y0 <= 2 else (5 if y0 >= 23 else -1)
        n1 = 5 if y1 <= 2 else (5 if y1 >= 23 else -1)
        ax.annotate(short, (0, y0), textcoords="offset points", xytext=(-4, n4),
                    ha="right", va="center", **lbl_kw)
        ax.annotate(short, (1, y1), textcoords="offset points", xytext=(4, n1),
                    ha="left", va="center", **lbl_kw)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(26.2, -0.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["4M", "1B"])
    ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("rank (1 = best)")
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.annotate("■", xy=(0, 1.02), xycoords="axes fraction", xytext=(0, 0),
                textcoords="offset points", fontsize=7.5, color=color,
                ha="left", va="bottom", annotation_clip=False)
    ax.annotate(title, xy=(0, 1.02), xycoords="axes fraction", xytext=(9, 0),
                textcoords="offset points", fontsize=7.5, fontweight="bold",
                color="#1a1a1a", ha="left", va="bottom", annotation_clip=False)
    ax.annotate(r_text, xy=(1, 1.02), xycoords="axes fraction", xytext=(0, 0),
                textcoords="offset points", fontsize=7.5, fontweight="bold",
                color=color, ha="right", va="bottom", annotation_clip=False)


fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.4))
fig.subplots_adjust(left=0.135, right=0.865, top=0.90, bottom=0.075, wspace=0.62)
alluvial_panel(axes[0], macro, C_WARN, "macro", "r = −0.56")
alluvial_panel(axes[1], bpb, C_CTRL, "bpb", "r = +0.87")
axes[1].set_ylabel("")
axes[1].set_yticklabels([])
fig.savefig(os.path.join(OUT, "appd_inversion_alluvial.pdf"))
plt.close(fig)
print("wrote appd_inversion_alluvial.pdf")
