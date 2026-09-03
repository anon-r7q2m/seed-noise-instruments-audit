#!/usr/bin/env python3
"""App D decision replay (redesign): pairwise recipe decisions vs the 1B
consensus, 300 pairs per scale.

Left:  decision accuracy by proxy scale, 3-seed means vs single seeds.
Right: the same errors stacked by seed-stability of the pair (vermillion =
pair flips across seeds on at least one side).
Rail labels instead of legends; the 80% reference is left as a dashed line
on the existing 0.8 tick (its meaning rides the caption -- a rail label
would collide with the single-seed line-end label in the gap).
Data: zero-gpu/tables/t3_decision_replay.csv (macro rows; paper numbers
asserted in-script).

Output: out/appd_replay.pdf (4.9 x 1.9 in, appendix row).
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
CSV = os.path.join(os.environ.get("NFT_TABLES", os.path.join(HERE, "..", "..", "data", "tables")), "t3_decision_replay.csv")

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

C_MEAN = "#003049"; C_SINGLE = "#2a9d8f"; C_REF = "#666666"  # PP petrol/teal
C_UNST = "#5ea39d"; C_STAB = "#cde5e2"; C_STAB_TX = "#003049"  # PP red / light petrol
C_GRID = "#d9dde2"; C_STRIP = "#dcdcdc"

def top_legend(fig, items, ax, y, fs=6.5):
    """PP-style mini legend row above a panel's title strip; centered on the
    axes box. items: (color, label, kind[, text_color]); kind in
    {line, dot, dash, patch}."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [fig.text(0, 0, lab, fontsize=fs,
                      color=(it[3] if len(it) > 3 else it[0]),
                      ha="left", va="center") for it in items
             for c, lab, *_ in [it]]
    fig.canvas.draw()
    wids = [t.get_window_extent(r).width for t in texts]
    sw, gap, inter = 13, 3, 12
    total = sum(wids) + len(items) * (sw + gap) + (len(items) - 1) * inter
    W = fig.get_size_inches()[0] * fig.dpi
    axbb = ax.get_window_extent(r)
    x = (axbb.x0 + axbb.x1) / 2 - total / 2
    for it, t, w in zip(items, texts, wids):
        c, lab, kind = it[0], it[1], it[2]
        if kind == "patch":
            fig.patches.append(Rectangle((x / W, y - 0.006), sw / W, 0.012,
                                         transform=fig.transFigure, facecolor=c,
                                         edgecolor="none"))
        else:
            fig.lines.append(Line2D([x / W, (x + sw) / W], [y, y],
                                    transform=fig.transFigure, color=c,
                                    lw=0.0 if kind == "dot" else 1.1,
                                    ls=(0, (4, 3)) if kind == "dash" else "-"))
            if kind in ("dot", "line"):
                fig.lines.append(Line2D([(x + sw / 2) / W], [y],
                                        transform=fig.transFigure, color=c, lw=0,
                                        marker="o", markersize=2.8,
                                        markeredgewidth=0, markerfacecolor=c))
        t.set_position(((x + sw + gap) / W, y))
        x += sw + gap + w + inter

def strip_title(fig, ax, title, color="#1a1a1a"):
    """PP/ggplot gray strip, exactly axes width; text shrink-to-fit >=6.8pt."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=ax.transAxes,
                           facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                           clip_on=False, zorder=1))
    t = ax.text(0.5, 1.080, title, transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", color=color, ha="center", va="center")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    while t.get_window_extent(r).width > ax.get_window_extent(r).width - 4 \
            and t.get_fontsize() > 6.8:
        t.set_fontsize(t.get_fontsize() - 0.2)

m = pd.read_csv(CSV)
m = m[m.task == "olmes_10_macro_avg"].reset_index(drop=True)
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M"]
assert m["size"].tolist() == ORDER

# ---- gates: paper numbers ---------------------------------------------
assert (m.n_pairs == 300).all()
assert (m.err == m.err_any_unstable + m.err_stable).all()
g = lambda s, c: float(m.loc[m["size"] == s, c].iloc[0])
# "80% at 150M x 1 seed vs 76% at 60M x 3" (rule-3 scale-beats-seeds line)
assert abs(g("150M", "acc_singleseed") - 0.802) < 0.002
assert abs(g("60M", "acc_seedmean") - 0.763) < 0.002
# majority of errors on seed-unstable pairs at every scale >= 8M
assert (m.share_err_any_unstable[8:] > 0.5).all()
print("replay gates pass: 300 pairs/scale, 150M single .802, 60M mean .763, "
      "unstable share > .5 from 8M up")

x = np.arange(len(ORDER))
XT = list(range(0, len(ORDER), 2))   # every other scale: density control
FIGW, FIGH = 4.9, 2.05
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, FIGH))
# legends moved to top rows (PP); the old rail space returns to the panels
fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.24,
                    wspace=0.38)

# ---------------- left: accuracy curves --------------------------------
a1.set_axisbelow(True)
a1.grid(True, color=C_GRID, lw=0.5)
a1.axhline(0.8, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
import numpy as _np
for col, cc in [("acc_seedmean", C_MEAN), ("acc_singleseed", C_SINGLE)]:
    se = _np.sqrt(m[col] * (1 - m[col]) / 300)   # pair-level binomial SE
    a1.fill_between(x, m[col] - se, m[col] + se, color=cc, alpha=0.13, lw=0,
                    zorder=1)
    a1.plot(x, m[col], color=cc, lw=1.1, zorder=3)
    a1.scatter(x, m[col], s=7, c=cc, edgecolors="none", zorder=4)
    a1.scatter([x[-1]], [m[col].iloc[-1]], s=18, c=cc, edgecolors="white",
               linewidths=0.5, zorder=5)
a1.set_ylim(0.25, 0.95)
a1.set_yticks([0.4, 0.6, 0.8])
a1.set_ylabel("decision accuracy")
strip_title(fig, a1, "decision replay vs 1B")
top_legend(fig, [(C_MEAN, "3-seed mean", "line", "#1a1a1a"),
                 (C_SINGLE, "single seed", "line", "#1a1a1a")], a1, y=0.945)

# ---------------- right: errors by pair stability ----------------------
a2.set_axisbelow(True)
a2.grid(True, axis="y", color=C_GRID, lw=0.5)
a2.bar(x, m.err_any_unstable, width=0.62, color=C_UNST, zorder=2)
a2.bar(x, m.err_stable, width=0.62, bottom=m.err_any_unstable, color=C_STAB,
       zorder=2)
a2.set_ylim(0, 215)
a2.set_yticks([0, 100, 200])
pass  # y-label folded into the title: the gap rail belongs to panel 1
strip_title(fig, a2, "errors by pair stability (of 300)")
top_legend(fig, [(C_UNST, "unstable", "patch", "#1a1a1a"),
                 (C_STAB, "stable", "patch", "#1a1a1a")], a2, y=0.945)

for ax in (a1, a2):
    ax.set_xticks(list(XT))
    ax.set_xticklabels([ORDER[i] for i in XT], rotation=28, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.minorticks_off()
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "appd_replay.pdf"))
plt.close(fig)
print("wrote appd_replay.pdf")
