#!/usr/bin/env python3
"""App B panel 3: on perplexity the RAW proxy over-reads seed noise (trend
contamination); detrending repairs it at the two largest scales.

Per scale, cell-level median of truth-SD / proxy-SD on log-ppl cells
(t1c_ppl_cells.parquet; exact aggregation of verify-scripts/w5_yx_t1c_gaps.py):
  raw       y/x    -> 0.14..0.89  (proxy over-reads 1.1-7x)
  detrended y/x_dt -> ~1.03-1.08 at 750M/1B (repair), off-parity elsewhere.

Output: out/appb3_ppl_reversal.pdf (1.78 x 1.66 in).
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
PPL_CELLS = os.path.join(os.environ.get("NFT_R", os.path.join(HERE, "..", "..", "data", "analysis")), "t1c_ppl_cells.parquet")

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

C_WARN = "#d62828"; C_DEC = "#003049"; C_REF = "#666666"  # PP red / PP petrol
C_GRID = "#d9dde2"; C_STRIP = "#dcdcdc"

t = pd.read_parquet(PPL_CELLS)
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
SIZE_M = {"4M":4,"6M":6,"8M":8,"10M":10,"14M":14,"16M":16,"20M":20,"60M":60,
          "90M":90,"150M":150,"300M":300,"530M":530,"750M":750,"1B":1000}
rows = []
for p in ORDER:
    sp = t[t.params == p].dropna(subset=["x", "x_dt", "y"])
    sp = sp[(sp.x > 0) & (sp.y > 0) & (sp.x_dt > 0)]
    rows.append((p, float((sp.y / sp.x).median()),
                 float((sp.y / sp.x_dt).median()),
                 float((sp.y / sp.x).quantile(0.25)), float((sp.y / sp.x).quantile(0.75)),
                 float((sp.y / sp.x_dt).quantile(0.25)), float((sp.y / sp.x_dt).quantile(0.75))))
r = pd.DataFrame(rows, columns=["scale", "raw", "detrended",
                                "raw_lo", "raw_hi", "dt_lo", "dt_hi"])
r["size_M"] = r.scale.map(SIZE_M)
r.to_csv(os.path.join(OUT, "appb3_ppl_reversal_data.csv"), index=False)

# gates: paper claims "overestimates 1.1-7x" and "repair at the two largest scales"
ovr = 1.0 / r.raw
assert (1.05 < ovr.min() < 1.2) and (6.5 < ovr.max() < 7.5), (ovr.min(), ovr.max())
dt2 = r[r.scale.isin(["750M", "1B"])].detrended
assert dt2.between(0.95, 1.15).all(), dt2
print(r.to_string(index=False))

FIGW, FIGH = 1.78, 1.80
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.28, right=0.97, top=0.83, bottom=0.26)
x = np.log10(r.size_M.values)
ax.set_xlim(np.log10(3.4), np.log10(1150))
ax.set_yscale("log")
ax.set_ylim(0.10, 4.0)

ax.set_axisbelow(True)
ax.grid(True, color=C_GRID, lw=0.5)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=ax.transAxes,
                       facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                       clip_on=False, zorder=1))
_t = ax.text(0.5, 1.080, "perplexity: over-read", transform=ax.transAxes, fontsize=7.5,
             fontweight="bold", color="#1a1a1a", ha="center", va="center")
# shrink-to-fit: strip text never exceeds the axes width (PP/ggplot alignment)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
while _t.get_window_extent(_r).width > ax.get_window_extent(_r).width - 4 \
        and _t.get_fontsize() > 6.8:
    _t.set_fontsize(_t.get_fontsize() - 0.2)
ax.axhline(1.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
ax.fill_between(x, r.raw_lo.values, r.raw_hi.values, color=C_WARN, alpha=0.13,
                lw=0, zorder=1)
ax.fill_between(x, r.dt_lo.values, r.dt_hi.values, color=C_DEC, alpha=0.13,
                lw=0, zorder=1)
ax.plot(x, r.raw, color=C_WARN, lw=1.1, zorder=2)
ax.scatter(x, r.raw, s=7, c=C_WARN, edgecolors="none", zorder=3)
ax.scatter([x[-1]], [r.raw.iloc[-1]], s=20, c=C_WARN, edgecolors="white",
           linewidths=0.5, zorder=4)
ax.plot(x, r.detrended, color=C_DEC, lw=1.1, zorder=2)
ax.scatter(x, r.detrended, s=7, c=C_DEC, edgecolors="none", zorder=3)
ax.scatter([x[-1]], [r.detrended.iloc[-1]], s=20, c=C_DEC, edgecolors="white",
           linewidths=0.5, zorder=4)


def top_legend(fig, items, ax=None, y=0.956, fs=6.5):
    """PP-style mini legend row above the title strip. items: (color, label,
    kind) with kind in {line, dot, dash}. Centered as a group; positions from
    MEASURED text extents."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [fig.text(0, 0, lab, fontsize=fs, color=c, ha="left", va="center")
             for c, lab, _ in items]
    fig.canvas.draw()
    wids = [t.get_window_extent(r).width for t in texts]
    sw, gap, inter = 13, 3, 12
    total = sum(wids) + len(items) * (sw + gap) + (len(items) - 1) * inter
    W = fig.get_size_inches()[0] * fig.dpi
    if ax is not None:  # center on the axes box (strip-aligned), not the figure
        axbb = ax.get_window_extent(r)
        x = (axbb.x0 + axbb.x1) / 2 - total / 2
    else:
        x = (W - total) / 2
    for (c, lab, kind), t, w in zip(items, texts, wids):
        if kind != "dot":
            fig.lines.append(Line2D([x / W, (x + sw) / W], [y, y],
                                    transform=fig.transFigure, color=c,
                                    lw=1.1, ls=(0, (4, 3)) if kind == "dash"
                                    else "-"))
        if kind in ("dot", "line"):  # single centered marker
            fig.lines.append(Line2D([(x + sw / 2) / W], [y],
                                    transform=fig.transFigure, color=c,
                                    lw=0, marker="o", markersize=2.8,
                                    markeredgewidth=0, markerfacecolor=c))
        t.set_position(((x + sw + gap) / W, y))
        x += sw + gap + w + inter

top_legend(fig, ax=ax, items=[(C_DEC, "detrended", "line"), (C_WARN, "raw proxy", "line")])
# the one in-axes note sits small at the bottom (no tinted region anymore)
ax.text(np.log10(700), 0.117, "over-reads 1.1–7×", fontsize=6.0,
        color=C_WARN, alpha=0.9, ha="right", va="center")

ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
ax.set_yticklabels(["0.125", "0.25", "0.5", "1", "2", "4"])
xt = [np.log10(v) for v in [4, 20, 90, 300, 1000]]
ax.set_xticks(xt)
ax.set_xticklabels(["4M", "20M", "90M", "300M", "1B"], rotation=28,
                   ha="right", rotation_mode="anchor")
ax.minorticks_off()
ax.set_xlabel("model scale")
ax.set_ylabel("truth / proxy SD")

fig.savefig(os.path.join(OUT, "appb3_ppl_reversal.pdf"))
plt.close(fig)
print("wrote appb3_ppl_reversal.pdf")
