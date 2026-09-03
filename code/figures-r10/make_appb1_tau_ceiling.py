#!/usr/bin/env python3
"""App B panel 1 (redesign): the proxy cannot rank recipes by their noise.

Dumbbell idiom (DataDecide Fig 5): per scale, a thin stem from the observed
task-median Kendall tau (gray-blue ring, near chance) to the perfect-proxy
ceiling (solid blue dot, ~0.53). The gap IS the claim -- no wiggle line, no
flattened band. Data: r9_perfect_proxy_ceiling.json per_task block; asserted
against the tab:t1-ranking tex values.

Output: out/appb1_tau_ceiling.pdf (1.78 x 1.66 in, Fig-2-row size).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
VS = os.path.join(HERE, "..", "..", "verify-scripts")

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

C_OBS = "#2a9d8f"; C_DEC = "#003049"; C_REF = "#666666"; C_STEM = "#d5dbe0"
C_GRID = "#d9dde2"; C_STRIP = "#dcdcdc"  # PP grammar

d = json.load(open(os.path.join(VS, "r9_perfect_proxy_ceiling.json")))
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
SIZE_M = {"4M":4,"6M":6,"8M":8,"10M":10,"14M":14,"16M":16,"20M":20,"60M":60,
          "90M":90,"150M":150,"300M":300,"530M":530,"750M":750,"1B":1000}

obs = np.array([d["per_task"][s]["obs_tau_median"] for s in ORDER])
ceil = np.array([d["per_task"][s]["ceil_tau_median_of_tasks"] for s in ORDER])

TEX = {"4M":(0.03,0.53),"6M":(-0.05,0.50),"8M":(0.09,0.56),"10M":(0.02,0.52),
       "14M":(0.09,0.56),"16M":(-0.04,0.49),"20M":(-0.00,0.54),"60M":(0.07,0.49),
       "90M":(0.06,0.51),"150M":(-0.02,0.56),"300M":(0.00,0.54),"530M":(0.05,0.56),
       "750M":(0.15,0.51),"1B":(0.00,0.53)}
for i, s in enumerate(ORDER):
    assert round(float(obs[i]), 2) == TEX[s][0], (s, obs[i], TEX[s][0])
    assert round(float(ceil[i]), 2) == TEX[s][1], (s, ceil[i], TEX[s][1])
print("14 scales match tab:t1-ranking")

FIGW, FIGH = 1.78, 1.80
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.28, right=0.97, top=0.83, bottom=0.26)
x = np.log10([SIZE_M[s] for s in ORDER])
ax.set_xlim(np.log10(3.4), np.log10(1150))

ax.set_axisbelow(True)
ax.grid(True, color=C_GRID, lw=0.5)
from matplotlib.patches import Rectangle
# strip spans figure width (rail zone above the axes is empty anyway)
ax.add_patch(Rectangle((0, 1.022), 1.0, 0.115, transform=ax.transAxes,
                       facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                       clip_on=False, zorder=1))
_t = ax.text(0.5, 1.080, "proxy ranks recipes", transform=ax.transAxes, fontsize=7.5,
             fontweight="bold", color="#1a1a1a", ha="center", va="center")
# shrink-to-fit: strip text never exceeds the axes width (PP/ggplot alignment)
fig.canvas.draw()
_r = fig.canvas.get_renderer()
while _t.get_window_extent(_r).width > ax.get_window_extent(_r).width - 4 \
        and _t.get_fontsize() > 6.8:
    _t.set_fontsize(_t.get_fontsize() - 0.2)
ax.axhline(0.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
# dumbbells: stem first, then endpoint markers
for xi, o, c in zip(x, obs, ceil):
    ax.plot([xi, xi], [o, c], color=C_STEM, lw=1.0, zorder=2,
            solid_capstyle="round")
ax.scatter(x, ceil, s=7, color=C_DEC, zorder=4)
ax.scatter(x, obs, s=7, color=C_OBS, edgecolors="none", zorder=5)


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

top_legend(fig, ax=ax, items=[(C_DEC, "ceiling", "dot"), (C_OBS, "observed", "dot")])

ax.set_ylim(-0.20, 0.66)
ax.set_yticks([0.0, 0.2, 0.4, 0.6])
xt = [np.log10(v) for v in [4, 20, 90, 300, 1000]]
ax.set_xticks(xt)
ax.set_xticklabels(["4M", "20M", "90M", "300M", "1B"], rotation=28,
                   ha="right", rotation_mode="anchor")
ax.minorticks_off()
ax.set_xlabel("proxy scale")
ax.set_ylabel(r"Kendall $\tau$")

fig.savefig(os.path.join(OUT, "appb1_tau_ceiling.pdf"))
plt.close(fig)
print("wrote appb1_tau_ceiling.pdf")
