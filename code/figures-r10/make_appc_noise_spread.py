#!/usr/bin/env python3
"""App C noise x spread plane (DD Fig 5 configuration, our t2 data).

Per-task scatter of noise (sigma_init, SD across seeds) vs spread
(sigma_order, SD across recipes) on the S&N 1B arms, log-log, square
decades so parity sits at a true 45 degrees. Circles = primary readout,
crosses = bpb readout; grey pair-lines join the 8 task identities measured
under both (hellaswag and winogrande cross the band -- the readout switch
changes the verdict, visible at a glance). Dashed diagonal = parity; dotted
diagonals = the [0.5, 2] equivalence band of Figure fig:t2 (whisper fill +
thin dashed boundaries per PP grammar, decoded in the caption). F intervals
live on the forest plot; here only the two excluders carry their F tick
(vermillion, anti-diagonal = the ratio coordinate), showing each clears
parity in opposite directions. Dense minerva cluster fans out to the right
with hairline leaders.

Data: zero-gpu/tables/t2_source_equivalence.csv (the forest's own table).
All paper-quoted numbers asserted in-script.

Output: out/appc_noise_spread.pdf/.png (~4.4 x 4.55 in, appendix square).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

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
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.labelcolor": "#333333", "ytick.labelcolor": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "pdf.fonttype": 3, "savefig.dpi": 300,
})

C_PRI = "#003049"; C_BPB = "#2a9d8f"; C_WARN = "#d62828"  # PP petrol/teal/red
C_REF = "#666666"; C_GRID = "#d9dde2"; C_STRIP = "#dcdcdc"; C_LAB = "#555555"

T = pd.read_csv(CSV)

# ---- gates: the paper's quoted numbers (tab:t2 + fig:t2 caption) -------
assert len(T) == 27
def row(task, mode):
    r = T[(T.task == task) & (T.metric_mode == mode)]
    assert len(r) == 1
    return r.iloc[0]
r = row("hellaswag", "primary_like")
assert abs(r.sd_init - 0.00428) < 1e-4 and abs(r.sd_order - 0.0124) < 5e-4
assert abs(r.ratio - 0.35) < 0.005 and abs(r.ci_lo_F - 0.17) < 0.01 and abs(r.ci_hi_F - 0.70) < 0.01
r = row("winogrande", "bits_per_byte")
assert abs(r.sd_init - 0.0232) < 1e-4 and abs(r.sd_order - 0.00982) < 1e-4
assert abs(r.ratio - 2.36) < 0.005 and abs(r.ci_lo_F - 1.13) < 0.01 and abs(r.ci_hi_F - 4.79) < 0.01
r = row("mmlu", "primary_like"); assert abs(r.ratio - 0.74) < 0.005
r = row("hellaswag", "bits_per_byte"); assert abs(r.ratio - 1.88) < 0.005
excl = T[(T.ci_hi_F < 1.0) | (T.ci_lo_F > 1.0)]
assert set(excl.task) == {"hellaswag", "winogrande"} and len(excl) == 2
assert ((T.ci_lo_F >= 0.5) & (T.ci_hi_F <= 2.0)).sum() == 0, "0/27 F intervals inside band"
print("gates pass: 27 rows; hellaswag-prim 0.35 [0.17,0.70]; winogrande-bpb 2.36 [1.13,4.79];"
      " 0/27 inside band")

pri = T[T.metric_mode == "primary_like"].set_index("task")
bpb = T[T.metric_mode == "bits_per_byte"].set_index("task")
shared = sorted(set(pri.index) & set(bpb.index))
assert len(shared) == 8

DISP = {  # DD-style short display names
    "minerva_math_algebra": "minerva algebra",
    "minerva_math_counting_and_probability": "minerva counting & prob",
    "minerva_math_geometry": "minerva geometry",
    "minerva_math_intermediate_algebra": "minerva interm algebra",
    "minerva_math_number_theory": "minerva number theory",
    "minerva_math_prealgebra": "minerva prealgebra",
    "minerva_math_precalculus": "minerva precalculus",
    "arc_challenge": "arc challenge", "arc_easy": "arc easy",
}
def disp(task):
    return DISP.get(task, task)

def color_of(mode, task):
    r = row(task, mode)
    if r.ci_hi_F < 1.0 or r.ci_lo_F > 1.0:
        return C_WARN
    return C_PRI if mode == "primary_like" else C_BPB

# ---------------- figure --------------------------------------------------
FIGW, FIGH = 4.4, 4.55
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.115, right=0.985, top=0.925, bottom=0.105)

LIM = (2e-3, 1.8e-1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*LIM); ax.set_ylim(*LIM)
ax.set_aspect("equal", adjustable="box")   # square decades -> true 45° parity

# equivalence band: whisper fill + dotted edges; parity dashed (decode in caption)
xs = np.array([LIM[0], LIM[1]])
ax.fill_between(xs, xs / 2, np.minimum(xs * 2, LIM[1]), color="0.45",
                alpha=0.07, lw=0, zorder=0)
ax.plot(xs, xs, ls=(0, (4, 3)), lw=0.8, color=C_REF, zorder=1)
ax.plot(xs, xs * 2, ls=(0, (1.5, 2.5)), lw=0.7, color=C_REF, zorder=1)
ax.plot(xs, xs / 2, ls=(0, (1.5, 2.5)), lw=0.7, color=C_REF, zorder=1)
# rail micro-labels just inside where the edges exit the panel
ax.text(0.072, 0.155, "×2", fontsize=6.0, color=C_REF, ha="right",
        va="top", rotation=45, rotation_mode="anchor")
ax.text(0.165, 0.082, "×½", fontsize=6.0, color=C_REF, ha="right",
        va="top", rotation=45, rotation_mode="anchor")

ax.set_axisbelow(True)
ax.grid(True, which="major", color=C_GRID, lw=0.5)

# pair lines (below points)
for t in shared:
    ax.plot([pri.loc[t, "sd_init"], bpb.loc[t, "sd_init"]],
            [pri.loc[t, "sd_order"], bpb.loc[t, "sd_order"]],
            color="0.45", lw=0.8, alpha=0.8, zorder=2)

# F-interval ticks ONLY for the two excluders: slope -1 (ratio coordinate)
for _, rr in T.iterrows():
    if not (rr.ci_hi_F < 1.0 or rr.ci_lo_F > 1.0):
        continue
    S = np.log10(rr.sd_init) + np.log10(rr.sd_order)
    pts = []
    for rr_ in (rr.ci_lo_F, rr.ci_hi_F):
        lx = (np.log10(rr_) + S) / 2
        ly = (S - np.log10(rr_)) / 2
        pts.append((10 ** lx, 10 ** ly))
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
            color=C_WARN, lw=1.0, alpha=0.85, zorder=3)

# points
for _, rr in T.iterrows():
    c = color_of(rr.metric_mode, rr.task)
    if rr.metric_mode == "primary_like":
        ax.scatter(rr.sd_init, rr.sd_order, s=17, marker="o", c=c,
                   edgecolors="white", linewidths=0.5, zorder=4)
    else:
        ax.scatter(rr.sd_init, rr.sd_order, s=17, marker="X", c=c,
                   edgecolors="white", linewidths=0.4, zorder=4)

def lab(task, mode, dx, dy, ha):
    src = pri if mode == "primary_like" else bpb
    rr = src.loc[task]
    ax.annotate(disp(task), xy=(rr.sd_init, rr.sd_order),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=6.0, color=C_LAB, ha=ha, va="center", zorder=5)

# pair / singleton labels (anchor chosen per task against occlusion)
lab("copycolors", "primary_like", -7, 6, "right")
lab("arc_challenge", "primary_like", 7, 3, "left")
lab("socialiqa", "primary_like", -6, 6, "right")
lab("hellaswag", "primary_like", -4, -8, "right")
lab("winogrande", "bits_per_byte", 0, -10, "center")  # straight below the red cross
lab("csqa", "bits_per_byte", 0, 8, "center")          # above its cross
lab("arc_easy", "bits_per_byte", -7, -4, "right")     # left-below its cross
lab("piqa", "bits_per_byte", -6, -4, "right")         # anchored at its cross
# primary singleton
lab("mmlu", "primary_like", -6, 0, "right")
# bpb singletons outside the minerva cluster
lab("humaneval", "bits_per_byte", 7, 2, "left")
lab("gsm8k", "bits_per_byte", 7, -2, "left")
lab("mbpp", "bits_per_byte", -7, 3, "right")

# minerva cluster: fan right with hairline leaders, ordered top->bottom
FAN = ["minerva_math_precalculus", "minerva_math_counting_and_probability",
       "minerva_math_geometry", "minerva_math_intermediate_algebra",
       "minerva_math_prealgebra", "minerva_math_algebra",
       "minerva_math_number_theory"]
y0, dyv = -2.26, -0.058   # log10 coords of the label column
for i, t in enumerate(FAN):
    rr = bpb.loc[t]
    ytxt = y0 + i * dyv
    ax.annotate(disp(t), xy=(rr.sd_init, rr.sd_order),
                xytext=(10 ** -2.14, 10 ** ytxt), textcoords="data",
                fontsize=6.0, color=C_LAB, ha="left", va="center", zorder=5,
                arrowprops=dict(arrowstyle="-", color="0.65", lw=0.45,
                                shrinkA=0.5, shrinkB=2.0))

# marker key, top-left inside (DD Fig 5 precedent) + line key for diagonals
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="none", markersize=4.5,
                  markerfacecolor=C_PRI, markeredgecolor="white",
                  markeredgewidth=0.5),
           Line2D([], [], marker="X", ls="none", markersize=4.5,
                  markerfacecolor=C_BPB, markeredgecolor="none"),
           Line2D([], [], color=C_REF, lw=0.8, ls=(0, (4, 3)))]
leg = ax.legend(handles, ["primary readout", "bpb readout",
                          "parity (dashed); band edges (dotted)"],
                loc="upper left", frameon=False, fontsize=6.2,
                handletextpad=0.5, borderaxespad=0.3, labelspacing=0.4)
for txt in leg.get_texts():
    txt.set_color("#333333")

ax.set_xticks([3e-3, 1e-2, 3e-2, 1e-1])
ax.set_xticklabels(["0.003", "0.01", "0.03", "0.1"])
ax.set_yticks([3e-3, 1e-2, 3e-2, 1e-1])
ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1"])
ax.minorticks_off()
ax.set_xlabel(r"noise:  $\hat\sigma_{\rm init}$  (SD across seeds)")
ax.set_ylabel(r"spread:  $\hat\sigma_{\rm order}$  (SD across recipes)")

ax.add_patch(Rectangle((0, 1.015), 1.0, 0.052, transform=ax.transAxes,
                       facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                       clip_on=False, zorder=1))
ax.text(0.5, 1.041, "noise × spread per task (S&N 1B arms)",
        transform=ax.transAxes, fontsize=7.5, fontweight="bold",
        color="#1a1a1a", ha="center", va="center")

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "appc_noise_spread.pdf"))
fig.savefig(os.path.join(OUT, "appc_noise_spread.png"))
plt.close(fig)
print("wrote appc_noise_spread.pdf/.png")
