#!/usr/bin/env python3
"""Fig 2 panel (a): separability SNR vs proxy scale -- the cliff line.

Two series, recomputed from dd_tidy.parquet with the exact audit() of
verify-scripts/rule3_repro.py (the reference implementation locked against
tab:snr digit-for-digit):
  observed     = sd(recipe means) / mean within-recipe sd          (measurement)
  deconvolved  = sqrt(max(var(means) - mean(sd^2)/3, 0)) / ...     (our estimand,
                 matches Table 1's SNR column; asserted below)

Design: rliable-Fig-1 cliff-line idiom -- SNR = 1 threshold dashed, the
noise-dominated region (<=60M) shaded vermillion 10%, the 60M->90M jump
annotated as a visual event; in-place legend (no legend box).

Output: out/fig2a_snr_cliff.pdf (1.78 x 1.66 in, matches the inversion panels
in the 3-across Fig 2 row). PNG regenerated from the PDF by the caller.
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
os.makedirs(OUT, exist_ok=True)

for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.size": 7.0,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 7.5,
    "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelcolor": "#333333",
    "ytick.labelcolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "pdf.fonttype": 3,
    "savefig.dpi": 300,
})

C_OBS = "#2a9d8f"    # observed / measurement (PP teal)
C_OBS_END = "#1d7268"  # observed endpoint dot (darker teal)
C_DEC = "#003049"    # deconvolved / claim (PP petrol, sampled from pp.pdf render)
C_WARN = "#d62828"   # warning region (PP red, sampled)
C_REF = "#666666"    # reference lines
C_GRID = "#d9dde2"   # PP/ggplot-style light grid
C_STRIP = "#dcdcdc"  # PP gray title strip

# ------------------------------------------------------------- data (canonical)
R_LAKE = os.environ.get("NFT_R", os.path.join(HERE, "..", "..", "data", "analysis"))
d = pd.read_parquet(f"{R_LAKE}/dd_tidy.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
SIZE_M = {"4M":4,"6M":6,"8M":8,"10M":10,"14M":14,"16M":16,"20M":20,"60M":60,
          "90M":90,"150M":150,"300M":300,"530M":530,"750M":750,"1B":1000}

def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt >= need].index
    return None if len(common) == 0 else common.max()

def audit_pair(scores):
    """Returns (deconvolved SNR, observed SNR); deconvolved = rule3_repro.audit."""
    n = scores.shape[1]
    mu, sd = scores.mean(1), scores.std(1, ddof=1)
    denom = np.sqrt((sd**2).mean())
    deconv = np.sqrt(max(mu.var(ddof=1) - (sd**2).mean() / n, 0.0)) / denom
    observed = np.sqrt(mu.var(ddof=1)) / denom
    return deconv, observed

EXPECTED = {  # tab:snr deconvolved column, locked (same dict as rule3_repro.py)
 "4M":0.30,"6M":0.63,"8M":0.33,"10M":0.00,"14M":0.29,"16M":0.72,"20M":0.77,
 "60M":1.23,"90M":2.61,"150M":3.18,"300M":3.57,"530M":2.98,"750M":2.52,"1B":3.52}

rows = []
bands = []
for p in ORDER:
    g = d[(d.params == p) & (d.task == "olmes_10_macro_avg")]
    scores = {}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None: continue
        v = gg[gg.step == s].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: scores[mix] = v.values[:3]
    M = np.array([scores[k] for k in sorted(scores)])
    dec, obs = audit_pair(M)
    assert round(dec, 2) == EXPECTED[p], f"{p}: deconv {dec:.3f} != table {EXPECTED[p]}"
    rows.append((p, SIZE_M[p], obs, dec))
    # PP-style band, honest version: bootstrap over RECIPES (seeded), 68% band of
    # each SNR statistic. Bands are figure-only context (paper claims use point
    # estimates already locked above); printed for audit.
    rng = np.random.default_rng(0)
    B = 2000
    n_rec = M.shape[0]
    bs = np.empty((B, 2))
    for b in range(B):
        Mb = M[rng.integers(0, n_rec, n_rec)]
        bs[b] = audit_pair(Mb)
    q = np.percentile(bs, [16, 84], axis=0)
    bands.append((q[0, 1], q[1, 1], q[0, 0], q[1, 0]))  # dec lo/hi, obs lo/hi

snr = pd.DataFrame(rows, columns=["scale", "size_M", "observed", "deconvolved"])
band = pd.DataFrame(bands, columns=["dec_lo", "dec_hi", "obs_lo", "obs_hi"])
snr = pd.concat([snr, band], axis=1)
snr.to_csv(os.path.join(OUT, "fig2a_snr_data.csv"), index=False)
print(snr.to_string(index=False))

# ------------------------------------------------------------- render
FIGW, FIGH = 1.78, 1.66
fig, ax = plt.subplots(figsize=(FIGW, FIGH))
fig.subplots_adjust(left=0.205, right=0.965, top=0.87, bottom=0.215)

x = np.log10(snr.size_M.values)
ax.set_xlim(np.log10(3.4), np.log10(1150))

# PP/ggplot grammar: light grid behind everything, gray title strip on top
ax.set_axisbelow(True)
ax.grid(True, color=C_GRID, lw=0.5)
from matplotlib.patches import Rectangle
strip = Rectangle((0, 1.018), 1.0, 0.105, transform=ax.transAxes,
                  facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                  clip_on=False, zorder=1)
ax.add_patch(strip)
ax.text(0.5, 1.070, "separability SNR", transform=ax.transAxes,
        fontsize=7.5, fontweight="bold", color="#1a1a1a",
        ha="center", va="center")

# noise-dominated region (<=60M): thin boundary line at the log-midpoint of the
# empty 60M-90M gap + small muted label (PP-style: no filled region)
EDGE = np.log10(np.sqrt(60 * 90))
ax.axvline(EDGE, ls=(0, (3, 3)), lw=0.8, color=C_WARN, alpha=0.55, zorder=1)
ax.axhline(1.0, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)

# series: observed (quiet teal) and deconvolved (claim petrol); PP marker grammar:
# small solid dot per point + larger endpoint dot (final-value anchor);
# PP-style same-color light band (68% recipe-bootstrap, seeded) under each line
for col, cc, ec in [("observed", C_OBS, C_OBS_END), ("deconvolved", C_DEC, C_DEC)]:
    lo = snr["obs_lo" if col == "observed" else "dec_lo"].values
    hi = snr["obs_hi" if col == "observed" else "dec_hi"].values
    ax.fill_between(x, lo, hi, color=cc, alpha=0.16, lw=0, zorder=1)
    ax.plot(x, snr[col], color=cc, lw=1.1, zorder=2)
    ax.scatter(x, snr[col], s=7, c=cc, edgecolors="none", zorder=3)
    ax.scatter([x[-1]], [snr[col].values[-1]], s=20, c=ec,
               edgecolors="white", linewidths=0.5, zorder=4)

# 60M->90M jump as a visual event: arrow along the deconvolved step + factor
x60, x90 = np.log10(60), np.log10(90)
y60 = float(snr.loc[snr.scale == "60M", "deconvolved"].iloc[0])
y90 = float(snr.loc[snr.scale == "90M", "deconvolved"].iloc[0])
ax.annotate("", xy=(x90, y90), xytext=(x60, y60),
            arrowprops=dict(arrowstyle="-|>", color=C_DEC, lw=1.0,
                            shrinkA=3, shrinkB=3), zorder=4)
ax.annotate(f"×{y90/y60:.1f}", xy=((x60 + x90) / 2, (y60 + y90) / 2),
            xytext=(4, 4), textcoords="offset points", fontsize=6.5,
            color=C_DEC, fontweight="bold", ha="left", va="bottom", zorder=4)

# in-place annotations (no legend box, no frames)
ax.text(np.log10(4.0), 3.62, "noise-dominated", fontsize=6.0, color=C_WARN,
        alpha=0.85, ha="left", va="center")
ax.text(np.log10(1100), 1.0, "SNR = 1", fontsize=6.5, color=C_REF,
        ha="right", va="bottom")
# legend in the verified-empty bottom-right; text right-aligned to the frame and
# swatches placed from the text's MEASURED extent (no unit-guessing)
lxr = np.log10(1080)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for y, cc, lab, tc in [(0.62, C_OBS, "observed", C_OBS_END),
                       (0.32, C_DEC, "deconvolved", C_DEC)]:
    t = ax.text(lxr, y, lab, fontsize=6.5, color=tc, va="center", ha="right")
    bb = t.get_window_extent(renderer)
    x_left = inv.transform((bb.x0, bb.y0))[0]
    ax.add_patch(Rectangle((x_left - 0.17, y - 0.13), 0.15, 0.26,
                           facecolor=cc, alpha=0.16, edgecolor="none", zorder=2))
    ax.plot([x_left - 0.16, x_left - 0.03], [y, y], color=cc, lw=1.1, zorder=3)

ax.set_ylim(-0.12, 4.0)
ax.set_yticks([0, 1, 2, 3, 4])
xt = [np.log10(v) for v in [4, 20, 90, 300, 1000]]
ax.set_xticks(xt); ax.set_xticklabels(["4M", "20M", "90M", "300M", "1B"])
ax.minorticks_off()
ax.set_xlabel("proxy scale")
ax.set_ylabel("SNR")

fig.savefig(os.path.join(OUT, "fig2a_snr_cliff.pdf"))
plt.close(fig)
print("wrote fig2a_snr_cliff.pdf")
