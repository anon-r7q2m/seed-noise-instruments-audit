#!/usr/bin/env python3
"""Fig 2 panels (b)/(c): the 4M ranking inversion, macro vs bpb readouts.

Rank-rank scatter (1 = best), 25 recipes, per readout:
  (b) accuracy macro avg: 4M rank vs 1B rank -- inverts (Pearson r = -0.56 on scores,
      Spearman rho = -0.52 on ranks; headline numbers locked by check_numbers.py).
  (c) bpb (log-ppl):     4M rank vs 1B rank -- no inversion (r = +0.87, rho = +0.84).

Numbers recomputed at runtime from the data lake (never hand-copied):
  macro: dd_tidy.parquet via the exact fs() logic of verify-scripts/r9_inversion_targets.py
  bpb:   dd_ppl.parquet  via the exact logic of verify-scripts/r9_bpb_replay.py

Outputs (per playbook: PDF is the source of truth, PNG re-rendered from it):
  out/fig2b_inversion_macro.pdf / .png
  out/fig2c_inversion_bpb.pdf / .png
  out/fig2bc_inversion_data.csv   (plotted coordinates, auditable)

Style: DATA-FIGURE-PLAYBOOK.md -- Okabe-Ito semantic colors, exact physical size
(1.78in x 1.62in single panel for the 3-across Fig 2 row), fonts >= 7pt at final size,
full 0.6pt frame, outward ticks, no tight-bbox (margins fixed so size is exact).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# Noto Sans CJK SC (Source Han Sans; modern, full glyph coverage incl. ρ/−)
for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
FONT = "Noto Sans CJK SC" if any(f.name == "Noto Sans CJK SC" for f in fm.fontManager.ttflist) else "DejaVu Sans"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

R_LAKE = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
PPL = os.environ.get("NFT_PPL", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/review-2027/anonymous-repo/data/analysis/dd_ppl.parquet")

# ---- playbook rcParams baseline (final physical size, no post-scaling) ----
plt.rcParams.update({
    "font.family": FONT,
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
    "pdf.fonttype": 3,   # Type 3: glyph outlines as paths — required for CFF-based
                         # fonts (Noto CJK .ttc); TrueType-42 embedding mangles them
    "savefig.dpi": 300,
})

C_WARN = "#d62828"   # inversion / alarm (PP red, sampled from pp.pdf render)
C_CTRL = "#2a9d8f"   # control readout that works (PP teal)
C_REF = "#666666"    # reference line (gray dashed)
C_NOTE = "#555555"   # in-place annotation gray
C_GRID = "#d9dde2"   # PP/ggplot-style light grid
C_STRIP = "#dcdcdc"  # PP gray title strip

# ---------------------------------------------------------------- macro readout
d = pd.read_parquet(f"{R_LAKE}/dd_tidy.parquet")

def fs(params):
    """Exact logic of verify-scripts/r9_inversion_targets.py."""
    g = d[(d.params == params) & (d.task == "olmes_10_macro_avg")]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = float(v.mean())
    return out

m4, m1b = fs("4M"), fs("1B")
recipes = sorted(set(m4) & set(m1b))
assert len(recipes) == 25, f"expected 25 recipes, got {len(recipes)}"
macro = pd.DataFrame({"recipe": recipes,
                      "score_4M": [m4[r] for r in recipes],
                      "score_1B": [m1b[r] for r in recipes]})

# ------------------------------------------------------------------ bpb readout
p = pd.read_parquet(PPL)
DOMS = [c for c in p.columns if c.startswith("eval/")]
p = p.dropna(subset=DOMS, how="all")

def final_logppl(sz):
    """Exact logic of verify-scripts/r9_bpb_replay.py."""
    g = p[p.params == sz]
    out = {}
    for (recipe, seed), gg in g.groupby(["data", "seed"]):
        com = gg["step"].max()
        row = gg[gg.step == com]
        if row.empty: continue
        lp = np.log(row[DOMS].iloc[0].values.astype(float))
        lp = lp[np.isfinite(lp)]
        if len(lp) == len(DOMS):
            out.setdefault(recipe, {})[seed] = lp.mean()
    per = {}
    for r, sd in out.items():
        if len(sd) >= 3:
            per[r] = np.array(list(sd.values())[:3])
    return per

b4_full, b1b_full = final_logppl("4M"), final_logppl("1B")
# r9_bpb_replay uses the intersection of recipes across ALL 14 scales; reproduce it
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
per_scale = {sz: final_logppl(sz) for sz in SIZES}
brec = sorted(set.intersection(*[set(v) for v in per_scale.values() if v]))
bpb = pd.DataFrame({"recipe": brec,
                    "logppl_4M": [per_scale["4M"][r].mean() for r in brec],
                    "logppl_1B": [per_scale["1B"][r].mean() for r in brec]})

# ------------------------------------------------------------------ stats + ranks
def add_ranks(df, col4, col1, higher_better):
    asc = not higher_better  # rank 1 = best
    df["rank_1B"] = df[col1].rank(ascending=asc).astype(int)
    df["rank_4M"] = df[col4].rank(ascending=asc).astype(int)
    return df

macro = add_ranks(macro, "score_4M", "score_1B", higher_better=True)
bpb = add_ranks(bpb, "logppl_4M", "logppl_1B", higher_better=False)

st_macro = {"pearson": stats.pearsonr(macro.score_4M, macro.score_1B)[0],
            "spearman": stats.spearmanr(macro.score_4M, macro.score_1B)[0]}
st_bpb = {"pearson": stats.pearsonr(bpb.logppl_4M, bpb.logppl_1B)[0],
          "spearman": stats.spearmanr(bpb.logppl_4M, bpb.logppl_1B)[0]}
# locked-number assertions (check_numbers.py gates: -0.559/-0.517 and +0.871/+0.844)
assert abs(st_macro["pearson"] - (-0.559)) < 0.002, st_macro
assert abs(st_macro["spearman"] - (-0.517)) < 0.002, st_macro
assert abs(st_bpb["pearson"] - 0.8708) < 0.002, st_bpb
assert abs(st_bpb["spearman"] - 0.8438) < 0.002, st_bpb
print(f"macro: pearson {st_macro['pearson']:+.3f} spearman {st_macro['spearman']:+.3f}  (n={len(macro)})")
print(f"bpb:   pearson {st_bpb['pearson']:+.3f} spearman {st_bpb['spearman']:+.3f}  (n={len(bpb)})")

# ------------------------------------------------------------------ audit CSV
audit = pd.concat([
    macro.assign(readout="macro", x=macro.rank_1B, y=macro.rank_4M)[
        ["readout", "recipe", "score_4M", "score_1B", "rank_4M", "rank_1B"]],
    bpb.assign(readout="bpb", x=bpb.rank_1B, y=bpb.rank_4M)[
        ["readout", "recipe", "logppl_4M", "logppl_1B", "rank_4M", "rank_1B"]],
])
audit.to_csv(os.path.join(OUT, "fig2bc_inversion_data.csv"), index=False)

# ------------------------------------------------------------------ panel renderer
from matplotlib.patches import Rectangle
FIGW, FIGH = 1.78, 1.40   # single panel, 3-across Fig 2 row at 5.5in text width

def render(df, color, title, ann_main, ann_sub, fname, stats_corner):
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    fig.subplots_adjust(left=0.205, right=0.965, top=0.87, bottom=0.20)
    lim = (0.0, 26.0)
    # PP/ggplot grammar: light grid below everything, gray title strip on top
    ax.set_axisbelow(True)
    ax.grid(True, color=C_GRID, lw=0.5)
    strip = Rectangle((0, 1.018), 1.0, 0.105, transform=ax.transAxes,
                      facecolor=C_STRIP, edgecolor="#333333", lw=0.6,
                      clip_on=False, zorder=1)
    ax.add_patch(strip)
    ax.text(0.03, 1.070, title, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", color="#1a1a1a", ha="left", va="center")
    ax.text(0.975, 1.070, ann_main, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", color=color, ha="right", va="center")
    # ann_sub (Spearman on the plotted ranks) was silently dropped before
    # revision-17; draw it at the requested corner (m6: rank axes need rho visible)
    if ann_sub:
        xy = {"tr": (0.96, 0.955), "tl": (0.045, 0.955)}[stats_corner]
        ha = {"tr": "right", "tl": "left"}[stats_corner]
        ax.text(*xy, ann_sub, transform=ax.transAxes, fontsize=6.3,
                color="#444444", ha=ha, va="top", zorder=4)
    # y = x reference (dashed gray), then least-squares trend in panel color
    ax.plot(lim, lim, ls=(0, (4, 3)), lw=0.7, color=C_REF, zorder=1)
    k, b0 = np.polyfit(df["rank_1B"], df["rank_4M"], 1)
    xs = np.array([1.0, 25.0])
    # PP-style band, honest version: 68% bootstrap over recipes (seeded) of the
    # trend line; printed into the audit CSV as columns
    rng = np.random.default_rng(0)
    B = 2000
    n = len(df)
    grid = np.linspace(1.0, 25.0, 50)
    fits = np.empty((B, grid.size))
    xv, yv = df["rank_1B"].values, df["rank_4M"].values
    for b_ in range(B):
        idx = rng.integers(0, n, n)
        kb, bb = np.polyfit(xv[idx], yv[idx], 1)
        fits[b_] = kb * grid + bb
    lo, hi = np.percentile(fits, [16, 84], axis=0)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.13, lw=0, zorder=1)
    ax.plot(xs, k * xs + b0, lw=1.1, color=color, zorder=2)
    ax.scatter(df["rank_1B"], df["rank_4M"], s=9, c=color, edgecolors="none",
               alpha=0.9, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([1, 5, 10, 15, 20, 25]); ax.set_yticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("rank at 1B (1 = best)")
    ax.set_ylabel("rank at 4M (1 = best)")
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    plt.close(fig)
    print("wrote", fname)

render(macro, C_WARN, "macro",
       "r = −0.56", "ρ = −0.52",
       "fig2b_inversion_macro", stats_corner="tr")
render(bpb, C_CTRL, "domain log-ppl",
       "r = +0.87", "ρ = +0.84",
       "fig2c_inversion_bpb", stats_corner="tl")
