#!/usr/bin/env python3
"""PolyPythias 50-run reanalysis + cross-suite S&N proxy test with 10-seed arms.

Suite: 5 sizes (14m,31m,70m,160m,410m) x 10 seeds (seed0 = original Pythia run,
seeds1-9 = PolyPythias re-runs; seed varies init AND data order together - bundled).
Released evals (EleutherAI/polypythias-evals): blimp (aggregate acc), arc_challenge
(acc/acc_norm), lambada_openai (acc/perplexity), hendrycks_math (near-floor), bias tasks.

Outputs:
  tables/pp_final_seed_noise.csv   per (size,task,metric): 10-seed final relSD etc.
  tables/pp_outliers.csv           |z|>3 runs and coverage gaps
  tables/pp_t1_ratio.csv           late-step noise vs 10-seed noise per cell
  figs/PP_seednoise.png
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "data"
FINAL = 143000
SIZES = ["14m","31m","70m","160m","410m"]

df = pd.read_parquet(f"{OUT}/analysis/pp_tidy.parquet")
CELLS = [("lambada_openai","acc"), ("lambada_openai","perplexity"),
         ("arc_challenge","acc"), ("arc_challenge","acc_norm"), ("blimp","acc")]

def relsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.mean(v) == 0: return np.nan
    return np.std(v, ddof=1) / abs(np.mean(v))

# ---------- final-step seed noise ----------
rows, outl = [], []
for (task, metric) in CELLS:
    g = df[(df.task==task) & (df.metric==metric)]
    for size in SIZES:
        gs = g[g["size"]==size]
        fin = gs[gs.step==FINAL].set_index("seed")["value"]
        n = fin.notna().sum()
        if n < 4:
            rows.append(dict(size=size, task=task, metric=metric, n_seeds=int(n),
                             mean=np.nan, sd=np.nan, relsd=np.nan, spread=np.nan)); continue
        v = fin.dropna()
        med, mad = v.median(), (v - v.median()).abs().median()
        for s, val in v.items():
            z = 0.6745*(val-med)/mad if mad>0 else 0
            if abs(z) > 3:
                outl.append(dict(size=size, task=task, metric=metric, seed=s, value=val,
                                 robust_z=round(float(z),2), median=med))
        missing = sorted(set(range(10)) - set(v.index))
        if missing:
            outl.append(dict(size=size, task=task, metric=metric, seed=str(missing),
                             value=np.nan, robust_z=np.nan, median=np.nan))
        rows.append(dict(size=size, task=task, metric=metric, n_seeds=int(n),
                         mean=v.mean(), sd=v.std(ddof=1), relsd=relsd(v),
                         spread=v.max()-v.min()))
F = pd.DataFrame(rows)
F.to_csv(f"{OUT}/tables/pp_final_seed_noise.csv", index=False)
O = pd.DataFrame(outl)
O.to_csv(f"{OUT}/tables/pp_outliers.csv", index=False)
print("== final-step 10-seed noise ==")
print(F.round(4).to_string(index=False))
print("\n== outliers / coverage gaps ==")
print(O.to_string(index=False) if len(O) else "(none)")

# ---------- late-step noise vs seed noise (T1 cross-suite with 10 seeds) ----------
t1 = []
for (task, metric) in CELLS:
    g = df[(df.task==task) & (df.metric==metric)]
    for size in SIZES:
        gs = g[g["size"]==size]
        xs = []
        for seed, gr in gs.groupby("seed"):
            grr = gr[(gr.step >= 0.72*FINAL) & (gr.step <= FINAL)].sort_values("step")
            if len(grr) >= 3:
                xs.append(relsd(grr["value"]))
        fin = gs[gs.step==FINAL]["value"]
        y = relsd(fin)
        if not xs or not np.isfinite(y): continue
        x = np.nanmean(xs)
        t1.append(dict(size=size, task=task, metric=metric, x_step=x, y_seed=y,
                       ratio=y/x if x else np.nan, n_runs_x=len(xs),
                       n_pts_med=int(np.median([len(gr[(gr.step>=0.72*FINAL)&(gr.step<=FINAL)])
                                                for _, gr in gs.groupby("seed")]))))
T1 = pd.DataFrame(t1)
T1.to_csv(f"{OUT}/tables/pp_t1_ratio.csv", index=False)
print("\n== PP: late-step noise (x, avg over runs) vs 10-seed noise (y) ==")
print(T1.round(4).to_string(index=False))
print(f"\nmedian ratio y/x = {T1.ratio.median():.2f}  IQR=({T1.ratio.quantile(.25):.2f},{T1.ratio.quantile(.75):.2f})  n_cells={len(T1)}")

# ---------- seed-noise trajectory + figure ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ax = axes[0]
for task, metric, sty in [("arc_challenge","acc","-"), ("blimp","acc","--")]:
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZES)))
    for c, size in zip(colors, SIZES):
        gs = df[(df.task==task) & (df.metric==metric) & (df["size"]==size)]
        if gs.empty: continue
        sd_by_step = (gs.groupby("step")
                      .apply(lambda s: s["value"].std(ddof=1) if s["seed"].nunique()>=6 else np.nan,
                             include_groups=False).dropna())
        sd_by_step = sd_by_step[sd_by_step.index >= 1000]
        if len(sd_by_step) < 3: continue
        ax.plot(sd_by_step.index, sd_by_step.values, sty, marker="o", ms=2,
                color=c, label=f"{size} {task[:4]}", lw=1)
ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("cross-seed SD (acc)")
ax.legend(fontsize=6, ncol=2); ax.set_title("PolyPythias: seed-noise over training\n(solid arc_challenge, dashed blimp)")

ax = axes[1]
w = 0.15
for k, (task, metric) in enumerate(CELLS):
    sub = F[(F.task==task) & (F.metric==metric)]
    vals = [sub[sub["size"]==s].relsd.mean() for s in SIZES]
    ax.bar(np.arange(len(SIZES)) + (k - 2)*w, vals, width=w, label=f"{task}:{metric}"[:26])
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES)
ax.set_ylabel("10-seed relative SD @ final"); ax.set_yscale("log")
ax.legend(fontsize=6); ax.set_title("PolyPythias: final seed noise by size/task")

ax = axes[2]
for (task, metric), mk in zip(CELLS, ["o","s","^","v","D"]):
    sub = T1[(T1.task==task) & (T1.metric==metric)]
    xi = [SIZES.index(s) for s in sub["size"]]
    ax.scatter(xi, sub.ratio, marker=mk, label=f"{task}:{metric}"[:26], s=28)
ax.axhline(1, color="k", lw=0.8); ax.axhline(np.sqrt(2), color="orange", ls="--", lw=0.8)
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels(SIZES); ax.set_yscale("log")
ax.set_ylabel("ratio 10-seed noise / late-step noise")
ax.legend(fontsize=6); ax.set_title("PP: S&N proxy calibration (10-seed ground truth)")
plt.tight_layout(); plt.savefig(f"{OUT}/figs/PP_seednoise.png", bbox_inches="tight"); plt.close()
print("fig saved")
