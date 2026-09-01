#!/usr/bin/env python3
"""R11-Q5: what drives the 60M->90M SNR jump -- signal rise or noise drop?

Table 1 (deconvolved): 60M signal 0.01034 noise 0.00766 -> SNR 1.23;
90M signal 0.01647 noise 0.00469 -> SNR 2.61. The 60M noise (0.00766) exceeds
both 20M (0.00649) and 90M (0.00469) -- is 60M noise anomalous (grid artifact,
one recipe), i.e. is the jump partly a 60M-noise artifact?

Decompose: log SNR jump = dlog(signal) - dlog(noise).
Checks: (1) per-recipe noise distribution at 60M vs 20M/90M (outlier-driven?);
(2) checkpoint grid density at 60M (grid_n in t1b_cells) vs neighbours;
(3) the same decomposition on the bpb readout (App G) for contrast.
Outputs r9_jump_decomposition.json. Pure CPU, deterministic.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

TASKS10 = ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu","openbookqa","piqa","socialiqa","winogrande"]

def cell_stats(params, task="olmes_10_macro_avg", col="primary_metric"):
    """recipe -> (3-seed mean, 3-seed sd) at common final step."""
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique()
        com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")[col].mean()
        if len(v) >= 3: out[mix] = (v.mean(), v.std(ddof=1))
    return out

def snr_parts(params, **kw):
    cs = cell_stats(params, **kw)
    means = np.array([v[0] for v in cs.values()])
    sds = np.array([v[1] for v in cs.values()])
    sds = sds[~np.isnan(sds)]
    if len(sds) < 10:
        return None
    S2 = means[~np.isnan(means)].var(ddof=1)
    noise_rms = np.sqrt((sds**2).mean())
    noise_med = float(np.median(sds))
    sig_dec2 = max(S2 - (sds**2).mean()/3, 0.0)
    return {"n_recipes": len(cs), "signal_obs": float(np.sqrt(S2)),
            "signal_deconv": float(np.sqrt(sig_dec2)),
            "noise_rms": float(noise_rms), "noise_median": noise_med,
            "snr_deconv": float(np.sqrt(sig_dec2)/noise_rms) if noise_rms>0 else None,
            "per_recipe_sd": {r: float(v[1]) for r, v in cs.items()}}

OUT = {"accuracy": {}, "bpb": {}}
SIZES = ["20M","60M","90M","150M"]
for sz in SIZES:
    OUT["accuracy"][sz] = snr_parts(sz)
    # bpb: mean over the 10 tasks per recipe (macro bpb), not the macro row
    g = d[(d.params == sz) & (d.task.isin(TASKS10))]
    rec_bpb = {}
    for mix, gg in g.groupby("data"):
        per_seed = gg.groupby(["seed","task"])["bits_per_byte_corr"].last().groupby("seed").mean()
        # align to common final step per task: simpler -- take each task's final common step
        vals = []
        for t, gt in gg.groupby("task"):
            c = gt.groupby("step")["seed"].nunique()
            com = c[c >= 3].index
            if len(com)==0: continue
            v = gt[gt.step==com.max()].groupby("seed")["bits_per_byte_corr"].mean()
            vals.append(v)
        if len(vals)==len(TASKS10):
            M = pd.concat(vals, axis=1).dropna()
            if M.shape[0]>=3:
                rec_bpb[mix] = (M.mean(axis=1).mean(), M.mean(axis=1).std(ddof=1))
    if len(rec_bpb)>=10:
        means = np.array([v[0] for v in rec_bpb.values()]); sds = np.array([v[1] for v in rec_bpb.values()])
        S2 = means.var(ddof=1); nr = np.sqrt((sds**2).mean())
        sd2 = max(S2-(sds**2).mean()/3, 0.0)
        OUT["bpb"][sz] = {"n_recipes":len(rec_bpb),"signal_deconv":float(np.sqrt(sd2)),
                          "noise_rms":float(nr),"noise_median":float(np.median(sds)),
                          "snr_deconv":float(np.sqrt(sd2)/nr)}
    else:
        OUT["bpb"][sz] = None

# decomposition of the 60M->90M jump (accuracy readout), on both noise aggregates:
# the deconvolved SNR uses the RMS noise; Table 1's displayed noise column is the median.
a = OUT["accuracy"]
jump = {
    "dlog_signal": float(np.log(a["90M"]["signal_deconv"]/a["60M"]["signal_deconv"])),
    "dlog_noise_rms": float(-np.log(a["90M"]["noise_rms"]/a["60M"]["noise_rms"])),
    "dlog_noise_median": float(-np.log(a["90M"]["noise_median"]/a["60M"]["noise_median"])),
    "dlog_snr": float(np.log(a["90M"]["snr_deconv"]/a["60M"]["snr_deconv"])),
}
OUT["jump_60_to_90_accuracy"] = jump

# per-recipe noise at 20M/60M/90M: which recipes carry 60M's high noise?
per = {sz: OUT["accuracy"][sz]["per_recipe_sd"] for sz in SIZES}
tab = pd.DataFrame(per).dropna()
tab["ratio_60_vs_neigh"] = tab["60M"] / tab[["20M","90M"]].mean(axis=1)
OUT["noise_outliers_60M"] = tab.sort_values("ratio_60_vs_neigh", ascending=False).head(5).round(5).to_dict()

# grid density: checkpoints per (recipe) at each scale from t1b_cells
cells = pd.read_parquet(f"{R}/t1b_cells.parquet")
grid = cells[cells.task=="arc_challenge"].groupby("params")["grid_n"].median()
OUT["grid_n_median"] = {k: int(v) for k, v in grid.items()}

print("accuracy readout:")
for sz in SIZES:
    v = OUT["accuracy"][sz]
    print(f"  {sz:>5}: signal_dec={v['signal_deconv']:.5f} noise_rms={v['noise_rms']:.5f} noise_med={v['noise_median']:.5f} SNR={v['snr_deconv']:.2f}")
print("bpb readout:")
for sz in SIZES:
    v = OUT["bpb"].get(sz)
    if v is None:
        print(f"  {sz:>5}: n/a"); continue
    print(f"  {sz:>5}: signal_dec={v['signal_deconv']:.5f} noise_rms={v['noise_rms']:.5f} noise_med={v['noise_median']:.5f} SNR={v['snr_deconv']:.2f}")
print(f"jump decomposition: dlog SNR {jump['dlog_snr']:.3f} = dlog signal {jump['dlog_signal']:.3f} + (-dlog noise_rms) {jump['dlog_noise_rms']:.3f}; with median noise: {-jump['dlog_noise_median']:.3f}")
print("60M noise outliers (ratio vs mean(20M,90M)):", {k: v for k, v in list(OUT['noise_outliers_60M']['ratio_60_vs_neigh'].items())})
print("grid_n medians:", {k: OUT['grid_n_median'][k] for k in ['20M','60M','90M','150M'] if k in OUT['grid_n_median']})

with open(os.path.join(HERE, "r9_jump_decomposition.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote r9_jump_decomposition.json")
