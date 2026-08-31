#!/usr/bin/env python3
"""make_tables.py -- assemble paper-ready tables from results/ (after bootstrap)."""
import json
import numpy as np
import pandas as pd

B = "./work-20260825/rank4-paper/enhancement3"
MACRO = "olmes_10_macro_avg"
POOL = ["90M", "150M", "300M", "530M", "750M"]
TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]

ce = pd.read_csv(f"{B}/results/cell_estimates.csv")
obs = pd.read_csv(f"{B}/results/observed_descriptive.csv")
boot = pd.read_csv(f"{B}/results/bootstrap_ci.csv")
bp = pd.read_csv(f"{B}/results/bootstrap_pooled_ci.csv")
pe = pd.read_parquet(f"{B}/results/pair_estimates.parquet")


def ci(task, size, est):
    r = boot[(boot.task == task) & (boot["size"] == size) & (boot.estimand == est)]
    if len(r) == 0:
        return (np.nan, np.nan)
    return (r.iloc[0].lo, r.iloc[0].hi)


def pci(size, est, cluster="recipe"):
    r = bp[(bp["size"] == size) & (bp.estimand == est) & (bp.cluster == cluster)]
    if len(r) == 0:
        return (np.nan, np.nan)
    return (r.iloc[0].lo, r.iloc[0].hi)


def fmt(x, pct=False, d=1):
    if isinstance(x, float) and not np.isfinite(x):
        return "--"
    return f"{x*100:.{d}f}" if pct else f"{x:.{d}f}"


def fmt_ci(lo, hi, pct=False, d=1):
    if not np.isfinite(lo):
        return "[--]"
    if pct:
        return f"[{lo*100:.{d}f}, {hi*100:.{d}f}]"
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


# ---- Table A: macro cells, model estimates + recipe-cluster CIs
rows = []
for size in POOL:
    c = ce[(ce.task == MACRO) & (ce["size"] == size)].iloc[0]
    o = obs[(obs.task == MACRO) & (obs["size"] == size)].iloc[0]
    rows.append({
        "size": size,
        "obs_err": int(o.errors),
        "exp_err": c.exp_errors,
        "exp_err_ci": ci(MACRO, size, "exp_errors"),
        "noise_share": c.exp_errors_noise / c.exp_errors,
        "noise_share_ci": ci(MACRO, size, "noise_share"),
        "mean_pflip": c.mean_p_flip,
        "mean_pflip_ci": ci(MACRO, size, "mean_p_flip"),
        "share_pflip25": c.share_pflip_gt25,
        "er_share": c.er_real_share,
        "er_share_ci": ci(MACRO, size, "er_real_share"),
        "pick": c.er_pick_real,
        "pick_ci": ci(MACRO, size, "er_pick_real"),
        "pick_latent": c.er_pick_latent,
        "rho": c.rho,
        "obs_share_unstable": o.share_err_any_unstable,
    })
A = pd.DataFrame(rows)
print("== Table A: macro (olmes_10_macro_avg) model estimates ==")
for r in rows:
    print(f"{r['size']:>5} | err {r['obs_err']:3d} obs / {r['exp_err']:5.1f} mod {fmt_ci(*r['exp_err_ci'],d=0)} "
          f"| noise-share {fmt(r['noise_share'],1)}% {fmt_ci(*r['noise_share_ci'],1)} | p_flip {fmt(r['mean_pflip'],1)}% {fmt_ci(*r['mean_pflip_ci'],1)} "
          f"| >25%: {fmt(r['share_pflip25'],True,1)}% | ER {fmt(r['er_share'],1)}% {fmt_ci(*r['er_share_ci'],1)} "
          f"| pick {r['pick']:.4f} {fmt_ci(*r['pick_ci'],d=4)} (latent {r['pick_latent']:.4f}) | rho {r['rho']:.2f}")

# ---- Table B: task-level pooled per size (point + pooled CIs)
print("\n== Table B: task-level pooled (10 tasks) ==")
for size in POOL:
    sub = pe[(pe.task != MACRO) & (pe["size"] == size)]
    o = obs[(obs.task != MACRO) & (obs["size"] == size)]
    err = sub.p_err.sum()
    line = dict(
        size=size,
        obs_err=int(o.errors.sum()),
        exp_err=err,
        exp_err_ci=pci(size, "exp_errors"),
        exp_err_ci_task=pci(size, "exp_errors", "task"),
        noise_share=sub.p_err_noise.sum() / err,
        noise_share_ci=pci(size, "noise_share"),
        mean_pflip=sub.p_flip.mean(),
        mean_pflip_ci=pci(size, "mean_p_flip"),
        nas_w=(sub.p_err * sub.p_flip).sum() / err,
        nas_w_ci=pci(size, "nas_weighted"),
        er_share=sub.er_real.sum() / sub.e_abs_dY.sum(),
        er_share_ci=pci(size, "er_real_share"),
        obs_share_unstable=o.err_any_unstable.sum() / o.errors.sum(),
    )
    print(f"{size:>5} | err {line['obs_err']:4d} obs / {line['exp_err']:6.1f} mod "
          f"{fmt_ci(*line['exp_err_ci'],d=0)}recipe {fmt_ci(*line['exp_err_ci_task'],d=0)}task "
          f"| noise-share {fmt(line['noise_share'],1)}% {fmt_ci(*line['noise_share_ci'],1)} | p_flip {fmt(line['mean_pflip'],1)}% "
          f"{fmt_ci(*line['mean_pflip_ci'],1)} | nas_w {fmt(line['nas_w'],1)}% {fmt_ci(*line['nas_w_ci'],1)} "
          f"| ER {fmt(line['er_share'],1)}% {fmt_ci(*line['er_share_ci'],1)} "
          f"| obs unstable-share {fmt(line['obs_share_unstable'],1)}%")

# ---- Table C: validation summary
val = json.load(open(f"{B}/results/validation.json"))
vp = json.load(open(f"{B}/results/validation_pooled.json"))
print("\n== Table C: parametric-bootstrap validation ==")
print("cell-level (observed vs simulated descriptive share, and in-sim TRUE flippable share):")
for key, v in val.items():
    o, s = v["observed"], v["simulated"]
    print(f"{key:>28} | err {o['errors']:3d} vs {s['errors_mean']:5.1f} [{s['errors_q025']:.0f},{s['errors_q975']:.0f}] "
          f"| share {o['share_err_any_unstable']*100:.1f}% vs {s['share_err_unstable_mean']*100:.1f}% "
          f"[{s['share_err_unstable_q025']*100:.0f},{s['share_err_unstable_q975']*100:.0f}] "
          f"| TRUE flip25 {s['true_flippable25_share_of_errors_mean']*100:.1f}% "
          f"[{s['true_flippable25_share_of_errors_q025']*100:.0f},{s['true_flippable25_share_of_errors_q975']*100:.0f}] "
          f"| true mean pflip {s['true_mean_pflip_of_errors']*100:.1f}%")
print("pooled:")
for label, d in vp.items():
    print(f"{label:>26} | sim classifier share {d['share_err_unstable']['mean']*100:.1f}% "
          f"[{d['share_err_unstable']['q025']*100:.1f},{d['share_err_unstable']['q975']*100:.1f}] "
          f"| TRUE flip25 {d['true_flip25_share']['mean']*100:.1f}% "
          f"[{d['true_flip25_share']['q025']*100:.1f},{d['true_flip25_share']['q975']*100:.1f}] "
          f"| true mean pflip {d['true_mean_pflip']['mean']*100:.1f}% "
          f"| sim errors {d['errors']['mean']:.0f} [{d['errors']['q025']:.0f},{d['errors']['q975']:.0f}]")
print("observed pooled: macro 58.6% (137/234) | task 90-750 82.4% (2475/3003) | task 150M 76.4% (567/742)")
