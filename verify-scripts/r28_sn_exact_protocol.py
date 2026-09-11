#!/usr/bin/env python3
"""r28_sn_exact_protocol -- EXT M1: replicate S&N's Fig-7 result under their *exact* protocol,
then ablate each of our protocol deviations one at a time.

S&N Fig 7 (arXiv 2508.13144, caption verified against the PDF): 8 tasks (ARC-C, ARC-E,
Winogrande, CSQA, SocialIQA, HellaSwag, MMLU, PIQA); proxy x = Rel. Std. over the LAST 20
training checkpoints; target y = relSD across arm runs of the per-run mean of the LAST 20
checkpoints; R2 = 0.82 (init) / 0.86 (order) [as printed in text; the figure's R values are
0.90/0.93/0.97].

Our home protocol (01_t1_sn_replication.py) differs in three ways: 9 primary-like tasks,
last-30-checkpoint proxy window, last-3-checkpoint target smoothing. This script recomputes:
row A = exact Fig-7 protocol (8 tasks, n=20, last-20 target);
rows B-D = one deviation re-added at a time (9 tasks / n=30 / last-3 target);
row E = our full home protocol. All rows computed from the same released parquet.

Writes r28_sn_exact_protocol.json. Pure CPU, deterministic. NFT_SN overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SN = os.environ.get("NFT_SN", "")
if not SN or not os.path.exists(SN):
    for cand in [os.path.normpath(os.path.join(HERE, "..", "data", "analysis", "random_seeds.parquet")),
                 "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis/raw/random_seeds.parquet"]:
        if os.path.exists(cand):
            SN = cand; break

rs = pd.read_parquet(SN)

THEIR8 = ["arc_challenge", "arc_easy", "winogrande", "csqa", "socialiqa", "hellaswag",
          "mmlu", "piqa"]


def pick_metric(g):
    ms = set(g["metric"].unique())
    if "acc_per_char" in ms: return "acc_per_char"
    if "acc" in ms: return "acc"
    return None


def relsd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.mean(v) == 0: return np.nan
    return np.std(v, ddof=1) / abs(np.mean(v))


def compute(tasks, n_proxy, n_target, metric_mode="primary_like", step_final=None):
    """x = per-run relSD of last n_proxy ckpts (mean over arm runs);
       y = relSD across arm runs of per-run mean of last n_target ckpts.
       Returns per-arm (r_raw, r_log)."""
    out = {}
    for arm in ["seed", "data"]:
        xs, ys, used = [], [], []
        for t in tasks:
            g = rs[rs.task_name == t]
            m = pick_metric(g) if metric_mode == "primary_like" else "bits_per_byte"
            g = g[g.metric == m]
            runs = [gr for _, gr in g[g.run_type == arm].groupby("run_name")]
            if len(runs) < 5: continue
            x_r, y_r = [], []
            for gr in runs:
                gr = gr if step_final is None else gr[gr.step <= step_final]
                gr = gr.sort_values("step")
                v = gr["value"].to_numpy(float)
                if len(v) < max(n_proxy, n_target): continue
                x_r.append(relsd(v[-n_proxy:]))
                y_r.append(v[-n_target:].mean())
            if len(x_r) < 5: continue
            xs.append(np.nanmean(x_r))
            ys.append(relsd(y_r))
            used.append(t)
        xs, ys = np.array(xs), np.array(ys)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        r_raw = stats.pearsonr(xs, ys)[0]
        r_log = stats.pearsonr(np.log10(xs), np.log10(ys))[0]
        out[arm] = {"r_raw": round(float(r_raw), 4), "r_log": round(float(r_log), 4),
                    "R2_raw": round(float(r_raw**2), 4), "R2_log": round(float(r_log**2), 4),
                    "n_tasks": int(ok.sum())}
    return out


ours9 = None  # our 9 primary-like tasks, reproduced from the paper's table source
# the paper's nine = the 9 primary-like tasks of Table 8 (tab:t2): all tasks with a
# primary-like metric, i.e. their task set after the same pick_metric filter
all_tasks = sorted(rs.task_name.unique())
nine = [t for t in all_tasks if pick_metric(rs[rs.task_name == t]) and t not in
        ["copycolors", "gsm8k", "humaneval", "mbpp"] and not t.startswith("minerva")]
# (matches the 9 primary-like rows of Table 8: arc_challenge/arc_easy/csqa/hellaswag/mmlu/
#  openbookqa/piqa/socialiqa/winogrande + copycolors is bpb-only there... resolve from data)
# the paper's 9 primary-like tasks = the primary panel of Table 8 (tab:t2)
nine = ["hellaswag", "socialiqa", "mmlu", "arc_challenge", "csqa", "winogrande",
        "arc_easy", "piqa", "copycolors"]

rows = {}
rows["A_exact_fig7"] = compute(THEIR8, 20, 20)
rows["B_plus_9th_task"] = compute(THEIR8 + ["copycolors"], 20, 20)
rows["C_plus_n30_window"] = compute(THEIR8, 30, 20)
rows["D_plus_last3_target"] = compute(THEIR8, 20, 3)
rows["E_our_home_protocol"] = compute(nine, 30, 3)
rows["E2_ours_plus_common_final69000"] = compute(nine, 30, 3, step_final=69000)
rows["A2_exact_plus_truncation69000"] = compute(THEIR8, 20, 20, step_final=69000)

print(f"our 9 primary-like tasks: {nine}")
print(f"{'row':<24} {'arm':<6} {'R2_raw':>7} {'R2_log':>7}  n")
for k, v in rows.items():
    if v is None: continue
    for arm, r in v.items():
        print(f"{k:<24} {arm:<6} {r['R2_raw']:>7.3f} {r['R2_log']:>7.3f}  {r['n_tasks']}")

out = {"published_fig7": {"init_R2": 0.82, "order_R2": 0.86},
       "published_fig7_protocol": {"tasks": THEIR8, "n_proxy": 20, "n_target": 20},
       "our_protocol": {"tasks": nine, "n_proxy": 30, "n_target": 3},
       "rows": rows}
json.dump(out, open(os.path.join(HERE, "r28_sn_exact_protocol.json"), "w"), indent=1)
print("wrote r28_sn_exact_protocol.json")
