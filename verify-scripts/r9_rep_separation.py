#!/usr/bin/env python3
"""Package-side regeneration of stage2_rep_separation.json from the shipped per-cell
eval_final.json files (rep_evals/): the five replicated 20M cells' run-level differences
and the run-noise-corrected interaction bounds. See stage2/stage2_rep_analysis.py for the
full protocol (PREREG-r5-stage2; revision-9 review response)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ST2 = os.environ.get("NFT_STAGE2", os.path.join(HERE, "..", "data", "stage_results"))
EVAL = os.path.join(ST2, "rep_evals")

PRIMARY = {"blimp": "acc", "lambada_openai": "acc", "social_iqa_local": "acc",
           "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa_local": "acc_norm"}
REPS = {"rep1": ("r5s2_rep1_i1_o1", 1, 1), "rep2": ("r5s2_rep2_i2_o2", 2, 2),
        "rep3": ("r9s2_rep3_i3_o3", 3, 3), "rep4": ("r9s2_rep4_i4_o4", 4, 4),
        "rep5": ("r9s2_rep5_i2_o3", 2, 3)}

def score(j, task):
    v = j.get(task)
    if v is None: return np.nan
    return float(v.get(PRIMARY[task], np.nan)) if isinstance(v, dict) else float(v)

cells = {}
for name, (rd, i, o) in REPS.items():
    orig = json.load(open(os.path.join(EVAL, f"r5s2_i{i}_o{o}", "eval_final.json")))
    rep = json.load(open(os.path.join(EVAL, rd, "eval_final.json")))
    cells[f"({i},{o})"] = {t: {"orig": score(orig, t), "rep": score(rep, t),
                               "absdiff": abs(score(orig, t) - score(rep, t))} for t in PRIMARY}

grid = json.load(open(os.path.join(ST2, "stage2_analysis.json")))["grid"]
per_task = {}
for t in PRIMARY:
    diffs = np.array([c[t]["absdiff"] for c in cells.values() if not np.isnan(c[t]["absdiff"])])
    sig2_run = float((diffs**2 / 2).mean())
    resid = grid[t]["sd_int"]
    new2 = max(resid**2 - sig2_run, 0.0)
    per_task[t] = {"resid_sd": resid, "run_sd": float(np.sqrt(sig2_run)),
                   "int_bound_old": resid, "int_bound_new": float(np.sqrt(new2)),
                   "n_rep_cells": int(len(diffs))}

out = {"cells": cells, "per_task": per_task}
with open(os.path.join(HERE, "..", "data", "stage_results", "stage2_rep_separation.json"), "w") as f:
    json.dump(out, f, indent=1)

diffs_all = [c[t]["absdiff"] for c in cells.values() for t in c]
ok = (len(cells) == 5
      and abs(per_task["blimp"]["int_bound_new"] - 0.0101) < 0.001
      and per_task["social_iqa_local"]["int_bound_new"] == 0.0
      and per_task["arc_challenge"]["int_bound_new"] == 0.0
      and abs(float(np.median(diffs_all)) - 0.0046) < 0.001
      and abs(max(diffs_all) - 0.0231) < 0.001)
print("PASS rep separation regenerates" if ok else "FAIL rep separation")
for t in PRIMARY:
    p = per_task[t]
    print(f"  {t:<16} resid {p['resid_sd']:.4f} run {p['run_sd']:.4f} int<= {p['int_bound_new']:.4f}")
sys.exit(0 if ok else 1)
