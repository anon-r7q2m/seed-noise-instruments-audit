#!/usr/bin/env python3
"""validate_pooled.py -- per-sim pooled parametric bootstrap for the two pooled
descriptive targets: macro pooled 90M-750M (59%) and task-level pooled 90-750M (82%),
plus task-level at 150M. Stores per-sim pooled values to validation_pooled.json."""
import sys, os, json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hme import CellFit, descriptive_stats, simulate_cell, true_pflip_share_of_errors

RES = "./work-20260825/rank4-paper/enhancement3/results"
MACRO = "olmes_10_macro_avg"
TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]
POOL_SIZES = ["90M", "150M", "300M", "530M", "750M"]
R = 200


def _sim_cell(job):
    """One sim replicate of one cell -> descriptive counts + in-sim truth."""
    key, seed = job
    z = np.load(f"{RES}/arrays.npz")
    fit = CellFit(z[key + "|X"], z[key + "|Y"])
    rng = np.random.default_rng(seed)
    X, Y, truth = simulate_cell(fit, rng)
    st = descriptive_stats(X, Y)
    t25, tm = true_pflip_share_of_errors(X, Y, truth)
    return dict(errors=st["errors"], err_un=st["err_any_unstable"],
                n_pairs=st["n_pairs"], t25=t25, tm=tm,
                unstable_pairs=st["share_pairs_unstable"] * st["n_pairs"])


def main():
    z = np.load(f"{RES}/arrays.npz")
    cells = [(MACRO, s) for s in POOL_SIZES] + \
            [(t, s) for s in POOL_SIZES for t in TASKS10]
    jobs, meta = [], []
    for (t, s) in cells:
        for r in range(R):
            jobs.append((f"{t}|{s}", hash((t, s, r)) % (2**31)))
            meta.append((t, s, r))
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_sim_cell, jobs))
    df = pd.DataFrame(meta, columns=["task", "size", "rep"])
    for k in results[0]:
        df[k] = [x[k] for x in results]

    def pooled(sub):
        err = sub.errors.sum()
        return dict(errors=int(err),
                    share_err_unstable=float(sub.err_un.sum() / err) if err else np.nan,
                    true_flip25_share=float(
                        (sub.t25 * sub.errors).sum() / err) if err else np.nan,
                    true_mean_pflip=float(
                        (sub.tm * sub.errors).sum() / err) if err else np.nan)

    out = {}
    # macro pooled 90-750M per rep
    for label, taskset, sizes in [
            ("macro_pooled_90_750", [MACRO], POOL_SIZES),
            ("tasklevel_pooled_90_750", TASKS10, POOL_SIZES),
            ("tasklevel_150M", TASKS10, ["150M"])]:
        reps = []
        for r in range(R):
            sub = df[(df.task.isin(taskset)) & (df["size"].isin(sizes)) & (df.rep == r)]
            reps.append(pooled(sub))
        for stat in ["share_err_unstable", "true_flip25_share", "true_mean_pflip", "errors"]:
            v = np.array([x[stat] for x in reps], dtype=float)
            out.setdefault(label, {})[stat] = dict(
                mean=float(np.nanmean(v)),
                q025=float(np.nanpercentile(v, 2.5)),
                q975=float(np.nanpercentile(v, 97.5)))
    with open(f"{RES}/validation_pooled.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
