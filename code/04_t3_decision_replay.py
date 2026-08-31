#!/usr/bin/env python3
"""T3 — replay DataDecide's recipe-pair decisions; estimate the seed-noise share of errors.

Published claim (2504.11393 abstract): ranking at a single small size (e.g. 150M)
predicts best-at-1B with ~80% of comparisons correct.

Replay on the released macro table:
  target(recipe)  = 1B score: per-seed mean of last-3 common ckpts, averaged over 3 seeds
  pred_s(recipe)  = small-size score at final common ckpt (per seed; and 3-seed mean)
  decision for pair (i,j): sign(pred_i - pred_j) vs sign(target_i - target_j)

Error decomposition per size (macro + per task):
  seed-unstable pair (small side): the 3 single-seed decisions do not all agree
  target-unstable pair: the 3 single-seed 1B orderings do not all agree
  error share attributable to seed noise = P(unstable | wrong) variants.
"""
import pandas as pd, numpy as np, itertools, os
OUT = "data"

df = pd.read_parquet(f"{OUT}/analysis/dd_tidy.parquet")
TASKS = ["olmes_10_macro_avg","arc_challenge","arc_easy","boolq","csqa","hellaswag",
         "mmlu","openbookqa","piqa","socialiqa","winogrande"]
SIZES = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M"]
METRIC = "primary_metric"

def scores_at(size, task, mode="final", avg_last=1):
    """return DataFrame recipe x seed of scores; mode final: last common step (avg_last ckpts)"""
    g = df[(df.params==size) & (df.task==task)]
    out = {}
    for d_, gd in g.groupby("data"):
        seeds = sorted(gd.seed.unique())
        step_sets = [set(gd[gd.seed==s].step) for s in seeds]
        common = sorted(set.intersection(*step_sets))
        if len(common) < avg_last: continue
        lw = common[-avg_last:]
        out[d_] = {s: gd[(gd.seed==s) & (gd.step.isin(lw))][METRIC].mean() for s in seeds}
    return pd.DataFrame(out).T  # index recipe, columns seeds

rows = []
pairrows = []
for task in TASKS:
    tgt = scores_at("1B", task, avg_last=3)          # 25 x 3 seeds
    tgt_mean = tgt.mean(axis=1)
    # target per-seed orderings for stability
    for size in SIZES:
        pred = scores_at(size, task, avg_last=1)
        recipes = sorted(set(pred.index) & set(tgt.index))
        n_ok = w_ok = 0
        n_pairs = 0
        wrong_unstable_small = wrong_unstable_tgt = wrong_unstable_any = 0
        stable_wrong = 0
        acc_single = []
        for i, j in itertools.combinations(recipes, 2):
            dt = tgt_mean[i] - tgt_mean[j]
            if dt == 0: continue
            n_pairs += 1
            dm = pred.loc[i].mean() - pred.loc[j].mean()
            correct = (dm * dt) > 0
            # small-side seed stability (per-seed decisions)
            ss = np.sign([pred.loc[i, s] - pred.loc[j, s] for s in pred.columns if pd.notna(pred.loc[i, s]) and pd.notna(pred.loc[j, s])])
            small_unstable = len(set(ss)) > 1
            # target-side stability
            ts = np.sign([tgt.loc[i, s] - tgt.loc[j, s] for s in tgt.columns])
            tgt_unstable = len(set(ts)) > 1
            # single-seed accuracy (average over seeds)
            acc_single.append(np.mean([ (v*dt)>0 for v in (pred.loc[i]-pred.loc[j]).dropna() ]))
            if correct:
                n_ok += 1
            else:
                if small_unstable: wrong_unstable_small += 1
                if tgt_unstable: wrong_unstable_tgt += 1
                if small_unstable or tgt_unstable: wrong_unstable_any += 1
                else: stable_wrong += 1
            pairrows.append(dict(task=task, size=size, i=i, j=j, correct=bool(correct),
                                 small_unstable=bool(small_unstable), tgt_unstable=bool(tgt_unstable),
                                 abs_dt=abs(dt), abs_dm=abs(dm)))
        n_err = n_pairs - n_ok
        rows.append(dict(task=task, size=size, n_pairs=n_pairs,
                         acc_seedmean=n_ok/n_pairs, acc_singleseed=np.mean(acc_single),
                         err=n_err,
                         err_small_unstable=wrong_unstable_small,
                         err_tgt_unstable=wrong_unstable_tgt,
                         err_any_unstable=wrong_unstable_any,
                         err_stable=stable_wrong,
                         share_err_any_unstable=wrong_unstable_any/n_err if n_err else np.nan,
                         share_err_small_unstable=wrong_unstable_small/n_err if n_err else np.nan))

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/tables/t3_decision_replay.csv", index=False)
P = pd.DataFrame(pairrows)
P.to_parquet(f"{OUT}/analysis/t3_pairs.parquet", index=False)

macro = R[R.task=="olmes_10_macro_avg"]
print("== macro (olmes_10_macro_avg) ==")
print(macro[["size","n_pairs","acc_seedmean","acc_singleseed","err",
             "err_small_unstable","err_tgt_unstable","err_stable",
             "share_err_any_unstable"]].round(3).to_string(index=False))

# aggregate over 10 individual tasks (not macro)
ind = R[R.task!="olmes_10_macro_avg"].groupby("size").apply(
    lambda s: pd.Series(dict(
        acc_seedmean=s.acc_seedmean.mean(), acc_singleseed=s.acc_singleseed.mean(),
        share_err_any_unstable=(s.err_any_unstable.sum()/s.err.sum()) if s.err.sum() else np.nan,
        share_err_small_unstable=(s.err_small_unstable.sum()/s.err.sum()) if s.err.sum() else np.nan,
        err=s.err.sum(), n_pairs=s.n_pairs.sum())), include_groups=False).reset_index()
ind["size_order"] = ind["size"].map({s:i for i,s in enumerate(SIZES)})
ind = ind.sort_values("size_order").drop(columns="size_order")
ind.to_csv(f"{OUT}/tables/t3_by_size_tasklevel.csv", index=False)
print("\n== average over 10 OLMES tasks ==")
print(ind.round(3).to_string(index=False))
