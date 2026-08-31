#!/usr/bin/env python3
"""Parse PolyPythias eval JSONs (big-band multitask + small-band single-task) into tidy table.

Files: pp_json/pythia-{size}-seed{k}/step{S}/{model_dir}/results_{timestamp}.json
lm-eval-harness results: d['results'][task] -> {'acc,none':..., 'perplexity,none':..., ...}
Dedupe rule: for (run,step,task,metric) keep the LATEST timestamp.
Output: analysis/pp_tidy.parquet (size, seed, step, task, metric, value, ts)
"""
import json, glob, os, re
import pandas as pd, numpy as np

TMP = "./tmp/rank04-zerogpu"
OUT = "data"

KEEP_METRICS = {"acc,none": "acc", "acc_norm,none": "acc_norm",
                "perplexity,none": "perplexity", "likelihood_diff,none": "likelihood_diff"}
SKIP_TASK_PREFIX = ("blimp_", "crows_pairs_", "winogender_")  # keep only aggregates for these

rows = []
files = glob.glob(f"{TMP}/pp_json/pythia-*/step*/**/results_*.json", recursive=True)
print("files to parse:", len(files))
pat = re.compile(r"pythia-(\d+m)-seed(\d+)/step(\d+)/([^/]+)/")
mismatch = 0
for fp in files:
    m = pat.search(fp)
    if not m: continue
    size, seed, step, model_dir = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    # integrity check: eval must have been run on the seed model the directory claims.
    # (release bug: e.g. pythia-410m-seed{1..9} lambada files contain evals of the BASE
    #  EleutherAI/pythia-410m -> identical scores across "seeds")
    expect = f"EleutherAI__pythia-{size}" if seed == 0 else f"EleutherAI__pythia-{size}-seed{seed}"
    if model_dir != expect:
        mismatch += 1
        continue
    ts = fp.rsplit("results_", 1)[-1].replace(".json", "")
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    res = d.get("results", {})
    for task, md in res.items():
        if task.startswith(SKIP_TASK_PREFIX): continue
        for k, name in KEEP_METRICS.items():
            if k in md and md[k] is not None:
                rows.append((size, seed, step, task, name, float(md[k]), ts))

df = pd.DataFrame(rows, columns=["size","seed","step","task","metric","value","ts"])
print("raw rows:", len(df), " | model-mismatch files skipped:", mismatch)
df = df.sort_values("ts").drop_duplicates(subset=["size","seed","step","task","metric"], keep="last")
print("dedup rows:", len(df))
df.to_parquet(f"{OUT}/analysis/pp_tidy.parquet", index=False)
print(df.groupby("task").size().sort_values(ascending=False).head(20))
print("\ncoverage (task=lambada_openai, metric=acc): runs x final step")
lam = df[(df.task=="lambada_openai") & (df.metric=="acc")]
cov = lam.groupby(["size","seed"])["step"].agg(["max","nunique"])
print(cov.to_string())
