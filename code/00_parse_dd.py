#!/usr/bin/env python3
"""Parse DataDecide macro_avg parquet into tidy long format.

Input : tmp/rank04-zerogpu/data/dd_macro_avg.parquet  (235,125 rows; metrics as JSON string)
Output: analysis/dd_tidy.parquet  with columns:
        params, data, task, step, seed, tokens, compute,
        primary_metric, acc_raw, acc_per_char, correct_prob_per_char, bpb  (float)
"""
import json, pandas as pd, numpy as np, os

TMP = "./tmp/rank04-zerogpu"
OUT = "data/analysis"

df = pd.read_parquet(f"{TMP}/data/dd_macro_avg.parquet")
print("rows:", len(df))

KEYS = ["primary_metric", "acc_raw", "acc_per_char", "correct_prob_per_char",
        "bits_per_byte_corr", "logits_per_char_corr", "norm_correct_prob_per_char"]

def parse_metrics(s):
    try:
        m = json.loads(s)
    except Exception:
        import ast
        m = ast.literal_eval(s)
    return [m.get(k, np.nan) for k in KEYS]

vals = np.array([parse_metrics(s) for s in df["metrics"].to_numpy()], dtype=float)
for i, k in enumerate(KEYS):
    df[k] = vals[:, i]
df = df.drop(columns=["metrics"])
df["step"] = df["step"].astype(int)
os.makedirs(OUT, exist_ok=True)
df.to_parquet(f"{OUT}/dd_tidy.parquet", index=False)
print("saved", f"{OUT}/dd_tidy.parquet", df.shape)
print(df.groupby("task").size())
print("\nsizes x seeds sanity:")
print(df[df.task=="olmes_10_macro_avg"].groupby(["params","seed"])["step"].nunique().unstack())
