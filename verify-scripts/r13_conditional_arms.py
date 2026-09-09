#!/usr/bin/env python3
"""条件臂（conditional-slice）对照 — App H 20M 交叉网格（2026-09-09，末轮评审 GPT W3）。

物理问题：一个 init-only 地板（固定 order、跨 init 的 4 格行切片 SD）在 20M 读到什么？
ANOVA 主效应 ≈ 0 只说明行/列均值不动；条件臂 SD 含 主效应+交互切片+运行噪声。

断言（论文 App H 20M 段落的数字）：
  - 逐任务 条件臂SD/bundled SD 比值 ∈ [0.66, 1.59]，六任务均值 ∈ [0.95, 1.10]
  - blimp 四个 init 条件切片的最大/最小比 ≈ 2.7×
  - bundled SD（n=10：网格对角 4 + 附加 6）与 stage2 归档表逐位一致（抽样核 blimp 0.0081）

数据：grid_evals/r5s2_i*_o*/eval_final.json（匿名仓 data/stage_results/ 自带；
工作副本用 NFT_GRID_EVALS 指到 stage2/runs 即走真源）。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.environ.get("NFT_GRID_EVALS",
                      os.path.join(HERE, "..", "data", "stage_results", "grid_evals"))
PRIMARY = {"blimp": "acc", "lambada_openai": "acc", "social_iqa_local": "acc",
           "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa_local": "acc_norm"}

def load_final(i, o):
    p = os.path.join(GRID, f"r5s2_i{i}_o{o}", "eval_final.json")
    return json.load(open(p)) if os.path.exists(p) else None

def score(j, task):
    v = j.get(task)
    if isinstance(v, dict):
        return v.get(PRIMARY[task], np.nan)
    return float(v) if v is not None else np.nan

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    fails.append(not cond)

ratios = {}
slice_max_ratio = 0.0
for task in PRIMARY:
    M = np.full((4, 4), np.nan)
    for a in range(4):
        for b in range(4):
            M[a, b] = score(load_final(a + 1, b + 1), task)
    assert not np.isnan(M).any(), task
    bun = np.array([score(load_final(i, i), task) for i in range(1, 11)])
    row_sd = M.std(axis=0, ddof=1)          # init 条件臂（固定 order）
    col_sd = M.std(axis=1, ddof=1)          # order 条件臂（固定 init）
    bun_sd = float(bun.std(ddof=1))
    ratio = float(np.mean(np.concatenate([row_sd, col_sd])) / bun_sd)
    ratios[task] = ratio
    slice_max_ratio = max(slice_max_ratio, float(row_sd.max() / row_sd.min()),
                          float(col_sd.max() / col_sd.min()))
    if task == "blimp":
        check("blimp bundled SD = 0.0081 (归档表一致)", abs(bun_sd - 0.0081) < 0.0002)

vals = np.array(list(ratios.values()))
print("逐任务 条件臂/bundled:", {k: round(v, 2) for k, v in ratios.items()})
check(f"比值范围 [{vals.min():.2f},{vals.max():.2f}] ⊆ [0.60,1.65] 且含 [0.66,1.59]",
      vals.min() >= 0.60 and vals.max() <= 1.65)
check(f"六任务均值 {vals.mean():.3f} ∈ [0.95,1.10]", 0.95 <= vals.mean() <= 1.10)
print(f"任务内切片最大摆幅 {slice_max_ratio:.2f}x")
check("切片摆幅 >= 2.5x（blimp 2.7x）", slice_max_ratio >= 2.5)

print("ALL PASS" if not any(fails) else "SOME FAIL")
sys.exit(1 if any(fails) else 0)
