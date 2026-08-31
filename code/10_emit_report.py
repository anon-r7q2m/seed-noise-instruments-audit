#!/usr/bin/env python3
"""Emit the M2 deliverable REPORT.md (required artifact at runs/exp-rank04/zero-gpu/REPORT.md)."""
import pathlib

REPORT = r"""# rank-04 零 GPU 段报告 — 种子噪声三靶审计（PI 终审 M2 交付）

- **日期**：2026-08-13（M2 要求"今天开跑"——数据拉取、三靶分析、本报告均于当日完成）
- **执行**：the analysis pipeline（登录节点纯 CPU；零 GPU、零训练）
- **审计对象（M1 预注册的三条已发表主张）**：
  - **T1** Signal & Noise（arXiv 2508.13144，AI2）："训练末段 checkpoint 间噪声与 seed 噪声高相关（R²=0.82/0.86/0.95，即 R≥0.9），最后 n 个 ckpt 的相对标准差可当 seed 重训的廉价代理。"
  - **T2** 2605.20798（腾讯）隐含假设："仅动 init 的 3-seed 噪声地板 ≈ 全部随机源的 run 间噪声"（附录与表题自相矛盾处）；对应可裁决问题 = init-only vs data-order-only 的源等效性。
  - **T3** DataDecide（arXiv 2504.11393，AI2）："单一小尺度（150M）排序预测 1B 最优数据配方 ~80% 配对正确"——其余 20% 错误里 seed 噪声占多少，36 篇施引无人计算。
- **数据资产**（全部公开、别人已花的 FLOPs；清单与瑕疵详见 `analysis/DATA_MANIFEST.md`）：
  DataDecide 141 万行逐 seed 逐 step 评测表 + macro 表 + 11 域 ppl 表；
  Signal&Noise 官方 `random_seeds`（20 个 1B-5xC：10 init 臂 + 9 data-order 臂 + 1 密评 run）；
  PolyPythias 50 run（5 尺寸 × 10 seed）公开 lm-eval JSON（blimp/arc_challenge/lambada/…）。
- **代码**：`code/00–09_*.py`（pandas/numpy/scipy，conda env `tanh`）；表格 `tables/`，图 `figs/`。

---

## 资产瑕疵声明（分析前置约束，全部如实进入方法设计）

1. **DataDecide 大尺寸截断（预注册瑕疵，实测口径）**：发布 eval/macro 表已按 3 seed 共同 step 网格对齐；
   750M 的网格止于 ~31B token（名义预算 75B 的 **41%**）——750M 的"终态"语义按 41% 预算解释，单列不并入跨尺度外推；
   1B 表内 3 seed（default+large aux 2/3）覆盖满 100B token（另有 2 个 small-aux 1B run 在 25% 步数截断，未进 eval 表，仅见于 ppl 表）；≤530M 全部满预算。
2. **DataDecide 的 seed 是捆绑源**（init+data order 同变）：只能给"全源合计"噪声，不能做源分解；源分解仅能来自 S&N 两臂。
3. **S&N 两臂步长不齐**（init 臂评至 81k、order 臂 71.5k）：统一截取 step≤69000（≈5xC 终点），终分=各 run 末 3 ckpt 均值。
4. **PolyPythias 公开评测覆盖崎岖**：仅 blimp（5 尺寸）、arc_challenge（14m/31m）、lambada、math、bias 类；无 Pythia 全 8 任务；410m-seed6 轨迹止于 step 70k；seed0（原版 run）多数任务缺终点评测。
5. **【本审计新发现的上游数据 bug】PolyPythias evals 的 lambada 08-15 批次张冠李戴**：存于 `pythia-410m-seed{1..9}/` 目录下的 lambada 结果，`config.model_args` 显示实际评测的是基线 `EleutherAI/pythia-410m`（非 seed 变体）——9 个"seed"终点分数逐位相同、SD=0。共 **1095 个文件**被本管线的模型一致性校验剔除（`07_parse_polypythias.py`）。**天真使用该数据集会得出"410m seed 方差=0"的荒谬结论。** 此发现本身即是"社区噪声仪器未校准"论点的实证案例。
6. `/tmp` 满、HF 官方 CDN 限速等工程绕道见 DATA_MANIFEST；对结果无影响。

---

## T1：S&N "末段 step 噪声 ≈ seed 噪声" 代理检验

### 方法
- **A. 原设置复现（校准仪器）**：用 S&N 自家 `random_seeds` 数据复算其 A.3.1 的相关：
  x = 每 run 最后 30 个 ckpt（500-step 网格）分数的相对 SD（10 个 init-run 平均；或密评 run 末 300 点），
  y = 臂内 run 间终分相对 SD；跨任务 Pearson（raw 与 log 空间）。
- **B. 跨尺度跨配方移植（DataDecide）**：对 14 尺寸 × 25 配方，任务 = OLMES 10 基准（primary metric）：
  x_b = default seed 末 n 个网格 ckpt 相对 SD（n=min(5,⌊网格/2⌋)≥3），y_b = 3 seed 终点相对 SD。
  每格跨 10 任务求 R + **Monte-Carlo 天花板**（若代理完美，3-seed χ²(2) 与 n-ckpt χ² 采样误差下 R 的可达分布）与置换 null；
  另报**校准比率** y/x（代理作"量级替身"的直接检验）。
- **C. 连续指标（11 域 ppl，log 空间）**：同 B；另加**去趋势**变体（末窗线性拟合残差 SD）分离"趋势污染"。
- **D. PolyPythias 交叉复验**：10-seed 真值 y vs 末窗（≥0.72×143k）step 噪声 x。

### 结果
| 检验 | 数字 |
|---|---|
| A 复现（1B、单配方、10+9 run） | R²(init)=**0.81**（log）/0.99（raw）；R²(order)=**0.88**（log）/0.98（raw）——与其发表值 0.82/0.86 一致；比率 y/x 中位 **0.98**（IQR 0.77–1.12）→ 原主张在原设置下成立且量化校准 |
| B 相关形式（350 格） | 每格 10 任务的 R_log 中位 **0.53–0.77**（各尺寸），无一尺寸中位 ≥0.9；但 MC 天花板中位仅 **0.66–0.82**——3 seed 下"R≥0.9"**不可检验**；60–96% 的格与"代理完美"相容，16–48% 与 null 不可分（`tables/t1b_summary_by_size.csv`） |
| B 校准形式（3497 格） | y/x 中位 **1.47**（全尺寸池），仅 **53.9%** 的格落在 2× 带内；y>x 占 68.8%。1B 中位 **1.43 ≈ √2**——与"DataDecide seed 捆绑两源、S&N 代理只标定单源"的解释定量吻合（见 T2）；6M/60M/150M/300M 达 1.8–2.2 |
| C 连续指标（原样代理） | y/x 中位 **0.14–0.89**（各尺寸全部 <1）：粗网格上末段 ppl 仍在下降，**代理被趋势污染、反向高估 seed 噪声 2–7×** |
| C 去趋势修复 | 中位比率恢复至 750M/1B ≈ **1.03–1.08**；但 ≤530M 仍散布 0.54–2.8，仅 26–64% 格进 2× 带 |
| D PolyPythias（10-seed 真值） | 比率中位 **1.80**（IQR 1.22–2.55，9 格）；arc_challenge 1.15–1.80，blimp 在 70m/160m 达 **4.2–5.2**（任务饱和段 step 噪声塌缩、seed 噪声不塌缩） |

图：`figs/T1_proxy.png`（R 分布+天花板；比率标定；ppl 原样 vs 去趋势）。

### 判定：**部分成立**
原论文设置（1B、单配方、密评网格、10-run 臂、跨任务相关）**复现成立**。但作为其宣传用途——"小尺度实验里免 seed 重训的廉价代理"——**不可移植**：
(i) "R≥0.9" 在现实 seed 预算（3）与公开 ckpt 网格粒度下原则上不可复核（估计误差天花板 0.66–0.82）；
(ii) 量级校准系统性失真且**方向随指标类型翻转**（accuracy：低估 1.4–2.2×，√2 部分归因于源捆绑；ppl 粗网格：高估 2–7×，去趋势后仅大尺度恢复）；
(iii) 独立套件（PolyPythias，10-seed 真值）复验低估中位 1.8×，任务饱和段最差 5×。
**使用许可证**：密评网格 + 去趋势 + 大尺度（≥750M）+ 单源语义下可当 2× 以内的量级代理；其余场景须实测 seed。

---

## T2：源等效性 — init-only vs data-order-only

### 方法
S&N `random_seeds`：10 个 init 臂 run vs 9 个 data-order 臂 run（1B-5xC、同配方）。
每任务（primary-like 9 任务 / bpb 18 任务）：σ_init、σ_order = 臂内终分（末 3 ckpt 均值）相对 SD；
比率 CI 用 F 分布（df 9,8）+ run 级 bootstrap（2000 次）双报；等效带预设 [0.5, 2]（界报告，不做"不显著=等效"）。
交叉核对：DataDecide 1B 捆绑 3-seed 噪声（25 配方中位）vs sqrt(σ_init²+σ_order²) 与 σ_init。

### 结果
| 量 | primary（9 任务） | bpb（18 任务） |
|---|---|---|
| σ_init/σ_order 中位 | **0.79**（IQR 0.74–1.02） | **1.17**（IQR 0.89–1.34） |
| init 方差份额中位 | **0.38** | **0.58** |
| CI 排除 1 的任务 | 1/9（hellaswag **0.35**，CI 0.17–0.70：数据序噪声≈3×init） | 1/18（winogrande **2.36**，CI 1.13–4.79：init≈2.4×order） |
| CI 完整落入 2× 等效带 | **0/9** | **0/18** |
| init-only 对捆绑 σ 的低估 | 中位 **×1.62**（IQR 1.40–1.68） | 中位 **×1.32**（IQR 1.25–1.50） |

交叉核对（`tables/t2_dd1b_vs_sn_bundle.csv`）：DataDecide-1B 捆绑噪声中位 ≈ **1.06×σ_init** ≈ **0.66×sqrt(σ_init²+σ_order²)**（跨套件、不同语料/架构，粗对比；3-seed SD 的 c₄ 低偏 ~0.89 解释其中一部分）。

图：`figs/T2_source_equivalence.png`。

### 判定：**部分成立**（等效性本身不可裁决；init-only 近似被定量修正）
- "源等效"作为普适命题：**既不能证实也不能证伪**——即使拿全公开领域最富的两臂资产（10v9 个 1B run），没有任何任务的 CI 能塞进 2× 等效带（这正是反方对本项目原 pilot"n=4–8 谈等效"批评的领域级重演：等效性验证需要的 run 数远超现存公开资产）。
- 点估计层面两源同数量级、但**方向按任务翻转**（hellaswag 数据序主导 3×；winogrande init 主导 2.4×）——"任一单源≈全源"（2103.04514 的 CV 结论）在 decoder 预训练上**不是任务级安全假设**。
- 对 2605.20798 的可执行修正：**init-only 3-seed 地板系统性低估全源噪声，中位 ×1.3–1.6**；把它当社区门槛会把过多"噪声内差值"放行为"显著"。

---

## T3：DataDecide 80% 决策正确率中的 seed 噪声份额

### 方法
重放其配方对比决策（macro 表，OLMES 10 任务 + olmes_10_macro_avg，primary metric）：
目标 = 1B 每配方 3-seed（各取末 3 共同 ckpt 均值）平均；预测 = 小尺寸终点分数（3-seed 均值 与 单 seed 双轨）；
25 配方 → 300 对/任务/尺寸。**seed-不稳定 pair** = 小尺度侧 3 个单 seed 决策不一致（small）或 1B 侧 3 seed 排序不一致（tgt）。
错误分解 = 错误 ∩ 不稳定（任一侧）vs 错误 ∩ 双侧稳定。

### 结果
- **原主张复现**：150M macro 决策正确率 **82.7%**（3-seed 均值）/ **80.2%**（单 seed 平均）——与"~80%"精确吻合（`figs/T3_decisions.png` 左）。
- **错误的 seed 份额**（150M macro，52 个错误对）：**54%** 落在 seed-不稳定 pair（small 18 + tgt 16，去重 28）；90M–750M 合计 **59%**（137/234）。任务级（10 基准合池，90M–750M）：**82%**（3003 个错误中 2462 个不稳定；仅小尺度侧不稳定即占 65%）。
- **富集检验**：不稳定 pair 总基率 26–31%（macro），在错误中占 54–60% → **富集 ~2.0×**（P(不稳定|错) / P(不稳定)，90M/150M/300M = 1.88/2.10/2.06）——错误显著集中于噪声主导的对比。
- **3-seed 平均的直接收益**：90M–750M macro 上相对单 seed 消除 **8–24%** 的错误（150M：12.4%）；小于 16M 时反而偶见负收益（近地板、排序无信息）。
- 3-seed 一致 ≠ 真稳定（p_flip≈0.25 的 pair 有 ~58% 概率 3-seed 全同向），故上述份额是**下界**。

### 判定：**成立（附新注脚）**
"~80% 配对正确"逐字复现（82.7% @150M macro）。但其"错误 20%"的语义须改写：**其中过半（macro 54–59%；任务级 ~82%）不是稳定的排序错误，而是 seed 重抽即翻转的噪声对比**——即 DataDecide 声称的"20% 决策代价"里，可由多 seed 消除/标注的份额远大于其文本暗示；该数字（本审计首次给出）直接量化了"决策仪器未做噪声记账"的代价。

---

## PolyPythias 50-run 再分析（M2 指定资产）

- **终点 10-seed 噪声图集**（`tables/pp_final_seed_noise.csv`）：
  arc_challenge acc 相对 SD：14m **4.0%**、31m **5.9%**（绝对 SD ≈0.7–1.0pp，acc≈0.17–0.22）；
  blimp acc 相对 SD 随尺寸单调下降：14m 2.46% → 410m **0.91%**（绝对 SD 1.6pp → 0.7pp；极差最高 5.1pp）。
  → 在 14m–31m 上，一次典型"0.5–1pp 消融差值"完全躺在 1σ seed 噪声之内。
- **离群 run**（`tables/pp_outliers.csv`）：31m-seed5 blimp 终分 robust-z = **−11.9**、31m-seed0 z = −4.2 —— 与 PolyPythias 论文自报的不稳定 run 相互印证；410m-seed6 训练轨迹在 step 70k 后无评测。
- **S&N 代理交叉复验**（上文 T1-D）：10-seed 真值下代理低估中位 1.8×。
- **上游数据 bug**（瑕疵声明 #5）：410m "seed" lambada 评测实为基线模型重复评测，1095 文件剔除——50-run 资产的 lambada 维度实际不可用于 seed 方差。
- 局限：公开评测无 loss 轨迹（wandb 未镜像）、无 Pythia 全任务套件；本再分析以 blimp/arc 维度为准。

---

## 总结论

| 靶 | 已发表主张 | 判定 | 一句话依据 |
|---|---|---|---|
| T1 | S&N：末段 ckpt 噪声≈seed 噪声（R≥0.9），可代理 | **部分成立** | 原设置复现（R² 0.81–0.99、比率 0.98）；移植到其宣称的使用场景即失效：3-seed 下 R≥0.9 不可检验，accuracy 低估 1.4–2.2×（1B 恰 √2=源捆绑）、粗网格 ppl 反向高估 2–7×（趋势污染，去趋势仅 ≥750M 修复），PP 10-seed 复验低估 1.8× |
| T2 | 2605.20798 隐含：init-only 3-seed = 全源噪声地板 | **部分成立** | 两源同数量级但任务级会翻转主导（hellaswag order×3；winogrande init×2.4）；等效性用 10v9 个 1B run 都无法过 2× 界（0/27）；init-only 低估全源 σ 中位 ×1.3–1.6——同数量级近似可用，作精确门槛必偏松 |
| T3 | DataDecide：150M 排序 ~80% 配对正确 | **成立**（附注脚） | 复现 82.7%；新增：错误对中 54–59%（macro）/~82%（任务级）为 seed-不稳定 pair，富集 2×——"20% 错误"过半是噪声而非可修的排序错误 |

**对本项目的直接价值**：(1) σ 图集（PP 10-seed + DataDecide 3-seed + S&N 两臂）给本项目所有"±2 分窗/bootstrap SE"合同判据提供了首批实测底数；(2) T1/T2 的失效边界正是训练段（析因分解 + CRN）要填的空位——步噪声代理在小尺度失效与源不等效迹象都指向"必须实测、且必须分源"；(3) 两个可外发的独立发现：S&N 代理的移植失效边界 + PolyPythias 发布数据的 base-model 评测 bug。

## 复现
```
code/00_parse_dd.py            # macro 表 → dd_tidy.parquet
code/01_t1_sn_replication.py   # T1-A（S&N 自数据复现）
code/02_t1_datadecide.py       # T1-B（跨尺度移植 + MC 天花板）
code/05_t1_ppl.py              # T1-C（ppl + 去趋势）
code/03_t2_source_equiv.py     # T2
code/04_t3_decision_replay.py  # T3
code/07_parse_polypythias.py   # PP 解析（含模型一致性校验）
code/08_polypythias_analysis.py# PP 再分析 + T1-D
code/06_figures.py, 09_aggregates.py
```
环境：`./.conda/envs/tanh/bin/python3`（pandas 3.0.5 / numpy 1.26.4 / scipy 1.17.1 / matplotlib）。
原始数据缓存：`the raw cache`（约 1.1 GB；再分析产物已全部落在本目录）。
"""

out = pathlib.Path("data/REPORT.md")
out.write_text(REPORT, encoding="utf-8")
print("wrote", out, len(REPORT), "chars")
