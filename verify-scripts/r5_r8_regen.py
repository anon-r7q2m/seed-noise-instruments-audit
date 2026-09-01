#!/usr/bin/env python3
"""R12 major-6: canonical generators for the r5_*/r8_* jsons (rewrites them in place).

Recovered definitions (the original ad-hoc scripts were lost; each definition below was
recovered by matching the archived value, then FIXED as canonical):
  r5_null_band.json    share a perfect proxy lands in the 2x band under pure sampling noise:
                       y/x = sqrt(chi2_2/2) / sqrt(chi2_4/4)  (3 seeds -> df 2; 5 checkpoints
                       -> df 4 for the checkpoint-window SD).
  r5_oos_snr.json      per-scale Spearman across the 10 tasks: per-task deconvolved SNR vs
                       per-task pairwise accuracy (t3_pairs 'correct'), at 4M/150M/530M.
  r5_excl750.json      seed-unstable share among errors, flag = small_unstable OR
                       tgt_unstable; macro rows / 10-task pooled; 90M-530M (excl. 750M).
  r5_topk_regret.json  per scale: argmax pick by 3-seed mean at proxy scale vs 1B target
                       (mean of last-3 common checkpoints; hme.extract_cell convention),
                       regret = best 1B mean - pick's 1B mean; CI: seed-resampled, B=2000.
  r5_family_bootstrap.json  4M macro vs 1B Pearson under the 10-family cluster bootstrap,
                       B=20000. SUPERSEDES the archived [-0.82,-0.16] (unreproducible;
                       original script lost): canonical value [-0.72,-0.08].
  r8_prescription3.json  BH decidable share per scale (9 rule-3 scales) + SNR->accuracy
                       bins over the valid task-level cells of those scales (n=80) +
                       Spearman. NOTE: supersedes the archived 90-cell bins (unreproducible).
  r8_q3_ci.json        4M accuracy gap (3-seed minus 1-seed protocol) bootstrap under
                       BROKEN pairing (recipes resampled independently at 4M and 1B): the
                       zero-centered null the text cites ("centered at zero").
  r8_seed_share.json   two-way ANOVA per (task, scale) cell: shared-seed variance component
                       (MS_seed - MS_resid)/R truncated at 0, over MS_resid; mean over tasks.

Pure CPU, deterministic. NFT_R / NFT_HME_SRC overridable.
Writes the jsons above + r5_r8_regen_report.json; prints PASS/FAIL per item vs the values
the paper prints (post-revision-9).
"""
import os, json, sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
sys.path.insert(0, os.environ.get("NFT_HME_SRC", "/var/tmp/scx7ew2/work-20260825/rank4-paper/enhancement3/src"))
import hme

d = pd.read_parquet(f"{R}/dd_tidy.parquet")
pairs = pd.read_parquet(f"{R}/t3_pairs.parquet")
rng = np.random.default_rng(20260901)
TASKS10 = ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu","openbookqa","piqa","socialiqa","winogrande"]
SIZES14 = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
S9 = ["4M","10M","20M","60M","90M","150M","300M","530M","1B"]
report = {}
def rec(name, ok, got, want):
    report[name] = {"ok": bool(ok), "got": got, "want": want}
    print(("PASS " if ok else "FAIL "), name, "got", got, "want", want)

def final_scores(params, task):
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = v.values[:3]
    return out

def snr_dec(per):
    mu = np.array([v.mean() for v in per.values()]); sd = np.array([v.std(ddof=1) for v in per.values()])
    return float(np.sqrt(max(mu.var(ddof=1)-(sd**2).mean()/3,0))/np.sqrt((sd**2).mean()))

# ---------- r5_null_band (exact by quadrature) ----------
from scipy.integrate import quad
def ratio_cdf(r):
    # P( sqrt(chi2_2/2)/sqrt(chi2_4/4) <= r ) = P( chi2_4 >= 2*chi2_2 / r^2 )
    f = lambda u: stats.chi2.pdf(u, 2) * stats.chi2.sf(2.0 * u / (r * r), 4)
    return quad(f, 0, np.inf, limit=200)[0]
share = ratio_cdf(2.0) - ratio_cdf(0.5)
json.dump({"null_band_m5_n3": share, "definition": "P(0.5 <= sqrt(chi2_2/2)/sqrt(chi2_4/4) <= 2) by quadrature"},
          open(os.path.join(HERE, "r5_null_band.json"), "w"), indent=1)
rec("r5_null_band", abs(share - 0.6794) < 0.005, round(share, 4), "0.6794 (paper prints 67.9%)")

# ---------- r5_oos_snr ----------
acc = pairs[pairs.task.isin(TASKS10)].groupby(["task","size"])["correct"].mean()
oos = {}
for sz in ["4M","150M","530M"]:
    xs, ys = [], []
    for t in TASKS10:
        per = final_scores(sz, t)
        if len(per) < 10 or (t, sz) not in acc.index: continue
        xs.append(snr_dec(per)); ys.append(float(acc.loc[(t, sz)]))
    oos[sz] = float(stats.spearmanr(xs, ys).statistic)
json.dump(oos, open(os.path.join(HERE, "r5_oos_snr.json"), "w"), indent=1)
rec("r5_oos_snr", abs(oos["150M"]-0.903)<0.005 and abs(oos["530M"]-0.915)<0.005,
    {k: round(v,3) for k,v in oos.items()}, "4M .39 / 150M .90 / 530M .92")

# ---------- r5_excl750 ----------
def unstable_share(df, sizes):
    sub = df[df["size"].isin(sizes)]
    err = sub[~sub["correct"]]
    return float((err["small_unstable"] | err["tgt_unstable"]).mean())
S_90_530 = ["90M","150M","300M","530M"]
excl = {"macro_excl750": unstable_share(pairs[pairs.task=="olmes_10_macro_avg"], S_90_530),
        "tasklevel_excl750": unstable_share(pairs[pairs.task.isin(TASKS10)], S_90_530),
        "definition": "share of error pairs with small_unstable OR tgt_unstable; 90M-530M"}
json.dump(excl, open(os.path.join(HERE, "r5_excl750.json"), "w"), indent=1)
rec("r5_excl750", abs(excl["macro_excl750"]-0.5864)<0.005 and abs(excl["tasklevel_excl750"]-0.8139)<0.005,
    {k: round(v,4) for k,v in excl.items() if isinstance(v,float)}, "macro .5864 / task .8139")

# ---------- r5_topk_regret ----------
sc1x, Y1, rec1 = None, None, None
def topk(sz):
    global sc1x, Y1, rec1
    Xp, _, recp = hme.extract_cell(d, "olmes_10_macro_avg", sz)
    _, Y1, rec1 = hme.extract_cell(d, "olmes_10_macro_avg", "1B")
    assert recp == rec1
    m1 = Y1.mean(axis=1)
    pick = int(np.argmax(Xp.mean(axis=1)))
    regret = float(m1.max() - m1[pick])
    boots = []
    for _ in range(2000):
        bs = rng.integers(0, 3, size=(len(recp), 3))
        sx = np.take_along_axis(Xp, bs, axis=1).mean(axis=1)
        sy = np.take_along_axis(Y1, bs, axis=1).mean(axis=1)
        pk = int(np.argmax(sx))
        boots.append(float(sy.max() - sy[pk]))
    # top-3 regret: mean regret of the 3 best proxy-scale picks
    o3 = np.argsort(-Xp.mean(axis=1))[:3]
    top3 = float(np.mean([m1.max() - m1[i] for i in o3]))
    return {"top1": regret, "top3": top3,
            "rank1b": int(1 + (m1 > m1[pick]).sum()),
            "top1_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
tk = {sz: topk(sz) for sz in ["4M","10M","20M","60M","90M","150M","300M","530M","750M","1B"]}
json.dump(tk, open(os.path.join(HERE, "r5_topk_regret.json"), "w"), indent=1)
rec("r5_topk_regret", abs(tk["4M"]["top1"]-0.0595)<1e-3 and abs(tk["150M"]["top1"]-0.0466)<1e-3
    and tk["4M"]["rank1b"]==24 and tk["150M"]["rank1b"]==21,
    {"4M": round(tk["4M"]["top1"],4), "150M": round(tk["150M"]["top1"],4)}, "5.9 / 4.7 pts, ranks 24 / 21")

# ---------- r5_family_bootstrap (canonical recompute) ----------
X4, Y4, rec4 = hme.extract_cell(d, "olmes_10_macro_avg", "4M")
_, Y1, _ = hme.extract_cell(d, "olmes_10_macro_avg", "1B")
FAM = {}
FAM["C4"]="c4"; FAM["DCLM-Baseline"]="dclm-base"
for r in ["DCLM-Baseline (QC 10%)","DCLM-Baseline (QC 20%)"]: FAM[r]="dclm-qc-pct"
for r in ["DCLM-Baseline (QC 7%, FW2)","DCLM-Baseline (QC 7%, FW3)"]: FAM[r]="dclm-qc-7fw"
for r in ["DCLM-Baseline (QC FW 3%)","DCLM-Baseline (QC FW 10%)"]: FAM[r]="dclm-qc-fw"
for r in ["DCLM-Baseline 25% / Dolma 75%","DCLM-Baseline 50% / Dolma 50%","DCLM-Baseline 75% / Dolma 25%"]: FAM[r]="dclm-dolma-mix"
for r in ["Dolma1.6++","Dolma1.7","Dolma1.7 (no Flan)","Dolma1.7 (no Reddit)","Dolma1.7 (no code)","Dolma1.7 (no math, code)"]: FAM[r]="dolma"
for r in ["Falcon","Falcon+CC"]: FAM[r]="falcon-base"
for r in ["Falcon+CC (QC 10%)","Falcon+CC (QC 20%)","Falcon+CC (QC Orig 10%)","Falcon+CC (QC Tulu 10%)"]: FAM[r]="falcon-qc"
for r in ["FineWeb-Edu","FineWeb-Pro"]: FAM[r]="fineweb"
m4 = X4.mean(axis=1); m1v = Y1.mean(axis=1)
fams = {}
for i, r in enumerate(rec4): fams.setdefault(FAM[r], []).append(i)
fl = list(fams)
point = float(stats.pearsonr(m4, m1v)[0])
boot = []
for _ in range(20000):
    pick = rng.choice(len(fl), size=len(fl), replace=True)
    idx = np.array([i for k in pick for i in fams[fl[k]]])
    aa, bb = m4[idx], m1v[idx]
    if aa.std() > 0 and bb.std() > 0:
        boot.append(stats.pearsonr(aa, bb)[0])
boot = np.array(boot)
fb = {"family_boot": {"point": point, "lo": float(np.percentile(boot,2.5)),
                      "hi": float(np.percentile(boot,97.5)), "p_pos": float((boot>0).mean()),
                      "B": 20000, "note": "canonical recompute; supersedes archived [-0.82,-0.16] (lost script)"}}
# merge: preserve the argmax / blocks_1b / acc4 sections produced by
# r5_review2_verification.py (still canonical for those quantities)
_prev = json.load(open(os.path.join(HERE, "r5_family_bootstrap.json"))) if os.path.exists(os.path.join(HERE, "r5_family_bootstrap.json")) else {}
for k in ("argmax", "blocks_1b", "acc4"):
    if k in _prev: fb[k] = _prev[k]
json.dump(fb, open(os.path.join(HERE, "r5_family_bootstrap.json"), "w"), indent=1)
rec("r5_family_bootstrap", abs(fb["family_boot"]["lo"]+0.72)<0.02 and abs(fb["family_boot"]["hi"]+0.08)<0.02,
    {"lo": round(fb["family_boot"]["lo"],3), "hi": round(fb["family_boot"]["hi"],3)}, "[-0.72,-0.08] (revised text)")

# ---------- r8_prescription3 ----------
def pair_ps(scores):
    rec_ = sorted(scores); n = len(rec_)
    M = np.array([scores[r].mean() for r in rec_]); V = np.array([scores[r].var(ddof=1) for r in rec_])
    iu = np.triu_indices(n, 1)
    dm = M[iu[0]]-M[iu[1]]; se = np.sqrt(V[iu[0]]/3+V[iu[1]]/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(dm)/se; nu = (V[iu[0]]/3+V[iu[1]]/3)**2/((V[iu[0]]/3)**2/2+(V[iu[1]]/3)**2/2)
    t[~np.isfinite(t)] = 0; nu[~np.isfinite(nu)] = 2
    return 2*stats.t.sf(t, nu)
bh = {}
for sz in S9:
    p = pair_ps(final_scores(sz, "olmes_10_macro_avg"))
    m = len(p); o = np.argsort(p)
    k = np.where(p[o] <= 0.05*np.arange(1, m+1)/m)[0]
    bh[sz] = float((k.max()+1)/m) if len(k) else 0.0
xs, ys = [], []
for sz in S9:
    for t in TASKS10:
        per = final_scores(sz, t)
        if len(per) < 10 or (t, sz) not in acc.index: continue
        xs.append(snr_dec(per)); ys.append(float(acc.loc[(t, sz)]))
xs = np.array(xs); ys = np.array(ys)
bins = [(0,0.5),(0.5,1),(1,1.5),(1.5,2.5),(2.5,4),(4,10)]
got_bins = {f"[{a},{b})": float(ys[(xs>=a)&(xs<b)].mean()) for a,b in bins}
p3 = {"bh_decidable_share": bh,
      "snr_acc_bins": {k: round(v,3) for k,v in got_bins.items()},
      "snr_acc_spearman": float(stats.spearmanr(xs, ys).statistic),
      "snr_acc_cells": int(len(xs)),
      "note": "bins/Spearman over the 9 rule-3 scales' valid task-level cells; supersedes archived 90-cell values (unreproducible)"}
json.dump(p3, open(os.path.join(HERE, "r8_prescription3.json"), "w"), indent=1)
bh_ok = all(abs(bh[k]-v)<5e-4 for k,v in    # archived json was rounded to 3 decimals
    {"4M":0.0,"10M":0.0,"20M":0.0,"60M":0.0,"90M":0.37,"150M":0.507,"300M":0.55,"530M":0.433,"1B":0.623}.items())
rec("r8_prescription3 BH shares", bh_ok, {k: round(v,3) for k,v in bh.items()},
    "0/0/0/0/.37/.51/.55/.43/.62")
bvals = [got_bins[k] for k in got_bins]
rec("r8_prescription3 bins (revised canonical)",
    len(xs)==80 and abs(p3["snr_acc_spearman"]-0.885)<0.01 and all(bvals[i]<bvals[i+1] for i in range(len(bvals)-1)),
    {"cells": p3["snr_acc_cells"], "spearman": round(p3["snr_acc_spearman"],3),
     "bins": p3["snr_acc_bins"]}, "80 cells, monotone bins, Spearman .885")

# ---------- r8_q3_ci (broken-pairing null) ----------
def gap(idx4, idx1):
    tot = c3 = c1 = 0
    for a in range(len(idx4)):
        for b in range(a+1, len(idx4)):
            i, j = idx4[a], idx4[b]; i1, j1 = idx1[a], idx1[b]
            dt = m1v[i1]-m1v[j1]
            if dt == 0: continue
            tot += 1; c3 += ((m4[i]-m4[j])*dt > 0)
            for s in range(3): c1 += ((X4[i, s]-X4[j, s])*dt > 0)/3.0
    return c3/tot - c1/tot
n = len(rec4)
boot = np.array([gap(rng.choice(n, n, replace=True), rng.choice(n, n, replace=True)) for _ in range(2000)])
q3 = {"ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
      "median": float(np.median(boot)), "frac_negative": float((boot < 0).mean()),
      "definition": "gap (3-seed minus 1-seed protocol accuracy) with recipes resampled independently at 4M and 1B (broken pairing)"}
json.dump(q3, open(os.path.join(HERE, "r8_q3_ci.json"), "w"), indent=1)
rec("r8_q3_ci", abs(q3["median"]) < 0.005 and abs(q3["frac_negative"]-0.5) < 0.05,
    {"ci": [round(x,3) for x in q3["ci"]], "median": round(q3["median"],4)}, "centered at zero")

# ---------- r8_seed_share ----------
def seed_share(sz):
    out = []
    for t in TASKS10:
        per = final_scores(sz, t)
        if len(per) < 10: continue
        rec_ = sorted(per); X = np.array([per[r] for r in rec_]); Rr, Cc = X.shape
        gm = X.mean()
        r_eff = X.mean(axis=1, keepdims=True)-gm; s_eff = X.mean(axis=0, keepdims=True)-gm
        resid = X - gm - r_eff - s_eff
        MS_s = Rr*(s_eff**2).sum()/(Cc-1)
        MS_e = (resid**2).sum()/((Rr-1)*(Cc-1))
        out.append(max((MS_s-MS_e)/Rr, 0.0)/MS_e)
    return float(np.mean(out))
ss = {sz: seed_share(sz) for sz in ["4M","20M","60M","90M","150M","1B"]}
ss["definition"] = "two-way ANOVA MoM per task cell; mean over tasks; share of residual MS"
json.dump(ss, open(os.path.join(HERE, "r8_seed_share.json"), "w"), indent=1)
rngpct = [v for k, v in ss.items() if k != "definition"]
rec("r8_seed_share", max(rngpct) <= 0.056 and min(rngpct) >= 0.005,
    {k: round(v,4) for k,v in ss.items() if k != "definition"}, "paper prints range 0.6-5.5%")

with open(os.path.join(HERE, "r5_r8_regen_report.json"), "w") as f:
    json.dump(report, f, indent=1)
n_fail = sum(1 for v in report.values() if not v["ok"])
print(f"\n{'ALL REGEN CHECKS PASSED' if n_fail==0 else f'{n_fail} FAILURES'}")
sys.exit(1 if n_fail else 0)
