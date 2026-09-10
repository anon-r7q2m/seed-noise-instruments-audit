#!/usr/bin/env python3
"""R16 external-review items (2026-09-10, review-r16-external EXT1-M5 / nit / m14):

(a) EXT1-M5: decidable share under a POOLED variance (reviewer: at het ratio ~1.1-1.2
    the pair-specific Welch df 2-4 is the lowest-power legal choice; pooled df~50 is
    defensible and halves the bands). Replicates the pair-specific BH shares first
    (must match r8_prescription3.json), then reports the pooled-variant shares under
    BH and BY. Output feeds the rule-3 wording decision.
(b) EXT1 nit: "correlation climbs monotonically to +0.91 at 530M" lists no 6M--14M
    values -- compute the full per-scale macro-vs-1B Pearson series to check monotonicity.
(c) EXT1-m14: k(R0) at R0 in {0.80, 0.85, 0.90} (80% power, alpha=0.05, the panel's
    m=5/n=3/sigma_a=0.74 operating point) so the appendix can show the requirement is
    not tuned to 0.9.
(d) EXT2 nit: Table 2 prints median noise but the SNR denominator is RMS noise --
    emit the RMS column values for the table.

Pure CPU, deterministic. NFT_R overridable.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
# self-locating default: inside the anonymous repo, verify-scripts/ sits next to data/analysis
_repo_r = os.path.normpath(os.path.join(HERE, "..", "data", "analysis"))
R = os.environ.get("NFT_R", _repo_r if os.path.exists(os.path.join(_repo_r, "dd_tidy.parquet"))
                   else "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")

def final_scores(params, task="olmes_10_macro_avg"):
    g = d[(d.params == params) & (d.task == task)]
    out = {}
    for mix, gg in g.groupby("data"):
        c = gg.groupby("step")["seed"].nunique(); com = c[c >= 3].index
        if len(com) == 0: continue
        v = gg[gg.step == com.max()].groupby("seed")["primary_metric"].mean()
        if len(v) >= 3: out[mix] = v.values[:3]
    return out

S9 = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B"]
ALL = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B"]

def bh_share(p, q=0.05):
    m = len(p); o = np.argsort(p)
    k = np.where(p[o] <= q*np.arange(1, m+1)/m)[0]
    return float((k.max()+1)/m) if len(k) else 0.0

def by_share(p, q=0.05):
    m = len(p); c = np.sum(1.0/np.arange(1, m+1))
    o = np.argsort(p)
    k = np.where(p[o] <= q*np.arange(1, m+1)/(m*c))[0]
    return float((k.max()+1)/m) if len(k) else 0.0

# ---------------- (a) pooled vs pair-specific decidable share ----------------
rep = json.load(open(f"{HERE}/r8_prescription3.json"))
out_a, ok_rep = {}, True
for sz in S9:
    sc = final_scores(sz)
    rec_ = sorted(sc); n = len(rec_)
    M = np.array([sc[r].mean() for r in rec_]); V = np.array([sc[r].var(ddof=1) for r in rec_])
    iu = np.triu_indices(n, 1)
    dm = M[iu[0]] - M[iu[1]]
    # pair-specific Welch (replication)
    se = np.sqrt(V[iu[0]]/3 + V[iu[1]]/3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(dm)/se
        nu = (V[iu[0]]/3+V[iu[1]]/3)**2/((V[iu[0]]/3)**2/2+(V[iu[1]]/3)**2/2)
    t[~np.isfinite(t)] = 0; nu[~np.isfinite(nu)] = 2
    p_welch = 2*stats.t.sf(t, nu)
    bh_w = bh_share(p_welch)
    if sz in rep["bh_decidable_share"]:
        ok_rep &= abs(bh_w - rep["bh_decidable_share"][sz]) < 5e-4
    # pooled variance: sp^2 = mean(V_i), df = 2n
    sp2 = V.mean(); df_pool = 2*n
    t_p = np.abs(dm)/np.sqrt(sp2*(1/3+1/3))
    p_pool = 2*stats.t.sf(t_p, df_pool)
    out_a[sz] = {"welch_bh": bh_w, "pooled_bh": bh_share(p_pool),
                 "pooled_by": by_share(p_pool), "het": float(V.mean()/np.median(np.sqrt(V))**2)}
print("(a) replication of pair-specific BH shares:", "MATCH" if ok_rep else "MISMATCH!")
for sz in S9:
    r = out_a[sz]
    print(f"  {sz:>5}: welch-BH {r['welch_bh']:.3f} | pooled-BH {r['pooled_bh']:.3f} | pooled-BY {r['pooled_by']:.3f} | het {r['het']:.2f}")

# ---------------- (b) per-scale macro-vs-1B Pearson (monotonicity check) ----------------
sc1 = final_scores("1B")
corr = {}
for sz in ALL:
    sc = final_scores(sz)
    rec_ = sorted(set(sc) & set(sc1))
    a = np.array([sc[r].mean() for r in rec_]); b = np.array([sc1[r].mean() for r in rec_])
    corr[sz] = float(stats.pearsonr(a, b)[0])
mono = all(corr[ALL[i]] <= corr[ALL[i+1]] + 1e-9 for i in range(len(ALL)-1))
print("(b) per-scale macro-1B Pearson:", {k: round(v,3) for k,v in corr.items()})
print("    monotone nondecreasing:", mono)

# ---------------- (c) k(R0) at three R0 levels (Prop 1 closed form) ----------------
def trigamma(x):
    return float(stats.polygamma(1, x)[0] if hasattr(stats, 'polygamma') else np.nan)
# trigamma via polygamma(1, .)
from scipy.special import polygamma
sig_a = 0.74; m = 5; nn = 3
A = 1.0/np.sqrt((1+polygamma(1, (m-1)/2)/(4*sig_a**2)) * (1+polygamma(1, (nn-1)/2)/(4*sig_a**2)))
za = stats.norm.ppf(0.975)  # alpha=0.05 one-sided -> 1.6449 for level .05; certificate test
z1a = stats.norm.ppf(0.95)
kc = {}
for R0 in [0.80, 0.85, 0.90]:
    delta = np.arctanh(A) - np.arctanh(A*R0)
    k = ((z1a + stats.norm.ppf(0.80))**2)/(delta**2) + 3
    kc[str(R0)] = int(np.ceil(k))
print(f"(c) A={A:.4f}; k(R0) at 80% power: {kc}")

# ---------------- (d) RMS noise column for Table 2 ----------------
rms = {}
for sz in ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]:
    sc = final_scores(sz)
    V = np.array([sc[r].var(ddof=1) for r in sorted(sc)])
    # relative SD basis (divide by |mean|), matching Table 2's noise convention
    M = np.array([sc[r].mean() for r in sorted(sc)])
    rel = np.sqrt(V)/np.abs(M)
    rms[sz] = float(np.sqrt((rel**2).mean()))
print("(d) RMS relative noise per scale:", {k: round(v,5) for k,v in rms.items()})

json.dump({"pooled_shares": out_a, "welch_replication_ok": bool(ok_rep),
           "macro_1b_corr": corr, "macro_1b_corr_monotone": bool(mono),
           "k_R0_80pct": kc, "rms_noise": rms},
          open(os.path.join(HERE, "r16_pooled_decidable.json"), "w"), indent=1)
print("wrote r16_pooled_decidable.json")
