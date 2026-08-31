# Acceptance gate (revision-3 batch 3): every number in abstract/intro/S5/S7 checked against
# values recomputed from the raw assets. Prints PASS/FAIL per check; exit 1 on any FAIL.
import os, re, sys, json, numpy as np, pandas as pd
from scipy import stats
R = os.environ.get("NFT_R", "data/analysis")
SN = os.environ.get("NFT_SN", "data/raw_cache/sn_datadecide_intermediate.parquet")
TEX = os.environ.get("NFT_TEX", "tex_snapshot")
d = pd.read_parquet(f"{R}/dd_tidy.parquet")
def fcs(g, need=3):
    c = g.groupby("step")["seed"].nunique(); com = c[c>=need].index
    return None if len(com)==0 else com.max()
def cell_stats(sz, task="olmes_10_macro_avg"):
    per = {}
    for mix, gg in d[(d.params==sz)&(d.task==task)].groupby("data"):
        s = fcs(gg)
        if s is None: continue
        v = gg[gg.step==s].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: per[mix] = (v.mean(), v.std(ddof=1))
    return per
def snr_dec(per):
    mu = np.array([v[0] for v in per.values()]); sd = np.array([v[1] for v in per.values()])
    return float(np.sqrt(max(mu.var(ddof=1)-np.mean(sd**2)/3,0))/np.sqrt(np.mean(sd**2)))
sys.path.insert(0, os.environ.get("NFT_HME_SRC", "code/enhancement3"))
import hme
def replay_paper(sz):
    """The paper's exact replay: target = 1B mean of last-3 common ckpts; proxy = final step."""
    X, Y, recipes = hme.extract_cell(d, "olmes_10_macro_avg", sz)
    st = hme.descriptive_stats(X, Y)
    acc3 = st["acc"]
    # 1-seed: per-seed decision accuracy averaged over seeds
    accs = []
    n = X.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    ybar = Y.mean(axis=1); dt = ybar[ii]-ybar[jj]; keep = dt!=0
    ii, jj, dt = ii[keep], jj[keep], dt[keep]
    for s in range(3):
        dm = X[ii,s]-X[jj,s]
        accs.append(float(((dm*dt)>0).mean()))
    return acc3, float(np.mean(accs))
fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ")+name); fails.append(not cond)

# SNR headline numbers
per60, per90, per150, per1b = cell_stats("60M"), cell_stats("90M"), cell_stats("150M"), cell_stats("1B")
s60, s90, s150, s1b = snr_dec(per60), snr_dec(per90), snr_dec(per150), snr_dec(per1b)
maxsmall = max(snr_dec(cell_stats(s)) for s in ["4M","6M","8M","10M","14M","16M","20M","60M"])
print(f"[recomputed] max deconv SNR <=60M: {maxsmall:.3f}; 60M {s60:.2f}; 90M {s90:.2f}; 150M {s150:.2f}; 1B {s1b:.2f}")
abs_t = open(f"{TEX}/sections/abstract.tex").read()
intro = open(f"{TEX}/sections/01_introduction.tex").read()
s3 = open(f"{TEX}/sections/03_t1_proxy.tex").read()
s5 = open(f"{TEX}/sections/05_t3_decision_snr.tex").read()
s7 = open(f"{TEX}/sections/08_prescriptions.tex").read()
appb = open(f"{TEX}/sections/app_b_t1.tex").read()
HERE = os.path.dirname(os.path.abspath(__file__))
check("abstract <=1.3 consistent (max %.2f <= 1.3)" % maxsmall, maxsmall <= 1.305)
check("abstract 1.2->2.6 consistent (60M %.2f, 90M %.2f)" % (s60,s90), abs(s60-1.23)<0.02 and abs(s90-2.61)<0.02)
check("80.2% replay protocol in S5", "80.2\\%" in s5 and "protocol" in s5)
# replay
acc150_3, acc150_1 = replay_paper("150M")
print(f"[recomputed] 150M: 3-seed {acc150_3*100:.1f}%, 1-seed {acc150_1*100:.1f}%")
check("82.7 / 80.2 replay", abs(acc150_3*100-82.7)<0.2 and abs(acc150_1*100-80.2)<0.2)
acc4_3, acc4_1 = replay_paper("4M")
print(f"[recomputed] 4M: 3-seed {acc4_3*100:.1f}%, 1-seed {acc4_1*100:.1f}%")
check("31.0 / 37.8", abs(acc4_3*100-31.0)<0.2 and abs(acc4_1*100-37.8)<0.2)
_abs1 = re.sub(r"\s+", " ", abs_t)
check("13.5 interval present", "13.5" in _abs1 and "8.3" in _abs1 and "57.0" in _abs1 and "does not exclude a majority" in _abs1)
# atlas 4M spread
c = pd.read_parquet(f"{R}/t1c_ppl_cells.parquet")
y = c[(c.params=="4M")&(c.dom=="c4_en")].y.dropna().to_numpy(); y=y[y>0]
sp = float(np.exp(np.sqrt(max(np.var(np.log(y),ddof=1)-0.411,0))))
print(f"[recomputed] 4M c4_en spread {sp:.2f}")
check("2.3x at 4M (C4-en)", abs(sp-2.29)<0.05 and "$2.3\\times$" in intro)
# 4M inversion
m4 = {k:v[0] for k,v in cell_stats("4M").items()}; m1 = {k:v[0] for k,v in cell_stats("1B").items()}
common = sorted(set(m4)&set(m1))
r4 = stats.pearsonr([m4[k] for k in common],[m1[k] for k in common])[0]
print(f"[recomputed] 4M-1B macro Pearson {r4:.3f}")
check("abstract macro-average ranking + -0.56", abs(r4-(-0.56))<0.02 and "macro-average readout" in abs_t and "negatively" in abs_t)
# revision-4 additions: 0/27 in abstract, MC calibration numbers in S3/appB, jump LORO in S5
t2 = pd.read_csv(os.environ.get("NFT_T2", os.environ.get("NFT_T2", "data/tables/t2_source_equivalence.csv")))
inband = int(((t2.ci_lo_F >= 0.5) & (t2.ci_hi_F <= 2.0)).sum())
check("0/27 band containment + abstract", inband == 0 and "($0/27$)" in abs_t)
mc = open(f"{HERE}/r4_prop1_mc_calibration.out.txt").read()
rows = [l for l in mc.splitlines() if "|" in l and l.strip()[0].isdigit()]
t1_r0, pw_r0 = [float(x) for x in rows[0].split("|")[1:]]
t1_r5, pw_r5 = [float(x) for x in rows[2].split("|")[1:]]
_s3 = re.sub(r"\s+", " ", s3); _appb = re.sub(r"\s+", " ", appb)
check("MC Type-I/power in S3+appB", abs(t1_r0-0.0691) < 5e-4 and abs(pw_r0-0.1240) < 5e-4
      and abs(t1_r5-0.0194) < 5e-4 and abs(pw_r5-0.0355) < 5e-4
      and "is $0.069$" in _s3 and "power collapses to $0.04$" in _s3
      and "Type-I error is $0.069$" in _appb and "$0.124$" in _appb and "$0.036$" in _appb)
_s5 = re.sub(r"\s+", " ", s5)
bj = json.load(open(f"{HERE}/r4_band_jump_sensitivity.json"))
loro = bj["jump_loro"]
njump = sum(1 for v in bj["jump_per_task"].values() if v["90M"] > v["60M"])
check("jump LORO ranges + 6/10 in S5",
      abs(loro["60M_range"][0]-1.13) < 0.01 and abs(loro["60M_range"][1]-1.35) < 0.01
      and abs(loro["90M_range"][0]-2.38) < 0.01 and abs(loro["90M_range"][1]-2.81) < 0.01
      and njump == 6 and "[1.13, 1.35]" in _s5 and "[2.38, 2.81]" in _s5 and "6 of 10" in _s5)
# 10M accuracy consistency check numbers
acc10_3, _ = replay_paper("10M")
print(f"[recomputed] 10M 3-seed acc {acc10_3*100:.1f}%")
check("58.0 at 10M in S5", abs(acc10_3*100-58.0)<0.2 and "58.0" in s5)
# per-task count positive
cnt=0
for t in ["arc_challenge","arc_easy","boolq","csqa","hellaswag","mmlu","openbookqa","piqa","socialiqa","winogrande"]:
    a_ = {k:v[0] for k,v in cell_stats("4M",t).items()}; b_ = {k:v[0] for k,v in cell_stats("1B",t).items()}
    cm = sorted(set(a_)&set(b_))
    if len(cm)>=10 and stats.pearsonr([a_[k] for k in cm],[b_[k] for k in cm])[0] > 0: cnt+=1
print(f"[recomputed] per-task positive at 4M: {cnt}/10")
check("5 of 10 positive in S5", cnt==5 and "positive on 5 of the 10 tasks" in s5)
# rule 4 multiples
band_eq = stats.t.ppf(0.975,4)*np.sqrt(2/3); band_dom = stats.t.ppf(0.975,2)/np.sqrt(3)
print(f"[recomputed] Welch band multiples: equal-var {band_eq:.2f} sigma, dominant-arm {band_dom:.2f} sigma_i")
check("rule 4 says 2.3--2.5", "2.3$--$2.5" in s7 and abs(band_eq-2.27)<0.05 and abs(band_dom-2.48)<0.05)
# S7 metric regularity ranges
check("S7 1.8--4.0 and deconvolved 1.9--5.8", "1.8$--$4.0" in s7 and "1.9$--$5.8" in s7)
# ---- revision-5 additions (second external review) ----
r5 = json.load(open(f"{HERE}/r5_family_bootstrap.json"))
_s5 = re.sub(r"\s+", " ", s5); _s7 = re.sub(r"\s+", " ", s7)
_appd = re.sub(r"\s+", " ", open(f"{TEX}/sections/app_d_t3.tex").read())
_s2 = re.sub(r"\s+", " ", open(f"{TEX}/sections/02_setup.tex").read())
_s3 = re.sub(r"\s+", " ", s3)
_alltex = re.sub(r"\s+", " ", " ".join(open(f"{TEX}/sections/{f}").read() for f in
              __import__("os").listdir(f"{TEX}/sections") if f.endswith(".tex")))
check("family bootstrap CI in S5", abs(r5["family_boot"]["lo"]+0.822)<0.01
      and abs(r5["family_boot"]["hi"]+0.155)<0.01 and "[-0.82, -0.16]" in _s5 and "0.001" in _s5)
check("argmax pick named + top3 ranks", r5["argmax"]["recipe"].startswith("DCLM-Baseline (QC FW 3")
      and r5["argmax"]["rank1b"] == 21 and r5["argmax"]["top3"] == [21, 11, 3]
      and "DCLM-Baseline (QC FW 3" in _s5 and "21st/11th/3rd" in _s5)
check("task-level blocks 1B", r5["blocks_1b"]["mmlu"] == 6 and r5["blocks_1b"]["hellaswag"] == 5
      and "six blocks" in _s7 and "six blocks" in _appd)
check("per-task 4M acc in appD", abs(r5["acc4"]["arc_easy"]*100-90.7) < 0.1
      and abs(r5["acc4"]["hellaswag"]*100-34.4) < 0.1 and "90.7" in _appd and "34.4" in _appd)
prim = t2[t2.metric_mode == "primary_like"]
infl = float((np.sqrt(prim.sd_init**2 + prim.sd_order**2) / prim.sd_init).median())
check("1.62 semantics factor in S3+abs", abs(infl-1.618) < 0.01 and "1.62" in _s3
      and "semantics gap" in abs_t)
check("Chinchilla fixed (5x, not 100x)", "5\\times$Chinchilla" in _s2
      and "100\\times$Chinchilla" not in _alltex)
# ---- appendix H (self-run crossed controls) ----
ST1 = os.environ.get("NFT_STAGE1", "data/stage_results")
ST2 = os.environ.get("NFT_STAGE2", "data/stage_results")
apph = re.sub(r"\s+", " ", open(f"{TEX}/sections/app_h_crossed.tex").read())
s1vd = json.load(open(f"{ST1}/stage1_variance_decomposition.json"))
_mains = ["blimp", "arc_easy", "arc_challenge", "piqa_local", "social_iqa_local", "lambada_openai"]
adds = sorted(s1vd[t]["addit_factor"] for t in _mains)
check("appH PP160 additivity median+range",
      abs(np.median(adds)-1.66) < 0.02 and abs(adds[0]-1.07) < 0.02 and abs(adds[-1]-12.97) < 0.02
      and "1.66" in apph and "12.97" in apph)
check("appH PP160 interaction CIs contain 0", all(s1vd[t]["int_ci"][0] < 0 < s1vd[t]["int_ci"][1] for t in _mains)
      and "consistent with zero on all six tasks" in apph)
s1pc = pd.read_parquet(f"{ST1}/stage1_proxy_calibration.parquet")
_bl = s1pc[s1pc.task == "blimp"].iloc[0]
check("appH PP160 proxy y/x 5.1", abs(_bl.yx_median - 5.06) < 0.05 and "5.1" in apph)
s2 = json.load(open(f"{ST2}/stage2_analysis.json"))
check("appH 20M proxy 4.0/2.4", abs(s2["proxy"]["blimp"]["yx_median"]-3.998) < 0.05
      and abs(s2["proxy"]["lambada_openai"]["yx_median"]-2.426) < 0.05
      and "4.0" in apph and "2.4" in apph)
check("appH 20M main effects ~0", s2["grid"]["arc_easy"]["sd_init"] == 0.0
      and s2["grid"]["blimp"]["sd_init"] == 0.0 and "undetectable" in apph)
# revision-7 additions (fourth review)
nb = json.load(open(f"{HERE}/r5_null_band.json"))
check("Q4 perfect-proxy band baseline 67.9", abs(nb["null_band_m5_n3"]-0.6794) < 5e-4
      and "67.9" in _s3)
oos = json.load(open(f"{HERE}/r5_oos_snr.json"))
check("out-of-sample SNR Spearman in S5", abs(oos["150M"]-0.903) < 0.005
      and abs(oos["530M"]-0.915) < 0.005 and "+0.90" in _s5 and "+0.92" in _s5)
e7 = json.load(open(f"{HERE}/r5_excl750.json"))
check("excl-750M figures in appD", abs(e7["macro_excl750"]-0.5864) < 5e-4
      and abs(e7["tasklevel_excl750"]-0.8139) < 5e-4 and "58.6" in _appd and "81.4" in _appd)
tk = json.load(open(f"{HERE}/r5_topk_regret.json"))
check("top-k regret 5.9/4.7 in S5", abs(tk["4M"]["top1"]-0.0595) < 2e-4
      and abs(tk["150M"]["top1"]-0.0466) < 2e-4 and "5.9" in _s5 and "4.7" in _s5)
sys.exit(1 if any(fails) else 0)
