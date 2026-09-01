# Acceptance gate (revision-3 batch 3): every number in abstract/intro/S5/S7 checked against
# values recomputed from the raw assets. Prints PASS/FAIL per check; exit 1 on any FAIL.
import os, re, sys, json, numpy as np, pandas as pd
from scipy import stats
R = os.environ.get("NFT_R", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/zero-gpu/analysis")
SN = os.environ.get("NFT_SN", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/tmp/rank04-zerogpu/data/sn_datadecide_intermediate.parquet")
TEX = os.environ.get("NFT_TEX", "/var/tmp/scx7ew2/work-20260825/overleaf/noise-floor-trio")
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
sys.path.insert(0, os.environ.get("NFT_HME_SRC", "/var/tmp/scx7ew2/work-20260825/rank4-paper/enhancement3/src"))
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
def _read_tex(p):
    # 只读正文：去掉注释行，避免头部 changelog 注释里的数字误伤门禁
    return "\n".join(l for l in open(p).read().splitlines() if not l.lstrip().startswith("%"))
abs_t = _read_tex(f"{TEX}/sections/abstract.tex")
intro = _read_tex(f"{TEX}/sections/01_introduction.tex")
s3 = _read_tex(f"{TEX}/sections/03_t1_proxy.tex")
s5 = _read_tex(f"{TEX}/sections/05_t3_decision_snr.tex")
s7 = _read_tex(f"{TEX}/sections/08_prescriptions.tex")
appb = _read_tex(f"{TEX}/sections/app_b_t1.tex")
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
_lim = re.sub(r"\s+", " ", open(f"{TEX}/sections/10_limitations.tex").read())
check("13.5 interval present (in S5 body)", "13.5" in re.sub(r"\s+"," ",s5) and "8.3" in re.sub(r"\s+"," ",s5)
      and "57.0" in re.sub(r"\s+"," ",s5) and "does not exclude a majority" in re.sub(r"\s+"," ",s5))
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
check("abstract inversion + -0.56", abs(r4-(-0.56))<0.02 and "inverts" in abs_t and "-0.56" in abs_t)
# revision-4 additions: 0/27 in abstract, MC calibration numbers in S3/appB, jump LORO in S5
t2 = pd.read_csv(os.environ.get("NFT_T2", f"{R}/../tables/t2_source_equivalence.csv"))
inband = int(((t2.ci_lo_F >= 0.5) & (t2.ci_hi_F <= 2.0)).sum())
inband_bs = int(((t2.ci_lo_bs >= 0.5) & (t2.ci_hi_bs <= 2.0)).sum())
_int1 = re.sub(r"\s+", " ", intro)
check("0/27 F + 2/27 bootstrap (intro)", inband == 0 and inband_bs == 2
      and "($0/27$ under the analytic F intervals" in _int1 and "$2/27$ under bootstrap" in _int1)
mc = open(f"{HERE}/r4_prop1_mc_calibration.out.txt").read()
rows = [l for l in mc.splitlines() if "|" in l and l.strip()[0].isdigit()]
t1_r0, pw_r0 = [float(x) for x in rows[0].split("|")[1:]]
t1_r5, pw_r5 = [float(x) for x in rows[2].split("|")[1:]]
_s3 = re.sub(r"\s+", " ", s3); _appb = re.sub(r"\s+", " ", appb)
check("MC Type-I/power in S3+appB", abs(t1_r0-0.0691) < 5e-4 and abs(pw_r0-0.1240) < 5e-4
      and abs(t1_r5-0.0194) < 5e-4 and abs(pw_r5-0.0355) < 5e-4
      and "Type-I $0.069$" in _s3 and "power $0.12$" in _s3
      and "Type-I error is $0.069$" in _appb and "$0.124$" in _appb and "$0.036$" in _appb)
_s5 = re.sub(r"\s+", " ", s5)
bj = json.load(open(f"{HERE}/r4_band_jump_sensitivity.json"))
loro = bj["jump_loro"]
njump = sum(1 for v in bj["jump_per_task"].values() if v["90M"] > v["60M"])
check("jump LORO ranges + 6/10 in appD",
      abs(loro["60M_range"][0]-1.13) < 0.01 and abs(loro["60M_range"][1]-1.35) < 0.01
      and abs(loro["90M_range"][0]-2.38) < 0.01 and abs(loro["90M_range"][1]-2.81) < 0.01
      and njump == 6 and "[1.13, 1.35]" in re.sub(r"\s+"," ",open(f"{TEX}/sections/app_d_t3.tex").read())
      and "[2.38, 2.81]" in re.sub(r"\s+"," ",open(f"{TEX}/sections/app_d_t3.tex").read())
      and "6 of 10" in re.sub(r"\s+"," ",open(f"{TEX}/sections/app_d_t3.tex").read()))
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
check("5 of 10 positive in S5", cnt==5 and "positive on 5 of the 10 tasks" in _s5)
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
check("family bootstrap CI in S5", abs(r5["family_boot"]["lo"]+0.722)<0.01
      and abs(r5["family_boot"]["hi"]+0.078)<0.01 and "[-0.72, -0.08]" in _s5 and "0.011" in _s5)
check("argmax pick named + top3 ranks", r5["argmax"]["recipe"].startswith("DCLM-Baseline (QC FW 3")
      and r5["argmax"]["rank1b"] == 21 and r5["argmax"]["top3"] == [21, 11, 3]
      and "DCLM-Baseline (QC FW 3" in _s5 and "21st/11th/3rd" in _s5)
check("task-level blocks 1B", r5["blocks_1b"]["mmlu"] == 6 and r5["blocks_1b"]["hellaswag"] == 5
      and "six blocks" in _appd)
check("per-task 4M acc in appD", abs(r5["acc4"]["arc_easy"]*100-90.7) < 0.1
      and abs(r5["acc4"]["hellaswag"]*100-34.4) < 0.1 and "90.7" in _appd and "34.4" in _appd)
prim = t2[t2.metric_mode == "primary_like"]
infl = float((np.sqrt(prim.sd_init**2 + prim.sd_order**2) / prim.sd_init).median())
check("1.62 semantics factor in S3", abs(infl-1.618) < 0.01 and "1.62" in _s3
      and "semantics" in _s3)
check("Chinchilla fixed (5x, not 100x)", "5\\times$Chinchilla" in _s2
      and "100\\times$Chinchilla" not in _alltex)
# ---- appendix H (self-run crossed controls) ----
ST1 = os.environ.get("NFT_STAGE1", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/review-2027/stage1/results")
ST2 = os.environ.get("NFT_STAGE2", "/home/bingxing2/home/scx7ew2/tanh/Tanhäuser/runs/exp-rank04/review-2027/stage2/results")
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
# revision-8 additions (reviews 5+6)
ss = json.load(open(f"{HERE}/r8_seed_share.json"))
check("shared-seed share <=5.5%", max(v for k, v in ss.items() if k != "definition") <= 0.056 and "5.5" in _appd)
q3 = json.load(open(f"{HERE}/r8_q3_ci.json"))
check("Q3 4M gap bootstrap centered at zero", abs(q3["median"]) < 0.005
      and abs(q3["frac_negative"]-0.5) < 0.05 and "centered at zero" in _s5)

# ---- revision-9 additions (reviews 10-12) ----
# tau ceiling (R11-Q1): observed per-task median 0.03, ceiling 0.53 [0.49,0.56]
ce = json.load(open(f"{HERE}/r9_perfect_proxy_ceiling.json"))
cei_pt = [v["ceil_tau_median_of_tasks"] for v in ce["per_task"].values()]
obs_pt = [v["obs_tau_median"] for v in ce["per_task"].values()]
check("tau ceiling: obs 0.03 / ceil 0.53 [0.49,0.56]",
      abs(ce["summary"]["per_task_obs_tau_median_over_scales"]-0.027) < 0.01
      and abs(np.median(cei_pt)-0.53) < 0.02
      and min(cei_pt) >= 0.48 and max(cei_pt) <= 0.57
      and "0.53" in _s3 and "0.49" in _s3 and "$0.56$" in _s3
      and abs(ce["summary"]["per_task_obs_top5_scale_range"][0]-0.1) < 0.02)
# joint bootstrap (R12-Q2): [-0.68, 0.08] in S5
jb = json.load(open(f"{HERE}/r9_joint_bootstrap.json"))
check("joint family+seed CI in S5", abs(jb["corr_4m_1b"]["joint_family_seed_ci"][0]+0.675) < 0.02
      and abs(jb["corr_4m_1b"]["joint_family_seed_ci"][1]-0.080) < 0.02
      and "[-0.68, 0.08]" in _s5)
# repetition (R11-Q2): 3 pools, 1.02-1.22 epochs, drop -> -0.53
rp = json.load(open(f"{HERE}/r9_repetition_check.json"))
check("repetition: 3 pools, -0.53", len(rp["repeating_at_1B"]) == 3
      and abs(rp["correlations"]["drop_repeating"]["pearson"]+0.534) < 0.01
      and "1.02" in _s5 and "1.22" in _s5 and "-0.53" in _s5)
# chance-task ablation (R10-Q3): +0.56 [0.14, 0.70]
ab = json.load(open(f"{HERE}/r9_chance_task_ablation.json"))
check("ablation +0.56 [0.14,0.70] in S5",
      abs(ab["correlations"]["V2_keep"]["result"]["pearson"]-0.560) < 0.01
      and "+0.56" in _s5 and "0.14" in _s5 and "0.70" in _s5)
# BY/maxT (R12-Q4): nonzero shares 1-16% in S7; zero at <=60M invariant
ds = json.load(open(f"{HERE}/r9_decidable_share.json"))
by_nonzero = [v["share_BY"] for v in ds.values() if v["share_BH"] > 0]
mt_nonzero = [v["share_maxT"] for v in ds.values() if v["share_BH"] > 0]
check("BY/maxT shares 1-16% + zero floor invariant",
      min(by_nonzero+mt_nonzero) >= 0.005 and max(by_nonzero+mt_nonzero) <= 0.17
      and all(ds[s]["share_BH"] == 0.0 and ds[s]["share_BY"] == 0.0 and ds[s]["share_maxT"] == 0.0
              for s in ["4M","10M","20M","60M"])
      and "1$--$16\\%" in _s7)
# head-to-head (R12 major 5): 0.83 vs 0.65 in appD
h2h = json.load(open(f"{HERE}/r9_sn_headtohead.json"))
check("S&N head-to-head 0.83/0.65 in appD", abs(h2h["spearman_ours"]-0.833) < 0.01
      and abs(h2h["spearman_sn"]-0.650) < 0.01 and "0.83" in _appd and "0.65" in _appd)
# revised bins (reproducibility): 80 cells, Spearman +0.89, monotone bins in S7/appD
p3 = json.load(open(f"{HERE}/r8_prescription3.json"))
check("revised bins 80 cells +0.89", p3["snr_acc_cells"] == 80
      and abs(p3["snr_acc_spearman"]-0.885) < 0.01
      and "+0.89" in _s7 and "63\\%" in _s7 and "0.52" in _appd and "0.93" in _appd)
# revision-9 additions (reviews 10-12) ----
# 20M replication separation (R10-W4/R11-W3b/R12 major-4)
rp2 = json.load(open(f"{ST2}/stage2_rep_separation.json"))
_pt = rp2["per_task"]
diffs = [c[t]["absdiff"] for c in rp2["cells"].values() for t in c]
check("20M rep separation: 5 cells, bounds, run range",
      len(rp2["cells"]) == 5
      and abs(_pt["blimp"]["int_bound_new"]-0.0101) < 0.001
      and _pt["social_iqa_local"]["int_bound_new"] == 0.0 and _pt["arc_challenge"]["int_bound_new"] == 0.0
      and abs(float(np.median(diffs))-0.0046) < 0.001 and abs(max(diffs)-0.0231) < 0.001
      and "0.023" in apph and "five replicated cells" in apph and "0.0101" in apph)
sys.exit(1 if any(fails) else 0)
