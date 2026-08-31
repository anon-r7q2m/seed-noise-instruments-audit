import pandas as pd, numpy as np
OUT="data"
C = pd.read_parquet(f"{OUT}/analysis/t1b_cells.parquet")
c = C.dropna(subset=["x","y"]); c = c[(c.x>0)&(c.y>0)]
r = c.y/c.x
print("T1b ALL sizes pooled: n_cells=%d  med_ratio=%.2f  frac in [0.5,2]=%.3f  frac y>x=%.3f"
      % (len(c), r.median(), r.between(0.5,2).mean(), (r>1).mean()))
c1 = c[c.params=="1B"]; r1 = c1.y/c1.x
print("T1b 1B only: n=%d med_ratio=%.2f  frac in 2x=%.3f" % (len(c1), r1.median(), r1.between(0.5,2).mean()))
rs = c[c.params!="1B"]; rr = rs.y/rs.x
print("T1b <=750M: n=%d med_ratio=%.2f" % (len(rs), rr.median()))
cs = C.dropna(subset=["x","y_stab"]); cs = cs[(cs.x>0)&(cs.y_stab>0)]
rstab = cs.y_stab/cs.x
print("T1b stab: med_ratio=%.2f frac2x=%.3f" % (rstab.median(), rstab.between(0.5,2).mean()))
R3 = pd.read_csv(f"{OUT}/tables/t3_decision_replay.csv")
m = R3[R3.task=="olmes_10_macro_avg"]
big = m[m["size"].isin(["90M","150M","300M","530M","750M"])]
print("\nT3 macro 90M-750M: acc_seedmean=%.3f err_total=%d any_unstable=%d (share %.2f) small_unstable=%d tgt_unstable=%d stable=%d"
      % (big.acc_seedmean.mean(), big.err.sum(), big.err_any_unstable.sum(),
         big.err_any_unstable.sum()/big.err.sum(), big.err_small_unstable.sum(),
         big.err_tgt_unstable.sum(), big.err_stable.sum()))
r150 = m[m["size"]=="150M"].iloc[0]
print("T3 150M macro: acc=%.3f (single-seed %.3f), err=%d, unstable-any=%d (%.2f), tgt=%d, small=%d, stable=%d"
      % (r150.acc_seedmean, r150.acc_singleseed, r150.err, r150.err_any_unstable,
         r150.share_err_any_unstable, r150.err_tgt_unstable, r150.err_small_unstable, r150.err_stable))
print("\nT3 macro seedmean-vs-single gaps:")
for _, row in m.iterrows():
    print("  %s: %.3f vs %.3f (gap %.3f, err reduction %.1f%%)" % (row["size"], row.acc_seedmean,
        row.acc_singleseed, row.acc_seedmean-row.acc_singleseed,
        100*(row.acc_seedmean-row.acc_singleseed)/(1-row.acc_singleseed) if row.acc_singleseed<1 else 0))
T = pd.read_csv(f"{OUT}/tables/t2_source_equivalence.csv")
for mm in ["primary_like","bits_per_byte"]:
    s = T[T.metric_mode==mm]
    understate = np.sqrt(1/s.share_init)
    print("\nT2 [%s]: median init/order ratio=%.2f; init-only understates bundled sigma by median x%.2f (IQR %.2f-%.2f)"
          % (mm, s.ratio.median(), understate.median(), understate.quantile(.25), understate.quantile(.75)))
# task-level T3 pooled
ind = R3[R3.task!="olmes_10_macro_avg"]
bigi = ind[ind["size"].isin(["90M","150M","300M","530M","750M"])]
print("\nT3 task-level 90M-750M pooled: err=%d any_unstable_share=%.2f small_share=%.2f"
      % (bigi.err.sum(), bigi.err_any_unstable.sum()/bigi.err.sum(), bigi.err_small_unstable.sum()/bigi.err.sum()))
