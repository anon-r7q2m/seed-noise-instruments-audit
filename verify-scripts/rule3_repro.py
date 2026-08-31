# Rule-3 reference implementation: must reproduce Table 1 (tab:snr) deconvolved SNR column
# digit-for-digit, plus the Welch coverage column.
import numpy as np, pandas as pd
from scipy import stats
import os
d = pd.read_parquet(os.environ.get("NFT_R", "data/analysis") + "/dd_tidy.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
def final_common_step(g, need=3):
    cnt = g.groupby("step")["seed"].nunique(); common = cnt[cnt>=need].index
    return None if len(common)==0 else common.max()

def audit(scores):
    """scores: (n_candidates, n_seeds) final scores at the proxy scale."""
    n_seeds = scores.shape[1]
    mu, sd = scores.mean(1), scores.std(1, ddof=1)
    snr = np.sqrt(max(mu.var(ddof=1) - (sd**2).mean()/n_seeds, 0.0)) / np.sqrt((sd**2).mean())
    ii, jj = np.triu_indices(len(mu), k=1)
    nu_ij = (sd[ii]**2/n_seeds + sd[jj]**2/n_seeds)**2 / (
        (sd[ii]**2/n_seeds)**2/(n_seeds-1) + (sd[jj]**2/n_seeds)**2/(n_seeds-1))
    band = stats.t.ppf(0.975, nu_ij) * np.sqrt(sd[ii]**2/n_seeds + sd[jj]**2/n_seeds)
    covered = np.abs(mu[ii]-mu[jj]) < band
    # indistinguishable groups: connected components of the COVERED graph
    parent = list(range(len(mu)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for a,b in zip(ii[covered], jj[covered]): parent[find(a)] = find(b)
    groups = len({find(i) for i in range(len(mu))})
    return snr, covered.mean(), groups

EXPECTED = {  # (snr_dec, welch_share) from tab:snr (landed ea714a3)
 "4M":(0.30,93.7),"6M":(0.63,83.7),"8M":(0.33,95.7),"10M":(0.00,93.0),"14M":(0.29,95.7),
 "16M":(0.72,83.0),"20M":(0.77,86.7),"60M":(1.23,72.7),"90M":(2.61,47.0),"150M":(3.18,38.7),
 "300M":(3.57,36.7),"530M":(2.98,39.3),"750M":(2.52,46.7),"1B":(3.52,33.0)}
ok = True
for p in ORDER:
    g = d[(d.params==p)&(d.task=="olmes_10_macro_avg")]
    scores = {}
    for mix, gg in g.groupby("data"):
        s = final_common_step(gg)
        if s is None: continue
        v = gg[gg.step==s].groupby("seed")["primary_metric"].mean()
        if len(v)>=3: scores[mix] = v.values[:3]
    M = np.array([scores[k] for k in sorted(scores)])
    snr, cov, ngroups = audit(M)
    es, ec = EXPECTED[p]
    match = (round(snr,2)==es) and (round(cov*100,1)==ec)
    ok &= match
    print(f"{p}: snr={snr:.2f} (expect {es})  welch={cov*100:.1f}% (expect {ec}%)  groups={ngroups}  {'OK' if match else 'MISMATCH'}")
print("\nALL MATCH" if ok else "\nFAILURES PRESENT")
