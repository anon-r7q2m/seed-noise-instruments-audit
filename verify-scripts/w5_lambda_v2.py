import pandas as pd, numpy as np
from scipy.special import polygamma
c = pd.read_parquet("data/analysis/t1b_cells.parquet")
ORDER = ["4M","6M","8M","10M","14M","16M","20M","60M","90M","150M","300M","530M","750M","1B"]
def psi1q(nu): return polygamma(1, nu/2.0)/4.0   # Var of log-SD estimate at df=nu
rng = np.random.default_rng(1)
# paper's MC ceiling medians (tab:t1b) for the ceiling-ratio variant
CEIL = {"4M":0.80,"6M":0.79,"8M":0.79,"10M":0.82,"14M":0.82,"16M":0.79,"20M":0.78,
        "60M":0.75,"90M":0.71,"150M":0.73,"300M":0.72,"530M":0.72,"750M":0.75,"1B":0.66}
rows=[]
for p in ORDER:
    s = c[c.params==p].dropna(subset=["x","y"]); s = s[(s.x>0)&(s.y>0)]
    per = []
    for mix, g in s.groupby("data"):
        if len(g) < 5: continue
        per.append(float(np.corrcoef(np.log(g.x), np.log(g.y))[0,1]))
    R_med = float(np.median(per)); n = len(per)
    m = int(s.groupby("data").n_ckpt.median().median()); nu_x, nu_y = m-1, 2
    # per-task medians over 25 recipe cells; sampling var of the median-of-25 ~ (pi/2)*v/25
    lx_t = s.groupby("task").x.apply(lambda v: np.log(v).median())
    ly_t = s.groupby("task").y.apply(lambda v: np.log(v).median())
    k = len(lx_t)
    sa2 = max(lx_t.var(ddof=1) - (np.pi/2)*psi1q(nu_x)/25, 0.01)
    sb2 = max(ly_t.var(ddof=1) - (np.pi/2)*psi1q(nu_y)/25, 0.01)
    A = ((1+psi1q(nu_x)/sa2)*(1+psi1q(nu_y)/sb2))**-0.5
    lam_f = R_med/A; lam_c = R_med/CEIL[p]
    # recipe bootstrap CIs for both
    bs_f, bs_c = [], []
    arr = np.array(per)
    for b_ in range(2000):
        idx = rng.integers(0,n,n)
        bs_f.append(np.median(arr[idx])/A); bs_c.append(np.median(arr[idx])/CEIL[p])
    rows.append(dict(size=p, m=m, k=k, sa=round(np.sqrt(sa2),2), sb=round(np.sqrt(sb2),2),
        A=round(A,3), R=round(R_med,2),
        lam_f=round(lam_f,2), lam_f_lo=round(np.percentile(bs_f,2.5),2), lam_f_hi=round(np.percentile(bs_f,97.5),2),
        lam_c=round(lam_c,2), lam_c_lo=round(np.percentile(bs_c,2.5),2), lam_c_hi=round(np.percentile(bs_c,97.5),2)))
df = pd.DataFrame(rows).set_index("size")
pd.set_option("display.width",220)
print(df.to_string())
print("\nmedian lam_f over sizes:", df.lam_f.median(), " median lam_c:", df.lam_c.median())
print("sizes whose lam_f CI covers 0.9:", sum(df.lam_f_lo<=0.9) , "of 14;", " covers via lam_c:", sum(df.lam_c_lo<=0.9))
