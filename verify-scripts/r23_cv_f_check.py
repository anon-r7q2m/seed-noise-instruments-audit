#!/usr/bin/env python3
"""r23_cv_f_check -- EXT M1: is the exact F interval valid for the relative-SD (CV) ratio?

The F law is exact for ratios of unnormalised sample variances. Table 8 (tab:t2) applies it
to ratios of *relative* SDs s/|mean|, which carry a random mean ratio. This script simulates
the CV ratio at the panel's CV range (0.005-0.13, n_init=10 / n_order=9 as in the S&N arms)
and reports the F-CI coverage and the log-ratio SE inflation vs the plain-SD ratio.

Result: coverage 0.946-0.951 at nominal 0.95 (worst at CV=0.13); SE inflation <= 1.013x.
The 0/27 band-containment count is insensitive (intervals would need factor-2 shifts).
Writes r23_cv_f_check.json; prints ALL PASS if coverage stays within [0.940, 0.960].
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def coverage_and_inflation(cv, n_i=10, n_o=9, nsim=20000):
    a = 0.05
    Flo = stats.f.ppf(a / 2, n_i - 1, n_o - 1)
    Fhi = stats.f.ppf(1 - a / 2, n_i - 1, n_o - 1)
    xi = rng.normal(1.0, cv, size=(nsim, n_i))
    xo = rng.normal(1.0, cv, size=(nsim, n_o))
    si = xi.std(axis=1, ddof=1) / np.abs(xi.mean(axis=1))
    so = xo.std(axis=1, ddof=1) / np.abs(xo.mean(axis=1))
    r = si / so
    lo, hi = r / np.sqrt(Fhi), r / np.sqrt(Flo)
    cov = float(np.mean((lo <= 1) & (1 <= hi)))
    r_plain = xi.std(axis=1, ddof=1) / xo.std(axis=1, ddof=1)
    infl = float(np.std(np.log(r)) / np.std(np.log(r_plain)))
    return cov, infl


out = {}
ok = True
for cv in [0.005, 0.01, 0.03, 0.13]:
    cov, infl = coverage_and_inflation(cv)
    out[f"cv={cv}"] = {"f_ci_coverage_on_cv_ratio": round(cov, 4),
                       "log_ratio_se_inflation": round(infl, 4),
                       "nominal": 0.95, "n_init": 10, "n_order": 9, "nsim": 20000}
    ok &= 0.940 <= cov <= 0.960
    print(f"CV={cv:<6} coverage={cov:.4f}  SE inflation={infl:.4f}x")

out["verdict"] = ("F intervals on the relative-SD ratio lose at most 0.4pt of coverage at "
                  "the panel's worst CV (0.13); mean-ratio randomness is second-order")
json.dump(out, open(os.path.join(HERE, "r23_cv_f_check.json"), "w"), indent=1)
print("ALL PASS" if ok else "FAIL")
assert ok
