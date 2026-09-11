#!/usr/bin/env python3
"""r23_hme_projection -- EXT M2: the HME removes the seed effect g_r estimated from the same
25x3 matrix, then treats s_i^2 = sum_r eps^2/2 as sigma^2 chi2_2/2. Under two-way demeaning
the sufficiency is scaled by (R-1)/R = 0.96 (residual df (R-1)(C-1)=48 over R=25 rows).

Part 1 (simulation, homoscedastic): confirm E[s2] = 0.96 sigma^2 and that the *shape* stays
exponential (Gamma shape ~1) -- i.e. the chi2_2/2 law up to a 0.96 scale.
Part 2 (real refit): refit the 150M macro cell with s2 rescaled by 1/0.96 and report the
seed-noise attribution shift (13.6% -> 13.9%, far inside the printed [8.3, 57.0] interval).

Reads dd_tidy.parquet via NFT_DD; HME code via NFT_HME_SRC. Writes r23_hme_projection.json.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HME_SRC = os.environ.get(
    "NFT_HME_SRC",
    os.path.normpath(os.path.join(HERE, "..", "code", "enhancement3")))
DD = os.environ.get(
    "NFT_DD",
    os.path.normpath(os.path.join(HERE, "..", "data", "analysis", "dd_tidy.parquet")))
sys.path.insert(0, HME_SRC)
import hme
from hme import CellFit, extract_cell, cell_summary

out = {}

# ---- part 1: projected sufficiency law under homoscedastic truth
rng = np.random.default_rng(0)
R, C, NSIM = 25, 3, 4000
s2 = np.empty((NSIM, R))
for b in range(NSIM):
    M = rng.normal(0, 1, (R, C))
    eps = M - M.mean(axis=1, keepdims=True) - (M.mean(axis=0, keepdims=True) - M.mean())
    s2[b] = (eps ** 2).sum(axis=1) / 2.0
m, v = float(s2.mean()), float(s2.var(ddof=1))
out["projection_sim"] = {
    "R": R, "C": C, "nsim": NSIM,
    "E_s2_over_sigma2": round(m, 4), "predicted": (R - 1) / R,
    "gamma_shape": round(m ** 2 / v, 3), "gamma_scale": round(v / m, 4),
    "E_log_s2": round(float(np.log(s2).mean()), 4),
    "Var_log_s2": round(float(np.log(s2).var(ddof=1)), 4),
    "assumed_E_log": -0.5772, "assumed_Var_log": round(float(np.pi ** 2 / 6), 4),
}
p1 = out["projection_sim"]
print(f"part1: E[s2]={p1['E_s2_over_sigma2']} (pred 0.96), Gamma shape {p1['gamma_shape']} "
      f"(exp law = 1), Var[log s2]={p1['Var_log_s2']} (assumed {p1['assumed_Var_log']})")
ok = abs(p1["E_s2_over_sigma2"] - 0.96) < 0.01 and abs(p1["gamma_shape"] - 1.0) < 0.05

# ---- part 2: refit the real 150M macro cell with the 0.96 correction
df = pd.read_parquet(DD)
X, Y, recipes = extract_cell(df, "olmes_10_macro_avg", "150M")
assert X.shape == (25, 3)

fit0 = CellFit(X, Y)
s0 = cell_summary(fit0, 3000, np.random.default_rng(1))


class CellFitCorr(CellFit):
    """Same fit with the sufficiency rescaled by 1/0.96 (projection-corrected law)."""

    def __init__(self, X, Y, zeta_floor=0.02):
        super().__init__(X, Y, zeta_floor)
        self.s2X = self.s2X / 0.96
        self.s2Y = self.s2Y / 0.96
        self.lamX, self.zetX = hme._logvar_moments(self.s2X, zeta_floor)
        self.lamY, self.zetY = hme._logvar_moments(self.s2Y, zeta_floor)
        self.gridX = hme.NoiseGrid(self.s2X, self.lamX, self.zetX)
        self.gridY = hme.NoiseGrid(self.s2Y, self.lamY, self.zetY)


fit1 = CellFitCorr(X, Y)
s1 = cell_summary(fit1, 3000, np.random.default_rng(1))
attr0 = 100 * s0["exp_errors_noise"] / s0["exp_errors"]
attr1 = 100 * s1["exp_errors_noise"] / s1["exp_errors"]
out["refit_150M_macro"] = {
    "attribution_pct_asis": round(attr0, 2),
    "attribution_pct_corrected": round(attr1, 2),
    "shift_pp": round(attr1 - attr0, 2),
    "printed_interval": [8.3, 57.0],
}
print(f"part2: attribution {attr0:.2f}% -> {attr1:.2f}% (shift {attr1-attr0:+.2f}pp, "
      f"interval [8.3, 57.0])")
ok &= abs(attr1 - attr0) < 1.0

json.dump(out, open(os.path.join(HERE, "r23_hme_projection.json"), "w"), indent=1)
print("ALL PASS" if ok else "FAIL")
assert ok
