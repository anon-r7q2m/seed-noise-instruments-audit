#!/usr/bin/env python3
"""r28_hme_full_correction -- EXT M2 (second external round): two consistency checks on the HME.

(i) Heteroscedastic projection correction. After two-way demeaning of the R x C matrix
(R=25 recipes, C=3 seeds), the per-recipe sufficiency obeys
    E[s_i^2] = (1 - 2/R) sigma_i^2 + (1/R^2) sum_j sigma_j^2
(exact expansion for the usual two-way projection; reduces to (R-1)/R = 0.96 under
homoscedasticity). The uniform 0.96 correction of r23 is therefore per-recipe wrong under
heteroscedasticity. We do a one-step correction: fit as-is, take posterior mean sigma_i^2,
form per-recipe bias factors b_i = 0.92 + 0.04*mean(sigma^2)/sigma_i^2, rescale s_i^2 <-
s_i^2 / b_i, refit, and report the attribution shift at the 150M macro cell.

(ii) Same-target refit. The paper's HME smooths the 1B target over the last three common
checkpoints (disclosed); here we refit with Y = the final-checkpoint value (the replay's own
target) and report the attribution.

Writes r28_hme_full_correction.json. Pure CPU. NFT_DD / NFT_HME_SRC overridable.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HME_SRC = os.environ.get("NFT_HME_SRC", os.path.normpath(os.path.join(HERE, "..", "code", "enhancement3")))
DD = os.environ.get("NFT_DD", os.path.normpath(os.path.join(HERE, "..", "data", "analysis", "dd_tidy.parquet")))
sys.path.insert(0, HME_SRC)
import hme
from hme import CellFit, extract_cell, cell_summary

df = pd.read_parquet(DD)
X, Y3, recipes = extract_cell(df, "olmes_10_macro_avg", "150M")
assert X.shape == (25, 3)

# ---- same-target Y: final common checkpoint per seed (no smoothing)
# proxy side X uses the final common checkpoint of the proxy scale already; build Y_final
# by the same common-step logic on the 1B rows
g = df[(df.params == "1B") & (df.task == "olmes_10_macro_avg")]
Yf = np.full((25, 3), np.nan)
for ii, r in enumerate(recipes):
    gr = g[g.data == r]
    seeds = sorted(gr.seed.unique())[:3]
    common = sorted(set.intersection(*[set(gr[gr.seed == s].step) for s in seeds]))
    fin = common[-1]
    for jj, s in enumerate(seeds):
        Yf[ii, jj] = gr[(gr.seed == s) & (gr.step == fin)]["primary_metric"].mean()
assert np.isfinite(Yf).all()

results = {}

def run(Xm, Ym, seed=1, label=""):
    fit = CellFit(Xm, Ym)
    s = cell_summary(fit, 3000, np.random.default_rng(seed))
    attr = 100 * s["exp_errors_noise"] / s["exp_errors"]
    print(f"{label}: attribution {attr:.2f}%")
    return attr, fit

attr_base, fit0 = run(X, Y3, label="as-is (smoothed target)            ")
results["asis_smoothed_target"] = round(attr_base, 2)

# (i) heteroscedastic one-step correction
sigX = fit0.gridX.post_mean_sig2
sigY = fit0.gridY.post_mean_sig2
bX = 0.92 + 0.04 * sigX.mean() / sigX
bY = 0.92 + 0.04 * sigY.mean() / sigY


class CellFitHet(CellFit):
    def __init__(self, X, Y, bX, bY, zeta_floor=0.02):
        super().__init__(X, Y, zeta_floor)
        self.s2X = self.s2X / bX
        self.s2Y = self.s2Y / bY
        self.lamX, self.zetX = hme._logvar_moments(self.s2X, zeta_floor)
        self.lamY, self.zetY = hme._logvar_moments(self.s2Y, zeta_floor)
        self.gridX = hme.NoiseGrid(self.s2X, self.lamX, self.zetX)
        self.gridY = hme.NoiseGrid(self.s2Y, self.lamY, self.zetY)


fit1 = CellFitHet(X, Y3, bX, bY)
s1 = cell_summary(fit1, 3000, np.random.default_rng(1))
attr_het = 100 * s1["exp_errors_noise"] / s1["exp_errors"]
print(f"heteroscedastic-corrected        : attribution {attr_het:.2f}%")
results["heteroscedastic_corrected"] = round(attr_het, 2)
results["bias_factor_range_X"] = [round(float(bX.min()), 4), round(float(bX.max()), 4)]

# (ii) same-target (final checkpoint) refit, as-is law
attr_same, _ = run(X, Yf, label="same-target (final ckpt), as-is law  ")
results["asis_same_target"] = round(attr_same, 2)

# both corrections together
fit2 = CellFitHet(X, Yf, bX, bY)
s2 = cell_summary(fit2, 3000, np.random.default_rng(1))
attr_both = 100 * s2["exp_errors_noise"] / s2["exp_errors"]
print(f"both corrections                 : attribution {attr_both:.2f}%")
results["both_corrections"] = round(attr_both, 2)

results["printed"] = {"point_pct": 13.5, "interval": [8.3, 57.0]}
results["note"] = ("E[s2] = (1-2/R) sigma_i^2 + (1/R^2) sum_j sigma_j^2; R=25. "
                   "One-step correction via posterior means of the as-is fit.")
json.dump(results, open(os.path.join(HERE, "r28_hme_full_correction.json"), "w"), indent=1)
print("wrote r28_hme_full_correction.json")
