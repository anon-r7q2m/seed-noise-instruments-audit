#!/usr/bin/env python3
"""Revision-4 W3/Q3: sensitivity of the 150M seed-noise attribution share.

Estimand (abstract): noise_share = exp_errors_noise / exp_errors for the
(olmes_10_macro_avg, 150M) cell. Archived baseline: 0.134507, recipe-cluster
bootstrap CI [0.0832, 0.5701] (enhancement3/results/{cell_estimates,bootstrap_ci}.csv).

Four textbook robustness pieces, all built on the archived hme.py machinery:
  1a. t5 noise likelihood (heavy tails) instead of chi2(df=2) on residuals;
      u-grid keeps VARIANCE semantics (scale^2 = exp(u)*(nu-2)/nu).
  1b. rank model: per-seed rank transform of X and Y before the whole pipeline
      (distribution-free alternative).
  2.  partial-pooling strength scan: zet (log-sigma^2 spread) x {0.25,0.5,1,2,4}.
  3.  leave-one-recipe-out over all 25 recipes.
  4.  1B target-side noise fully in the model: s2Y re-estimated from the full
      3-seed x 3-checkpoint layout (both main effects removed, df=4) instead of
      the across-3-seed df=2 statistic on checkpoint-averaged values.

Deterministic (fixed seed). Pure numpy/pandas/scipy. CPU.
"""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, rankdata

import os
sys.path.insert(0, os.environ.get("NFT_HME_SRC", "code/enhancement3"))
import hme

DD = os.environ.get("NFT_DD", "data/analysis/dd_tidy.parquet")
TASK, SIZE = "olmes_10_macro_avg", "150M"
M = 20000          # posterior draws per fit (LORO uses M_LOO)
M_LOO = 8000
rng = np.random.default_rng(20260828)

ARCH_SHARE, ARCH_LO, ARCH_HI = 0.134507, 0.0832, 0.5701


# ------------------------------------------------------------- generalized pieces
class GridDF(hme.NoiseGrid):
    """NoiseGrid with general df: s2 ~ exp(u) * chi2_df/df -> ll = -(df/2)(u + s2 e^-u)."""

    def __init__(self, s2, lam, zet, df=2, n_grid=400):
        self._df = df
        lo = min(np.log(s2).min() - 8.0, 2 * lam - 8 * max(zet, 0.05))
        hi = max(np.log(s2).max() + 4.0, 2 * lam + 8 * max(zet, 0.05))
        u = np.linspace(lo, hi, n_grid)
        prior = np.exp(-0.5 * ((u - 2 * lam) / (2 * zet)) ** 2)
        ll = -(df / 2.0) * u[None, :] - (df / 2.0) * s2[:, None] * np.exp(-u)[None, :]
        w = prior[None, :] * np.exp(ll - ll.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        self.u, self.w = u, w
        self.cdf = np.cumsum(w, axis=1)
        self.post_mean_sig2 = (w * np.exp(u)[None, :]).sum(axis=1)


class GridT(hme.NoiseGrid):
    """t_nu likelihood on the raw residuals; u keeps VARIANCE semantics."""

    def __init__(self, eps, lam, zet, nu=5, n_grid=400):
        s2 = (eps ** 2).sum(axis=1) / eps.shape[1]
        s2 = np.maximum(s2, 1e-12)
        lo = min(np.log(s2).min() - 8.0, 2 * lam - 8 * max(zet, 0.05))
        hi = max(np.log(s2).max() + 4.0, 2 * lam + 8 * max(zet, 0.05))
        u = np.linspace(lo, hi, n_grid)
        prior = np.exp(-0.5 * ((u - 2 * lam) / (2 * zet)) ** 2)
        # variance = exp(u) -> t scale^2 = exp(u)*(nu-2)/nu
        scale = np.sqrt(np.exp(u) * (nu - 2.0) / nu)
        ll = t_dist.logpdf(eps[:, :, None], df=nu, scale=scale[None, None, :]).sum(axis=1)
        w = prior[None, :] * np.exp(ll - ll.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        self.u, self.w = u, w
        self.cdf = np.cumsum(w, axis=1)
        self.post_mean_sig2 = (w * np.exp(u)[None, :]).sum(axis=1)


class FitVariant(hme.CellFit):
    """CellFit with pluggable noise grid and zeta scaling."""

    ZSCALE = 1.0
    GRID = "chi2"      # 'chi2' (df=2), 't5'
    DFY = None         # piece 4: override df for Y grid
    S2Y_OVERRIDE = None

    def __init__(self, X, Y, zeta_floor=0.02):
        super().__init__(X, Y, zeta_floor)
        zx = max(self.zetX * self.ZSCALE, 0.02)
        zy = max(self.zetY * self.ZSCALE, 0.02)
        if self.GRID == "t5":
            self.gridX = GridT(self.epsX, self.lamX, zx)
            self.gridY = GridT(self.epsY, self.lamY, zy)
        else:
            dfy = self.DFY or 2
            self.gridX = GridDF(self.s2X, self.lamX, zx, df=2)
            s2y = self.S2Y_OVERRIDE if self.S2Y_OVERRIDE is not None else self.s2Y
            self.gridY = GridDF(s2y, self.lamY, zy, df=dfy)
        mE_X = self.gridX.post_mean_sig2.mean()
        mE_Y = self.gridY.post_mean_sig2.mean()
        self.omX2 = max(0.0, self.xbar.var(ddof=1) - mE_X / 3.0)
        self.omY2 = max(0.0, self.ybar.var(ddof=1) - mE_Y / 3.0)
        self.omX, self.omY = np.sqrt(self.omX2), np.sqrt(self.omY2)
        if self.omX > 0 and self.omY > 0:
            cov = np.cov(self.xbar, self.ybar, ddof=1)[0, 1]
            self.rho = float(np.clip(cov / (self.omX * self.omY), -1.0, 1.0))
        else:
            self.rho = 0.0


def share_of(fit, m):
    s = hme.cell_summary(fit, m, rng)
    return s["exp_errors_noise"] / max(s["exp_errors"], 1e-12)


def rank_per_seed(MX):
    return np.apply_along_axis(rankdata, 0, MX).astype(float)


# ------------------------------------------------------------- data
d = pd.read_parquet(DD)
X, Y, recipes = hme.extract_cell(d, TASK, SIZE)
n = len(recipes)
print(f"cell ({TASK}, {SIZE}): {n} recipes x {X.shape[1]} seeds", flush=True)

out = {"cell": f"{TASK}/{SIZE}", "n_recipes": n,
       "archived": {"share": ARCH_SHARE, "ci": [ARCH_LO, ARCH_HI]}}

# ------------------------------------------------------------- baseline reproduction
base = hme.CellFit(X, Y)
sh_base = share_of(base, M)
print(f"[baseline]            share = {sh_base:.4f}   (archived {ARCH_SHARE})", flush=True)
out["baseline_recomputed"] = sh_base

# ------------------------------------------------------------- piece 1a: t5 likelihood
class FitT5(FitVariant):
    GRID = "t5"

sh_t5 = share_of(FitT5(X, Y), M)
print(f"[1a t5-likelihood]    share = {sh_t5:.4f}", flush=True)
out["piece1a_t5"] = sh_t5

# ------------------------------------------------------------- piece 1b: rank model
Xr, Yr = rank_per_seed(X), rank_per_seed(Y)
sh_rank = share_of(hme.CellFit(Xr, Yr), M)
print(f"[1b rank model]       share = {sh_rank:.4f}", flush=True)
out["piece1b_rank"] = sh_rank

# ------------------------------------------------------------- piece 2: pooling scan
out["piece2_pooling_scan"] = {}
for c in (0.25, 0.5, 1.0, 2.0, 4.0):
    cls = type(f"FitZ{int(c*100)}", (FitVariant,), {"ZSCALE": c})
    sh = share_of(cls(X, Y), M)
    out["piece2_pooling_scan"][str(c)] = sh
    print(f"[2 pooling zeta x{c:<4}] share = {sh:.4f}", flush=True)

# ------------------------------------------------------------- piece 3: LORO
loo = []
for i in range(n):
    keep = [j for j in range(n) if j != i]
    sh = share_of(hme.CellFit(X[keep], Y[keep]), M_LOO)
    loo.append(float(sh))
    print(f"[3 LORO drop {recipes[i][:28]:<28}] share = {sh:.4f}", flush=True)
out["piece3_loro"] = {"min": min(loo), "median": float(np.median(loo)), "max": max(loo),
                      "all": loo, "dropped_recipe": recipes}
print(f"[3 LORO]              min/med/max = {min(loo):.4f} / {np.median(loo):.4f} / {max(loo):.4f}",
      flush=True)

# ------------------------------------------------------------- piece 4: target-side noise df 2 -> 4
# full 3-seed x 3-ckpt layout at 1B for the same recipe set
g = d[(d.params == "1B") & (d.task == TASK)]
per = {}
for d_, gd in g.groupby("data"):
    if d_ not in recipes:
        continue
    seeds = sorted(gd.seed.unique())
    step_sets = [set(gd[gd.seed == s].step) for s in seeds]
    common = sorted(set.intersection(*step_sets))
    if len(common) < 3:
        continue
    lw = common[-3:]
    per[d_] = np.array([[gd[(gd.seed == s) & (gd.step == st)]["primary_metric"].iloc[0]
                         for st in lw] for s in seeds])          # (seed, ckpt)
recipes4 = sorted(per)
assert recipes4 == recipes, "recipe set drifted"
V = np.array([per[r] for r in recipes4])                          # (n, 3 seeds, 3 ckpts)
# remove seed + ckpt main effects per recipe; residual df = (3-1)(3-1) = 4
Vr = V - V.mean(axis=2, keepdims=True)       # remove seed means (over ckpts)
Vr = Vr - Vr.mean(axis=1, keepdims=True)     # remove ckpt-position means of the residual
s2_cell = (Vr ** 2).sum(axis=(1, 2)) / 4.0   # per-recipe single-cell noise var, df=4
s2Y_aug = np.maximum(s2_cell / 3.0, 1e-12)   # variance of one seed's ckpt-mean

class FitY4(FitVariant):
    DFY = 4
    S2Y_OVERRIDE = s2Y_aug

sh_y4 = share_of(FitY4(X, Y), M)
print(f"[4 target-noise df=4] share = {sh_y4:.4f}   (baseline s2Y med {np.median(base.s2Y):.3e} -> aug med {np.median(s2Y_aug):.3e})",
      flush=True)
out["piece4_target_noise_df4"] = sh_y4

# ------------------------------------------------------------- dump
with open("r4_attribution_sensitivity.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote r4_attribution_sensitivity.json")
