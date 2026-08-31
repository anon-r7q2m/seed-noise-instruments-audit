#!/usr/bin/env python3
"""hme.py -- hierarchical measurement-error model for T3 (enhancement #3).

Implements model-spec.md: EB moment fits, grid posteriors for per-recipe noise,
MC posterior integration for pair estimands (p_flip, p_err, excess risk),
recipe-cluster bootstrap, task-cluster bootstrap, parametric-bootstrap validation.
All pure numpy/pandas/scipy. No MCMC.
"""
import numpy as np
from scipy.special import ndtr  # Phi
from scipy.stats import shapiro

EULER_GAMMA = 0.5772156649015329
SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------- extraction
def extract_cell(df, task, size, metric="primary_metric"):
    """Replicate 04_t3_decision_replay.py extraction exactly.
    Returns X (n_recipes x 3 proxy, final common ckpt), Y (n_recipes x 3 target 1B,
    per-seed mean of last-3 common ckpts), recipe names.
    """
    def scores_at(sz, avg_last):
        g = df[(df.params == sz) & (df.task == task)]
        out = {}
        for d_, gd in g.groupby("data"):
            seeds = sorted(gd.seed.unique())
            step_sets = [set(gd[gd.seed == s].step) for s in seeds]
            common = sorted(set.intersection(*step_sets))
            if len(common) < avg_last:
                continue
            lw = common[-avg_last:]
            out[d_] = {s: gd[(gd.seed == s) & (gd.step.isin(lw))][metric].mean()
                       for s in seeds}
        return out  # recipe -> {seed: score}

    tgt = scores_at("1B", 3)
    pred = scores_at(size, 1)
    recipes = sorted(set(pred) & set(tgt))
    seeds_p = sorted({s for r in recipes for s in pred[r]})
    seeds_t = sorted({s for r in recipes for s in tgt[r]})
    X = np.array([[pred[r][s] for s in seeds_p] for r in recipes])
    Y = np.array([[tgt[r][s] for s in seeds_t] for r in recipes])
    return X, Y, recipes


def remove_seed_effects(M):
    """Two-way decomposition M_ir = a_i + g_r + eps; return (row_means, g, eps)."""
    a = M.mean(axis=1, keepdims=True)
    g = M.mean(axis=0, keepdims=True) - M.mean()
    eps = M - a - g
    return a.ravel(), g.ravel(), eps


# ---------------------------------------------------------------- EB fit
class CellFit:
    """EB fit for one (task, size) cell."""

    def __init__(self, X, Y, zeta_floor=0.02):
        self.n, self.nseeds = X.shape
        aX, gX, eX = remove_seed_effects(X)
        aY, gY, eY = remove_seed_effects(Y)
        self.xbar, self.ybar = aX, aY
        self.gX, self.gY = gX, gY
        self.s2X = (eX ** 2).sum(axis=1) / 2.0  # df=2
        self.s2Y = (eY ** 2).sum(axis=1) / 2.0
        # guard against zero variance (degenerate cells)
        self.s2X = np.maximum(self.s2X, 1e-12)
        self.s2Y = np.maximum(self.s2Y, 1e-12)
        self.lamX, self.zetX = _logvar_moments(self.s2X, zeta_floor)
        self.lamY, self.zetY = _logvar_moments(self.s2Y, zeta_floor)
        # per-recipe noise posteriors on u = log sigma^2 grid
        self.gridX = NoiseGrid(self.s2X, self.lamX, self.zetX)
        self.gridY = NoiseGrid(self.s2Y, self.lamY, self.zetY)
        # latent scale: Var(xbar) = omega^2 + mean(E[sig^2|s2])/3
        mE_X = self.gridX.post_mean_sig2.mean()
        mE_Y = self.gridY.post_mean_sig2.mean()
        self.omX2 = max(0.0, self.xbar.var(ddof=1) - mE_X / 3.0)
        self.omY2 = max(0.0, self.ybar.var(ddof=1) - mE_Y / 3.0)
        self.omX, self.omY = np.sqrt(self.omX2), np.sqrt(self.omY2)
        self.mX, self.mY = self.xbar.mean(), self.ybar.mean()
        if self.omX > 0 and self.omY > 0:
            cov = np.cov(self.xbar, self.ybar, ddof=1)[0, 1]
            self.rho = float(np.clip(cov / (self.omX * self.omY), -1.0, 1.0))
        else:
            self.rho = 0.0
        self.epsX, self.epsY = eX, eY

    # ---- posterior draws: per recipe M samples of (mu, nu, sigX2, sigY2)
    def draw_posterior(self, M, rng):
        uX = self.gridX.sample_u(M, rng)   # (n, M)
        uY = self.gridY.sample_u(M, rng)
        sigX2, sigY2 = np.exp(uX), np.exp(uY)
        mu = np.empty_like(uX)
        nu = np.empty_like(uY)
        om = np.array([[self.omX2, self.rho * self.omX * self.omY],
                       [self.rho * self.omX * self.omY, self.omY2]])
        if self.omX2 <= 0 or self.omY2 <= 0 or abs(self.rho) >= 1.0:
            # diagonal fallback
            mu = _shrink_draw(self.xbar, sigX2, self.omX2, self.mX, rng)
            nu = _shrink_draw(self.ybar, sigY2, self.omY2, self.mY, rng)
        else:
            om_inv = np.linalg.inv(om)
            # vectorized 2x2 posterior update per (recipe, draw)
            a = om_inv[0, 0] + 3.0 / sigX2
            b = om_inv[0, 1] * np.ones_like(sigX2)
            c = om_inv[1, 1] + 3.0 / sigY2
            det = a * c - b * b
            # posterior precision P = [[a,b],[b,c]]; cov = inv
            V00, V01, V11 = c / det, -b / det, a / det
            # posterior mean = V (om_inv m + D_inv obs)
            r0 = om_inv[0, 0] * self.mX + om_inv[0, 1] * self.mY + 3.0 * self.xbar[:, None] / sigX2
            r1 = om_inv[1, 0] * self.mX + om_inv[1, 1] * self.mY + 3.0 * self.ybar[:, None] / sigY2
            m0 = V00 * r0 + V01 * r1
            m1 = V01 * r0 + V11 * r1
            # Cholesky of [[V00,V01],[V01,V11]]
            L00 = np.sqrt(np.maximum(V00, 1e-30))
            L10 = V01 / L00
            L11 = np.sqrt(np.maximum(V11 - L10 ** 2, 1e-30))
            z0 = rng.standard_normal(sigX2.shape)
            z1 = rng.standard_normal(sigX2.shape)
            mu = m0 + L00 * z0
            nu = m1 + L10 * z0 + L11 * z1
        return mu, nu, sigX2, sigY2


def _shrink_draw(xbar, sig2, om2, m, rng):
    w = om2 / (om2 + sig2 / 3.0) if om2 > 0 else np.zeros_like(sig2)
    mean = w * xbar[:, None] + (1 - w) * m
    var = w * sig2 / 3.0
    return mean + np.sqrt(np.maximum(var, 0)) * rng.standard_normal(sig2.shape)


def _logvar_moments(s2, zeta_floor):
    """s2 ~ sig^2 * chi2_2/2, log sig ~ N(lam, zeta^2)."""
    l = np.log(s2)
    lam = (l.mean() + EULER_GAMMA) / 2.0
    zet2 = max(0.0, (l.var(ddof=1) - np.pi ** 2 / 6.0) / 4.0)
    return lam, max(np.sqrt(zet2), zeta_floor)


class NoiseGrid:
    """Posterior of u = log sigma^2 given s^2 (df=2) under lognormal prior."""

    def __init__(self, s2, lam, zet, n_grid=400):
        lo = min(np.log(s2).min() - 8.0, 2 * lam - 8 * max(zet, 0.05))
        hi = max(np.log(s2).max() + 4.0, 2 * lam + 8 * max(zet, 0.05))
        u = np.linspace(lo, hi, n_grid)
        prior = np.exp(-0.5 * ((u - 2 * lam) / (2 * zet)) ** 2)
        # loglik: -u - s2*exp(-u)  (per recipe)
        ll = -u[None, :] - s2[:, None] * np.exp(-u)[None, :]
        w = prior[None, :] * np.exp(ll - ll.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        self.u = u
        self.w = w
        self.cdf = np.cumsum(w, axis=1)
        self.post_mean_sig2 = (w * np.exp(u)[None, :]).sum(axis=1)

    def sample_u(self, M, rng):
        n = self.w.shape[0]
        q = rng.random((n, M))
        # inverse-cdf per draw via searchsorted
        idx = np.empty((n, M), dtype=int)
        for i in range(n):
            idx[i] = np.searchsorted(self.cdf[i], q[i])
        idx = np.clip(idx, 0, len(self.u) - 1)
        return self.u[idx]


# ---------------------------------------------------------------- estimands
def fast_estimands(fit, M, rng):
    """Single-draw, float32 version of all cell estimands (for bootstrap reps).
    Same definitions as cell_summary; one shared posterior draw for pair+pick."""
    mu, nu, sigX2, sigY2 = fit.draw_posterior(M, rng)
    mu = mu.astype(np.float32); nu = nu.astype(np.float32)
    sigX2 = sigX2.astype(np.float32)
    n = fit.n
    ii, jj = np.triu_indices(n, k=1)
    dX = mu[ii] - mu[jj]              # (P, M) float32
    dY = nu[ii] - nu[jj]
    tau = np.sqrt((sigX2[ii] + sigX2[jj]) / np.float32(3.0))
    d = np.sign(fit.xbar[ii] - fit.xbar[jj]).astype(np.float32)
    d[d == 0] = 1.0
    d_col = d[:, None]
    sign_dY = np.sign(dY)
    sign_dX = np.sign(dX)
    err_draw = (sign_dY != d_col)
    discord = (sign_dX != sign_dY)
    p_flip = ndtr(-d_col * dX / tau).mean(axis=1)
    p_err = err_draw.mean(axis=1)
    p_err_discord = (err_draw & discord).mean(axis=1)
    p_err_noise = (err_draw & ~discord).mean(axis=1)
    tau1 = np.sqrt(sigX2[ii] + sigX2[jj])
    p_err_pros3 = ndtr(-sign_dY * dX / tau).mean(axis=1)
    p_err_pros1 = ndtr(-sign_dY * dX / tau1).mean(axis=1)
    abs_dY = np.abs(dY)
    er_real = (abs_dY * err_draw).mean(axis=1)
    e_abs_dY = abs_dY.mean(axis=1)
    tot_signal = e_abs_dY.sum()
    nP = len(p_flip)
    # pick risk from same draws
    nup = nu.T
    best = nup.max(axis=1)
    i_hat = int(np.argmax(fit.xbar))
    er_pick_real = float((best - nup[:, i_hat]).mean())
    pick_lat = np.argmax(mu, axis=0)
    er_pick_latent = float((best - nup[np.arange(M), pick_lat]).mean())
    exp_err = float(p_err.sum())
    return dict(
        mean_p_flip=float(p_flip.mean()),
        share_pflip_gt25=float((p_flip > 0.25).mean()),
        exp_errors=exp_err,
        exp_errors_discord=float(p_err_discord.sum()),
        exp_errors_noise=float(p_err_noise.sum()),
        exp_errors_pros3=float(p_err_pros3.sum()),
        exp_errors_pros1=float(p_err_pros1.sum()),
        er_real_share=float(er_real.sum() / tot_signal) if tot_signal > 0 else np.nan,
        er_real_sum=float(er_real.sum()), signal_sum=float(tot_signal),
        n_pairs=int(nP),
        er_pick_real=er_pick_real, er_pick_latent=er_pick_latent,
        nas_weighted=float((p_err * p_flip).sum() / max(exp_err, 1e-12)),
    )


def pair_estimates(fit, M, rng, d_obs_sign=None):
    """MC posterior integration of pair estimands.
    Returns dict of (P,) arrays over unordered pairs i<j."""
    mu, nu, sigX2, sigY2 = fit.draw_posterior(M, rng)
    n = fit.n
    ii, jj = np.triu_indices(n, k=1)
    dX = mu[ii] - mu[jj]              # (P, M)
    dY = nu[ii] - nu[jj]
    tau = np.sqrt((sigX2[ii] + sigX2[jj]) / 3.0)
    # observed decision sign (fixed): from xbar
    if d_obs_sign is None:
        d = np.sign(fit.xbar[ii] - fit.xbar[jj])
        d[d == 0] = 1.0
    else:
        d = d_obs_sign
    d_col = d[:, None]
    # p_flip: P(fresh redraw D' = dX + e' has sign != d) = Phi(-d*dX/tau)
    p_flip = ndtr(-d_col * dX / tau).mean(axis=1)
    # single-seed instrument flip prob (noise sig not averaged over 3)
    tau1 = np.sqrt(sigX2[ii] + sigX2[jj])
    p_flip_single = ndtr(-d_col * dX / tau1).mean(axis=1)
    # prospective error prob of fresh instruments vs latent target
    p_err_pros3 = ndtr(-np.sign(dY) * dX / tau).mean(axis=1)   # fresh 3-seed mean
    p_err_pros1 = ndtr(-np.sign(dY) * dX / tau1).mean(axis=1)  # fresh single seed
    # p_err: P(sign(dY) != d), decomposed:
    #   discord: latent proxy and latent target disagree (irreducible by more seeds)
    #   noise:   latents agree but the observed (noisy) decision departed from them
    sign_dY = np.sign(dY)
    sign_dX = np.sign(dX)
    err_draw = (sign_dY != d_col).astype(float)
    discord = (sign_dX != sign_dY)
    p_err = err_draw.mean(axis=1)
    p_err_discord = (err_draw * discord).mean(axis=1)
    p_err_noise = (err_draw * ~discord).mean(axis=1)
    abs_dY = np.abs(dY)
    er_real = (abs_dY * err_draw).mean(axis=1)
    # prospective: P(fresh decision wrong vs latent target) = Phi(-sign(dY)*dX/tau)
    p_wrong_pros = ndtr(-np.sign(dY) * dX / tau)
    er_pros = (abs_dY * p_wrong_pros).mean(axis=1)
    e_abs_dY = abs_dY.mean(axis=1)
    return dict(p_flip=p_flip, p_err=p_err, er_real=er_real, er_pros=er_pros,
                e_abs_dY=e_abs_dY, p_flip_single=p_flip_single,
                p_err_discord=p_err_discord, p_err_noise=p_err_noise,
                p_discord=discord.astype(float).mean(axis=1),
                p_err_pros3=p_err_pros3, p_err_pros1=p_err_pros1)


def pick_risk(fit, M, rng):
    """Argmax excess risk: realized pick by xbar; prospective pick by fresh redraw."""
    mu, nu, sigX2, sigY2 = fit.draw_posterior(M, rng)
    nup = nu.T  # (M, n)
    best = nup.max(axis=1)
    i_hat = int(np.argmax(fit.xbar))
    er_pick_real = float((best - nup[:, i_hat]).mean())
    # prospective: pick argmax of mu + fresh noise
    fresh = mu + np.sqrt(sigX2 / 3.0) * rng.standard_normal(sigX2.shape)
    pick = np.argmax(fresh, axis=0)         # (M,)
    er_pick_pros = float((best - nup[np.arange(M), pick]).mean())
    # oracle-achievable floor: pick argmax of latent mu
    pick_lat = np.argmax(mu, axis=0)
    er_pick_latent = float((best - nup[np.arange(M), pick_lat]).mean())
    return er_pick_real, er_pick_pros, er_pick_latent


def cell_summary(fit, M, rng, keep_pairs=False):
    pe = pair_estimates(fit, M, rng)
    er_pick_real, er_pick_pros, er_pick_latent = pick_risk(fit, M, rng)
    tot_signal = pe["e_abs_dY"].sum()
    out = dict(
        n_recipes=fit.n, n_pairs=len(pe["p_flip"]),
        mX=fit.mX, omX=fit.omX, mY=fit.mY, omY=fit.omY, rho=fit.rho,
        lamX=fit.lamX, zetX=fit.zetX, lamY=fit.lamY, zetY=fit.zetY,
        med_sigX=float(np.exp(fit.lamX)), med_sigY=float(np.exp(fit.lamY)),
        mean_p_flip=float(pe["p_flip"].mean()),
        mean_p_flip_single=float(pe["p_flip_single"].mean()),
        share_pflip_gt10=float((pe["p_flip"] > 0.10).mean()),
        share_pflip_gt25=float((pe["p_flip"] > 0.25).mean()),
        exp_errors=float(pe["p_err"].sum()),
        exp_errors_discord=float(pe["p_err_discord"].sum()),
        exp_errors_noise=float(pe["p_err_noise"].sum()),
        exp_discord_pairs=float(pe["p_discord"].sum()),
        exp_errors_pros3=float(pe["p_err_pros3"].sum()),
        exp_errors_pros1=float(pe["p_err_pros1"].sum()),
        er_real=float(pe["er_real"].sum()),
        er_pros=float(pe["er_pros"].sum()),
        er_real_share=float(pe["er_real"].sum() / tot_signal) if tot_signal > 0 else np.nan,
        er_pros_share=float(pe["er_pros"].sum() / tot_signal) if tot_signal > 0 else np.nan,
        er_per_decision=float(pe["er_real"].sum() / len(pe["p_flip"])),
        er_pick_real=er_pick_real, er_pick_pros=er_pick_pros,
        er_pick_latent=er_pick_latent,
        # noise-admitting share of error mass (model-based, V3)
        nas_c10=float((pe["p_err"] * (pe["p_flip"] > 0.10)).sum() /
                      max(pe["p_err"].sum(), 1e-12)),
        nas_c25=float((pe["p_err"] * (pe["p_flip"] > 0.25)).sum() /
                      max(pe["p_err"].sum(), 1e-12)),
        nas_weighted=float((pe["p_err"] * pe["p_flip"]).sum() /
                           max(pe["p_err"].sum(), 1e-12)),
    )
    if keep_pairs:
        out["_pairs"] = pe
    return out


# ---------------------------------------------------------------- descriptive replay (for validation)
def descriptive_stats(X, Y):
    """The exact descriptive pipeline of 04_t3_decision_replay.py on given matrices.
    Returns dict(n_pairs, errors, err_any_unstable, share_err_any_unstable,
    share_pairs_unstable)."""
    n = X.shape[0]
    xbar, ybar = X.mean(axis=1), Y.mean(axis=1)
    ii, jj = np.triu_indices(n, k=1)
    dt = ybar[ii] - ybar[jj]
    keep = dt != 0
    ii, jj, dt = ii[keep], jj[keep], dt[keep]
    dm = xbar[ii] - xbar[jj]
    correct = (dm * dt) > 0
    # per-seed decisions
    ss = np.sign(X[ii] - X[jj])          # (P, 3)
    small_unstable = (ss == ss[:, :1]).sum(axis=1) < 3
    ts = np.sign(Y[ii] - Y[jj])
    tgt_unstable = (ts == ts[:, :1]).sum(axis=1) < 3
    any_unstable = small_unstable | tgt_unstable
    n_pairs = len(dt)
    errors = int((~correct).sum())
    err_un = int((~correct & any_unstable).sum())
    return dict(n_pairs=n_pairs, errors=errors, err_any_unstable=err_un,
                share_err_any_unstable=err_un / errors if errors else np.nan,
                share_pairs_unstable=float(any_unstable.mean()),
                acc=float(correct.mean()))


# ---------------------------------------------------------------- simulation from fitted model
def simulate_cell(fit, rng, with_seed_effects=True):
    """Draw a fake (X, Y) dataset from the fitted hierarchical model."""
    n = fit.n
    om = np.array([[fit.omX2, fit.rho * fit.omX * fit.omY],
                   [fit.rho * fit.omX * fit.omY, fit.omY2]])
    if fit.omX2 > 0 and fit.omY2 > 0 and abs(fit.rho) < 1:
        L = np.linalg.cholesky(om)
        z = rng.standard_normal((n, 2))
        lat = np.array([fit.mX, fit.mY]) + z @ L.T
        mu, nu = lat[:, 0], lat[:, 1]
    else:
        mu = fit.mX + fit.omX * rng.standard_normal(n)
        nu = fit.mY + fit.omY * rng.standard_normal(n)
    sigX = np.exp(fit.lamX + fit.zetX * rng.standard_normal(n))
    sigY = np.exp(fit.lamY + fit.zetY * rng.standard_normal(n))
    X = mu[:, None] + sigX[:, None] * rng.standard_normal((n, 3))
    Y = nu[:, None] + sigY[:, None] * rng.standard_normal((n, 3))
    if with_seed_effects:
        X = X + fit.gX[None, :]
        Y = Y + fit.gY[None, :]
    truth = dict(mu=mu, nu=nu, sigX=sigX, sigY=sigY)
    return X, Y, truth


def true_pflip_share_of_errors(X, Y, truth, thresh=0.25):
    """In-simulation ground truth: among pairs whose observed decision is wrong
    (vs observed 3-seed target mean, as the descriptive pipeline defines errors),
    the share whose TRUE redraw flip probability exceeds thresh."""
    n = X.shape[0]
    xbar, ybar = X.mean(axis=1), Y.mean(axis=1)
    ii, jj = np.triu_indices(n, k=1)
    dt = ybar[ii] - ybar[jj]
    keep = dt != 0
    ii, jj, dt = ii[keep], jj[keep], dt[keep]
    dm = xbar[ii] - xbar[jj]
    wrong = (dm * dt) < 0
    dX_true = truth["mu"][ii] - truth["mu"][jj]
    tau_true = np.sqrt((truth["sigX"][ii] ** 2 + truth["sigX"][jj] ** 2) / 3.0)
    d = np.sign(dm)
    d[d == 0] = 1
    pflip_true = ndtr(-d * dX_true / tau_true)
    if wrong.sum() == 0:
        return np.nan, np.nan
    return float((pflip_true[wrong] > thresh).mean()), float(pflip_true[wrong].mean())
