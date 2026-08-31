#!/usr/bin/env python3
"""run_all.py -- staged driver for enhancement #3 (T3 hierarchical ME model).

Stages (incremental writes after each):
  extract     -> results/arrays.npz + results/observed_descriptive.csv
  fit         -> results/cell_estimates.csv + results/pair_estimates.parquet
  diagnostics -> results/diagnostics.json
  bootstrap   -> results/bootstrap_ci.csv (recipe-cluster per cell +
                 pooled recipe-cluster per size + task-cluster pooled)
  validate    -> results/validation.json (parametric bootstrap vs descriptive)

Usage: nice -n 19 python3 run_all.py <stage> [--workers N]
"""
import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hme import (CellFit, extract_cell, cell_summary, descriptive_stats,
                 simulate_cell, true_pflip_share_of_errors)

BASE = "./work-20260825/rank4-paper/enhancement3"
RES = f"{BASE}/results"
DD = "data/analysis/dd_tidy.parquet"
TASKS10 = ["arc_challenge", "arc_easy", "boolq", "csqa", "hellaswag",
           "mmlu", "openbookqa", "piqa", "socialiqa", "winogrande"]
TASKS = ["olmes_10_macro_avg"] + TASKS10
SIZES = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M",
         "150M", "300M", "530M", "750M"]
MACRO = "olmes_10_macro_avg"
os.makedirs(RES, exist_ok=True)


def _load():
    return pd.read_parquet(DD)


# ------------------------------------------------------------------ extract
def stage_extract():
    df = _load()
    store, index, obs_rows = {}, [], []
    for task in TASKS:
        for size in SIZES:
            X, Y, recipes = extract_cell(df, task, size)
            key = f"{task}|{size}"
            store[key + "|X"] = X
            store[key + "|Y"] = Y
            index.append(dict(key=key, task=task, size=size,
                              n_recipes=X.shape[0], n_seeds=X.shape[1]))
            st = descriptive_stats(X, Y)
            obs_rows.append(dict(task=task, size=size, **st))
            if task == TASKS[0] and size == SIZES[0]:
                store["recipes"] = np.array(recipes)
    np.savez_compressed(f"{RES}/arrays.npz", **store)
    pd.DataFrame(index).to_csv(f"{RES}/cells_index.csv", index=False)
    pd.DataFrame(obs_rows).to_csv(f"{RES}/observed_descriptive.csv", index=False)
    print("extract done:", len(index), "cells")


def _arrays():
    z = np.load(f"{RES}/arrays.npz", allow_pickle=False)
    return z


# ------------------------------------------------------------------ fit
def _fit_one(key, task, size, z, M, seed, keep_pairs=True):
    rng = np.random.default_rng(seed)
    fit = CellFit(z[key + "|X"], z[key + "|Y"])
    s = cell_summary(fit, M, rng, keep_pairs=keep_pairs)
    s["task"], s["size"] = task, size
    return s


def stage_fit(M=4000, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    z = _arrays()
    idx = pd.read_csv(f"{RES}/cells_index.csv")
    jobs = [(r.key, r.task, r.size, M, 1000 + k) for k, r in
            enumerate(idx.itertuples())]
    rows, pair_rows = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s in ex.map(_fit_star, jobs):
            pe = s.pop("_pairs")
            rows.append(s)
            # pair-level rows
            z2 = _arrays()
            X = z2[f"{s['task']}|{s['size']}|X"]
            n = X.shape[0]
            ii, jj = np.triu_indices(n, k=1)
            for a in range(len(ii)):
                pair_rows.append(dict(
                    task=s["task"], size=s["size"], i=int(ii[a]), j=int(jj[a]),
                    p_flip=pe["p_flip"][a], p_err=pe["p_err"][a],
                    er_real=pe["er_real"][a], er_pros=pe["er_pros"][a],
                    e_abs_dY=pe["e_abs_dY"][a],
                    p_err_discord=pe["p_err_discord"][a],
                    p_err_noise=pe["p_err_noise"][a],
                    p_err_pros3=pe["p_err_pros3"][a],
                    p_err_pros1=pe["p_err_pros1"][a]))
    pd.DataFrame(rows).to_csv(f"{RES}/cell_estimates.csv", index=False)
    pd.DataFrame(pair_rows).to_parquet(f"{RES}/pair_estimates.parquet", index=False)
    print("fit done:", len(rows), "cells,", len(pair_rows), "pairs")


def _fit_star(job):
    key, task, size, M, seed = job
    z = _arrays()
    return _fit_one(key, task, size, z, M, seed)


# ------------------------------------------------------------------ diagnostics
def stage_diagnostics(workers=8):
    from scipy import stats as sst
    z = _arrays()
    idx = pd.read_csv(f"{RES}/cells_index.csv")
    rng = np.random.default_rng(7)
    diag = {"cells": [], "summary": {}}
    skews, kurts, sw_p, zets = [], [], [], []
    g_ratio_all = []
    for r in idx.itertuples():
        fit = CellFit(z[r.key + "|X"], z[r.key + "|Y"])
        # D1 seed main effects vs noise
        med_sig = np.exp(fit.lamX)
        g_ratio = float(np.max(np.abs(fit.gX)) / med_sig) if med_sig > 0 else np.nan
        # D2 normality of standardized residuals (pooled, both scales)
        std_res = np.concatenate([
            (fit.epsX / np.sqrt(fit.s2X)[:, None]).ravel(),
            (fit.epsY / np.sqrt(fit.s2Y)[:, None]).ravel()])
        sk = float(sst.skew(std_res))
        ku = float(sst.kurtosis(std_res))
        try:
            p = float(sst.shapiro(std_res).pvalue)
        except Exception:
            p = np.nan
        # D4 cross-scale noise-level correlation
        d4 = float(np.corrcoef(np.log(fit.s2X), np.log(fit.s2Y))[0, 1])
        # D3 cross-scale residual corr on matching position (labels differ; position-0 = default both sides)
        d3 = float(np.corrcoef(fit.epsX[:, 0], fit.epsY[:, 0])[0, 1])
        diag["cells"].append(dict(task=r.task, size=r.size, g_ratio=g_ratio,
                                  skew=sk, exkurt=ku, shapiro_p=p,
                                  d4_logvar_corr=d4, d3_default_resid_corr=d3))
        skews.append(sk); kurts.append(ku); sw_p.append(p); zets.append(fit.zetX)
        g_ratio_all.append(g_ratio)
        # D2 proper: posterior-predictive shape check of residuals under the fitted
        # scale-mixture (lognormal sig); KS distance calibrated by simulation.
        # (Naive Shapiro on eps/s_i is invalid: ratio of 3 residuals to their own
        # SD is bounded, exkurt = -1.5 by construction.)
        def _sim_z(lam, zet, rng_):
            s_ = np.exp(lam + zet * rng_.standard_normal(25))[:, None] \
                * rng_.standard_normal((25, 3))
            e_ = s_ - s_.mean(axis=1, keepdims=True) - \
                (s_.mean(axis=0, keepdims=True) - s_.mean())
            return np.sort((e_ / np.exp(lam)).ravel())

        def _ks(a, b):
            allv = np.concatenate([a, b])
            ca = np.searchsorted(a, allv, side="right") / len(a)
            cb = np.searchsorted(b, allv, side="right") / len(b)
            return float(np.max(np.abs(ca - cb)))

        ppc_p = []
        for eps, lam, zet in ((fit.epsX, fit.lamX, fit.zetX),
                              (fit.epsY, fit.lamY, fit.zetY)):
            zr = np.sort((eps / np.exp(lam)).ravel())
            sims = [_sim_z(lam, zet, rng) for _ in range(50)]
            stat_real = float(np.mean([_ks(zr, s) for s in sims]))
            stat_fake = [float(np.mean([_ks(f, s) for s in sims[:20]]))
                         for f in (_sim_z(lam, zet, rng) for _ in range(100))]
            ppc_p.append(float(np.mean(np.array(stat_fake) >= stat_real)))
        diag["cells"][-1]["ppc_shape_p_X"] = ppc_p[0]
        diag["cells"][-1]["ppc_shape_p_Y"] = ppc_p[1]
    ppcX = np.array([c["ppc_shape_p_X"] for c in diag["cells"]])
    ppcY = np.array([c["ppc_shape_p_Y"] for c in diag["cells"]])
    diag["summary"] = dict(
        g_ratio_median=float(np.nanmedian(g_ratio_all)),
        g_ratio_p95=float(np.nanpercentile(g_ratio_all, 95)),
        abs_skew_gt05=int(np.sum(np.abs(np.array(skews)) > 0.5)),
        exkurt_gt1=int(np.sum(np.array(kurts) > 1)),
        shapiro_p_lt001=int(np.sum(np.array(sw_p) < 0.01)),
        shapiro_note=("invalid diagnostic: eps/s_i ratio is bounded (n=3), "
                      "exkurt=-1.5 by construction; use ppc_shape instead"),
        ppc_shape_p_lt005_X=int(np.sum(ppcX < 0.05)),
        ppc_shape_p_lt005_Y=int(np.sum(ppcY < 0.05)),
        ppc_shape_p_median_X=float(np.median(ppcX)),
        ppc_shape_p_median_Y=float(np.median(ppcY)),
        zetX_median=float(np.median(zets)),
        zetX_q10=float(np.percentile(zets, 10)),
        zetX_q90=float(np.percentile(zets, 90)),
        n_cells=len(idx))
    # D6 influence: 150M macro leave-one-recipe-out
    key = f"{MACRO}|150M"
    X, Y = z[key + "|X"], z[key + "|Y"]
    base = cell_summary(CellFit(X, Y), 3000, np.random.default_rng(1))
    loo = []
    for k in range(X.shape[0]):
        m = np.arange(X.shape[0]) != k
        s = cell_summary(CellFit(X[m], Y[m]), 3000, np.random.default_rng(2 + k))
        loo.append(dict(drop=int(k),
                        d_mean_p_flip=s["mean_p_flip"] - base["mean_p_flip"],
                        d_er_real_share=s["er_real_share"] - base["er_real_share"],
                        d_nas_weighted=s["nas_weighted"] - base["nas_weighted"]))
    diag["loo_150M_macro"] = dict(
        max_abs_d_mean_p_flip=float(np.max(np.abs([l["d_mean_p_flip"] for l in loo]))),
        max_abs_d_er_real_share=float(np.max(np.abs([l["d_er_real_share"] for l in loo]))),
        max_abs_d_nas_weighted=float(np.max(np.abs([l["d_nas_weighted"] for l in loo]))))
    with open(f"{RES}/diagnostics.json", "w") as f:
        json.dump(diag, f, indent=1)
    print("diagnostics done:", json.dumps(diag["summary"], indent=1))
    print("LOO:", json.dumps(diag["loo_150M_macro"], indent=1))


# ------------------------------------------------------------------ bootstrap
from hme import fast_estimands

ESTIMANDS = ["mean_p_flip", "exp_errors", "er_real_share", "er_pick_real",
             "nas_weighted", "share_pflip_gt25", "exp_errors_noise",
             "exp_errors_discord", "exp_errors_pros1", "er_pick_latent",
             "noise_share", "discord_share"]

# cells that need per-cell CIs: macro at all sizes + all tasks at 150M
BOOT_CELLS = [(MACRO, s) for s in SIZES] + [(t, "150M") for t in TASKS10]


def _boot_cell(job):
    key, task, size, B, M, seed = job
    z = _arrays()
    X, Y = z[key + "|X"], z[key + "|Y"]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    out = {e: [] for e in ESTIMANDS}
    for b in range(B):
        ridx = rng.integers(0, n, n)
        try:
            fit = CellFit(X[ridx], Y[ridx])
            s = fast_estimands(fit, M, rng)
        except Exception:
            continue
        for e in ESTIMANDS:
            if e == "noise_share":
                v = s["exp_errors_noise"] / max(s["exp_errors"], 1e-12)
            elif e == "discord_share":
                v = s["exp_errors_discord"] / max(s["exp_errors"], 1e-12)
            else:
                v = s[e]
            out[e].append(v)
    rows = []
    for e in ESTIMANDS:
        v = np.array(out[e])
        v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        rows.append(dict(task=task, size=size, estimand=e, B=len(v),
                         boot_mean=float(v.mean()),
                         lo=float(np.percentile(v, 2.5)),
                         hi=float(np.percentile(v, 97.5))))
    return rows


def stage_bootstrap(B=1000, M=400, workers=4):
    """Incremental + resumable: appends per-cell rows as cells complete."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    path = f"{RES}/bootstrap_ci.csv"
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = {(r.task, r.size) for r in prev.itertuples()}
        header = False
    else:
        header = True
    jobs = [(f"{t}|{s}", t, s, B, M, 5000 + k)
            for k, (t, s) in enumerate(BOOT_CELLS) if (t, s) not in done]
    print(f"bootstrap: {len(jobs)} cells to do, {len(done)} done")
    t0 = time.time()
    with open(path, "a") as f:
        if header:
            f.write("task,size,estimand,B,boot_mean,lo,hi\n")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_boot_cell, j): j for j in jobs}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    f.write(",".join(str(r[k]) for k in
                                     ["task", "size", "estimand", "B",
                                      "boot_mean", "lo", "hi"]) + "\n")
                f.flush()
                j = futs[fut]
                print(f"  done {j[1]}|{j[2]} ({time.time()-t0:.0f}s)", flush=True)
    print("bootstrap done in", round(time.time() - t0, 1), "s")


# ---------------------------------------------- pooled bootstraps (per size)
def _pooled_from_pairs(pe_list):
    """Pooled task-level estimands from a list of per-task pair dicts."""
    p_err = np.concatenate([p["p_err"] for p in pe_list])
    p_flip = np.concatenate([p["p_flip"] for p in pe_list])
    er = np.concatenate([p["er_real"] for p in pe_list])
    sig = np.concatenate([p["e_abs_dY"] for p in pe_list])
    return dict(
        exp_errors=float(p_err.sum()),
        mean_p_flip=float(p_flip.mean()),
        nas_weighted=float((p_err * p_flip).sum() / max(p_err.sum(), 1e-12)),
        er_real_share=float(er.sum() / max(sig.sum(), 1e-12)),
    )


def _boot_pooled_recipe(job):
    size, B, M, seed = job
    z = _arrays()
    rng = np.random.default_rng(seed)
    cells = {t: (z[f"{t}|{size}|X"], z[f"{t}|{size}|Y"]) for t in TASKS10}
    n = cells[TASKS10[0]][0].shape[0]
    out = []
    for b in range(B):
        ridx = rng.integers(0, n, n)
        agg = []
        ok = True
        for t in TASKS10:
            X, Y = cells[t]
            try:
                fit = CellFit(X[ridx], Y[ridx])
                agg.append(fast_estimands(fit, M, rng))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        err = sum(a["exp_errors"] for a in agg)
        sig = sum(a["signal_sum"] for a in agg)
        npair = sum(a["n_pairs"] for a in agg)
        out.append(dict(
            exp_errors=err,
            mean_p_flip=sum(a["mean_p_flip"] * a["n_pairs"] for a in agg) / npair,
            nas_weighted=sum(a["nas_weighted"] * a["exp_errors"] for a in agg) / max(err, 1e-12),
            er_real_share=sum(a["er_real_sum"] for a in agg) / max(sig, 1e-12),
            noise_share=sum(a["exp_errors_noise"] for a in agg) / max(err, 1e-12),
            discord_share=sum(a["exp_errors_discord"] for a in agg) / max(err, 1e-12),
        ))
    rows = []
    for e in ["exp_errors", "mean_p_flip", "nas_weighted", "er_real_share",
              "noise_share", "discord_share"]:
        v = np.array([o[e] for o in out])
        v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        rows.append(dict(size=size, estimand=e, B=len(v),
                         boot_mean=float(v.mean()),
                         lo=float(np.percentile(v, 2.5)),
                         hi=float(np.percentile(v, 97.5)),
                         cluster="recipe"))
    return rows


def stage_bootstrap_pooled(B=500, M=300, workers=4):
    """Incremental + resumable per size."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    path = f"{RES}/bootstrap_pooled_ci.csv"
    done = set()
    header = True
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = set(prev[prev.cluster == "recipe"]["size"])
        header = False
    jobs = [(s, B, M, 9000 + k) for k, s in enumerate(SIZES) if s not in done]
    print(f"pooled bootstrap: {len(jobs)} sizes to do, {len(done)} done")
    with open(path, "a") as f:
        if header:
            f.write("size,estimand,B,boot_mean,lo,hi,cluster\n")
        if jobs:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_boot_pooled_recipe, j): j for j in jobs}
                for fut in as_completed(futs):
                    for r in fut.result():
                        f.write(",".join(str(r[k]) for k in
                                         ["size", "estimand", "B", "boot_mean",
                                          "lo", "hi", "cluster"]) + "\n")
                    f.flush()
                    print("  done size", futs[fut][0], flush=True)
    # task-cluster bootstrap from stored pair estimates (cheap; no refit)
    pe = pd.read_parquet(f"{RES}/pair_estimates.parquet")
    rng = np.random.default_rng(11)
    tc_rows = []
    for size in SIZES:
        per_task = []
        for t in TASKS10:
            sub = pe[(pe.task == t) & (pe["size"] == size)]
            per_task.append(sub)
        vals = {e: [] for e in ["exp_errors", "mean_p_flip", "nas_weighted",
                                "er_real_share", "noise_share", "discord_share"]}
        for b in range(2000):
            tidx = rng.integers(0, len(TASKS10), len(TASKS10))
            parts = [per_task[k] for k in tidx]
            p_err = np.concatenate([p.p_err.values for p in parts])
            p_flip = np.concatenate([p.p_flip.values for p in parts])
            er = np.concatenate([p.er_real.values for p in parts])
            sig = np.concatenate([p.e_abs_dY.values for p in parts])
            pnz = np.concatenate([p.p_err_noise.values for p in parts])
            pdc = np.concatenate([p.p_err_discord.values for p in parts])
            esum = p_err.sum()
            vals["exp_errors"].append(esum)
            vals["mean_p_flip"].append(p_flip.mean())
            vals["nas_weighted"].append((p_err * p_flip).sum() / max(esum, 1e-12))
            vals["er_real_share"].append(er.sum() / max(sig.sum(), 1e-12))
            vals["noise_share"].append(pnz.sum() / max(esum, 1e-12))
            vals["discord_share"].append(pdc.sum() / max(esum, 1e-12))
        for e, v in vals.items():
            v = np.array(v)
            tc_rows.append(dict(size=size, estimand=e, B=len(v),
                                boot_mean=float(v.mean()),
                                lo=float(np.percentile(v, 2.5)),
                                hi=float(np.percentile(v, 97.5)),
                                cluster="task"))
    with open(path, "a") as f:
        for r in tc_rows:
            f.write(",".join(str(r[k]) for k in
                             ["size", "estimand", "B", "boot_mean",
                              "lo", "hi", "cluster"]) + "\n")
    print("pooled bootstrap done")


# ------------------------------------------------------------------ validate
def stage_validate(R=200, M=3000, workers=8):
    """Parametric bootstrap: simulate fake cells from fitted model; run the exact
    descriptive pipeline; compare observed stats; compute in-sim TRUE flippable
    share of errors (attenuation of the 3-seed-agreement classifier)."""
    from concurrent.futures import ProcessPoolExecutor
    z = _arrays()
    obs = pd.read_csv(f"{RES}/observed_descriptive.csv")
    jobs = []
    # macro cells 90M..750M + all 10 tasks at 150M (for task-level pooled)
    val_cells = [(MACRO, s) for s in ["90M", "150M", "300M", "530M", "750M"]]
    val_cells += [(t, "150M") for t in TASKS10]
    for k, (t, s) in enumerate(val_cells):
        jobs.append((f"{t}|{s}", t, s, R, 20000 + k))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, summ in ex.map(_validate_cell, jobs):
            results[key] = summ
    # observed reference values
    obsd = {(r.task, r.size): r for r in obs.itertuples()}
    out = {}
    for (t, s) in val_cells:
        key = f"{t}|{s}"
        o = obsd[(t, s)]
        sim = results[key]
        out[key] = dict(
            observed=dict(errors=int(o.errors),
                          share_err_any_unstable=float(o.share_err_any_unstable),
                          share_pairs_unstable=float(o.share_pairs_unstable)),
            simulated=sim)
    with open(f"{RES}/validation.json", "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, json.dumps(v))


def _validate_cell(job):
    key, task, size, R, seed = job
    z = _arrays()
    fit = CellFit(z[key + "|X"], z[key + "|Y"])
    rng = np.random.default_rng(seed)
    errs, shares, unshares = [], [], []
    true25, truemean = [], []
    for r in range(R):
        X, Y, truth = simulate_cell(fit, rng)
        st = descriptive_stats(X, Y)
        errs.append(st["errors"])
        shares.append(st["share_err_any_unstable"])
        unshares.append(st["share_pairs_unstable"])
        t25, tm = true_pflip_share_of_errors(X, Y, truth)
        true25.append(t25)
        truemean.append(tm)
    errs = np.array(errs, dtype=float)
    shares = np.array(shares, dtype=float)
    unshares = np.array(unshares, dtype=float)
    true25 = np.array(true25, dtype=float)
    truemean = np.array(truemean, dtype=float)
    summ = dict(
        R=R,
        errors_mean=float(np.nanmean(errs)),
        errors_q025=float(np.nanpercentile(errs, 2.5)),
        errors_q975=float(np.nanpercentile(errs, 97.5)),
        share_err_unstable_mean=float(np.nanmean(shares)),
        share_err_unstable_q025=float(np.nanpercentile(shares, 2.5)),
        share_err_unstable_q975=float(np.nanpercentile(shares, 97.5)),
        share_pairs_unstable_mean=float(np.nanmean(unshares)),
        true_flippable25_share_of_errors_mean=float(np.nanmean(true25)),
        true_flippable25_share_of_errors_q025=float(np.nanpercentile(true25, 2.5)),
        true_flippable25_share_of_errors_q975=float(np.nanpercentile(true25, 97.5)),
        true_mean_pflip_of_errors=float(np.nanmean(truemean)),
    )
    return key, summ


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    stage = sys.argv[1]
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    if stage == "extract":
        stage_extract()
    elif stage == "fit":
        stage_fit(workers=workers)
    elif stage == "diagnostics":
        stage_diagnostics(workers=workers)
    elif stage == "bootstrap":
        stage_bootstrap(workers=workers)
    elif stage == "bootstrap_pooled":
        stage_bootstrap_pooled(workers=workers)
    elif stage == "validate":
        stage_validate(workers=workers)
    else:
        print("unknown stage", stage)
