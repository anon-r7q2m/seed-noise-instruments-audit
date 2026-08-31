import sys, numpy as np, pandas as pd
sys.path.insert(0, "code/enhancement3")
import hme
DD = "data/analysis/dd_tidy.parquet"
d = pd.read_parquet(DD)
rng = np.random.default_rng(7)
SIZES = ["90M","150M","300M","530M","750M"]
R = 20000
print(f"{'size':>5} | {'realized':>8} | {'pros pt':>8} | {'pros seedonly mean [95%]':>26} | {'latent':>7}")
for sz in SIZES:
    X, Y, recipes = hme.extract_cell(d, "olmes_10_macro_avg", sz)
    fit = hme.CellFit(X, Y)
    # posterior means of latents
    M = 20000
    mu, nu, sX2, sY2 = fit.draw_posterior(M, rng)
    muh, nuh, sXh2 = mu.mean(axis=1), nu.mean(axis=1), sX2.mean(axis=1)
    # PI-specified: fixed recipes, resample seed noise only
    z = rng.standard_normal((R, len(recipes)))
    fresh = muh[None,:] + np.sqrt(sXh2/3.0)[None,:]*z
    picks = np.argmax(fresh, axis=1)
    loss = nuh.max() - nuh[picks]
    # identities for the caption explanation
    i_real = int(np.argmax(fit.xbar)); i_lat = int(np.argmax(muh))
    rank = lambda i: int((nuh > nuh[i]).sum()+1)
    from collections import Counter
    top = Counter(picks.tolist()).most_common(3)
    top_s = "; ".join(f"{recipes[i][:28]}({c/R:.0%},nu-rank {rank(i)})" for i,c in top)
    print(f"{sz:>5} | {float((nuh.max()-nuh[i_real])):8.4f} | {float((nuh.max()-nuh[np.argmax(muh)])):8.4f}..."
          f" | {loss.mean():.4f} [{np.percentile(loss,2.5):.4f}, {np.percentile(loss,97.5):.4f}]   ", flush=True)
    print(f"        realized pick: {recipes[i_real][:30]} (nu-rank {rank(i_real)}); latent argmax: {recipes[i_lat][:30]} (nu-rank {rank(i_lat)})")
    print(f"        pros pick distribution top3: {top_s}")
