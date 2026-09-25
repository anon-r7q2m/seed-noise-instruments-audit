#!/usr/bin/env python3
"""Finite-sample Monte-Carlo calibration of the one-sided Fisher-z certificate
for lambda >= R0 under the attenuation model: actual Type-I error and power at
the operating point (k=10 tasks, nux=4, nuy=2, sigma_a=0.74, R0=0.9, alpha=0.05),
plus robustness to cross-task dependence (equicorrelated task latents).
"""
import numpy as np
from scipy.special import polygamma
from scipy.stats import norm, chi2

rng = np.random.default_rng(20260828)

K, NUX, NUY, SA, R0, ALPHA = 10, 4, 2, 0.74, 0.9, 0.05
N = 400_000

psi1 = lambda v: polygamma(1, v)  # trigamma: Var(0.5*log(chi2_nu/nu)) = psi1(nu/2)/4
A = ((1 + psi1(NUX / 2) / (4 * SA**2)) * (1 + psi1(NUY / 2) / (4 * SA**2))) ** -0.5
z_a = norm.ppf(1 - ALPHA)

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def simulate(lam, rho_task=0.0):
    """One vector of N simulated certificate decisions at true lambda."""
    # latent (a,b) with corr lam; optional equicorrelation across tasks via shared factor
    g = rng.standard_normal((N, 1)) * np.sqrt(rho_task)          # shared task factor
    z1 = rng.standard_normal((N, K)) * np.sqrt(1 - rho_task)
    a = SA * (g + z1)
    z2 = rng.standard_normal((N, K))
    b = SA * (lam * (g + z1) / 1.0 + np.sqrt(max(1 - lam**2, 0)) * z2)
    # note: shared factor enters both arms identically -> cross-task corr of the PAIR
    # chi-square sampling noise
    ex = 0.5 * np.log(chi2.rvs(NUX, size=(N, K), random_state=rng) / NUX)
    ey = 0.5 * np.log(chi2.rvs(NUY, size=(N, K), random_state=rng) / NUY)
    U, Z = a + ex, b + ey
    # observed correlation per sim
    Uc = U - U.mean(axis=1, keepdims=True)
    Zc = Z - Z.mean(axis=1, keepdims=True)
    r = (Uc * Zc).sum(axis=1) / np.sqrt((Uc**2).sum(axis=1) * (Zc**2).sum(axis=1))
    # one-sided Fisher-z certificate of rho >= A*R0 (asymptotic, k-3 df)
    stat = np.sqrt(K - 3) * (fisher_z(r) - np.arctanh(A * R0))
    return stat > z_a

# nominal power (Prop 1 closed form)
nom_power = norm.cdf(np.sqrt(K - 3) * (np.arctanh(A) - np.arctanh(A * R0)) - z_a)
print(f"A = {A:.4f}  (attenuation ceiling at nux={NUX}, nuy={NUY}, sigma_a={SA})")
print(f"nominal asymptotic: alpha = {ALPHA}, power at lambda=1 = {nom_power:.4f}")
print()
hdr = f"{'rho_task':>8} | {'Type-I (lam=0.9)':>16} | {'power (lam=1.0)':>15}"
print(hdr); print("-" * len(hdr))
for rt in (0.0, 0.2, 0.5):
    t1 = simulate(R0, rt).mean()
    pw = simulate(1.0, rt).mean()
    print(f"{rt:>8.1f} | {t1:>16.4f} | {pw:>15.4f}")

# reference: nominal at other k
print()
for k in (5, 10, 20, 50):
    pw = norm.cdf(np.sqrt(k - 3) * (np.arctanh(A) - np.arctanh(A * R0)) - z_a)
    print(f"nominal power at k={k:>3}: {pw:.4f}")
