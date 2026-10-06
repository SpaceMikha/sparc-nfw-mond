"""Maximisation and MCMC sampling of one galaxy / one model."""
import zlib

import emcee
import numpy as np
from scipy.optimize import minimize

from . import models as M

N_WALKERS, N_STEPS, N_BURN = 32, 1200, 400


def _seed(gal, model):
    return zlib.crc32(f"{gal.name}:{model}".encode())


def starting_points(gal, model):
    """Nine starts on a grid in the halo parameters, a single one for MOND."""
    nuis = [np.log10(M.YD0), np.log10(M.YB0), 1.0, gal.inc]
    if model == "mond":
        return [np.array(nuis)]
    if model == "mond_a0":
        return [np.array([np.log10(M.A0_SI)] + nuis)]
    return [np.array([lc, lv] + nuis) for lc in (0.3, 0.9, 1.5) for lv in (1.5, 2.0, 2.5)]


def maximise(gal, model, starts=None):
    """Nelder-Mead maximum of the posterior. Returns (theta, log_post)."""
    lo, hi = M.bounds(gal, model)

    def neg(theta):
        value = M.log_post(theta, gal, model)[0]
        return -value if np.isfinite(value) else 1e30

    best, best_lp = None, -np.inf
    for x0 in (starts if starts is not None else starting_points(gal, model)):
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        res = minimize(neg, x0, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 6000, "maxfev": 6000})
        res = minimize(neg, res.x, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 6000, "maxfev": 6000})
        if -res.fun > best_lp:
            best, best_lp = res.x, -res.fun
    return best, best_lp


def sample(gal, model, theta0, n_steps=N_STEPS, n_burn=N_BURN, n_walkers=N_WALKERS):
    """Affine-invariant ensemble sampling started in a small ball around theta0."""
    rng = np.random.default_rng(_seed(gal, model))
    lo, hi = M.bounds(gal, model)
    width = 1e-3 * (hi - lo)
    p0 = np.clip(theta0 + width * rng.standard_normal((n_walkers, theta0.size)),
                 lo + 1e-9, hi - 1e-9)
    sampler = emcee.EnsembleSampler(n_walkers, theta0.size, M.log_post,
                                    args=(gal, model), vectorize=True)
    sampler.random_state = np.random.RandomState(_seed(gal, model)).get_state()
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=n_burn, flat=True)
    logp = sampler.get_log_prob(discard=n_burn, flat=True)
    return chain, logp


def fit(gal, model):
    """Full fit: MAP point, chi^2, information criteria and posterior summaries."""
    theta_nm, lp_nm = maximise(gal, model)
    chain, logp = sample(gal, model, theta_nm)
    j = int(np.argmax(logp))
    theta_map, lp_map = (chain[j], logp[j]) if logp[j] > lp_nm else (theta_nm, lp_nm)

    k = M.n_free(model, gal)
    c2 = float(M.chi2(theta_map, gal, model)[0])
    out = {
        "galaxy": gal.name, "model": model, "N": gal.N, "k": k,
        "chi2": c2, "chi2nu": c2 / max(gal.N - k, 1),
        "bic": c2 + k * np.log(gal.N), "aic": c2 + 2 * k, "logpost_map": float(lp_map),
    }
    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    for i, name in enumerate(M.PARAMS[model]):
        out[f"{name}_map"] = float(theta_map[i])
        out[f"{name}_p16"], out[f"{name}_med"], out[f"{name}_p84"] = map(float, (q16[i], q50[i], q84[i]))
    if model in ("nfw", "lcdm"):
        lm = M.log_m200(chain[:, 1])
        out["log_M200_p16"], out["log_M200_med"], out["log_M200_p84"] = map(float, np.percentile(lm, [16, 50, 84]))
        out["log_M200_map"] = float(M.log_m200(theta_map[1]))
    return out


def profile_a0(gal, grid=None):
    """Profile posterior of log10(a0) for the MOND fit with a free a0.

    Returns (best log a0, 1-sigma half width from Delta(-2 ln P) = 1, grid, -2 ln P).
    """
    grid = np.arange(M.LOGA0_LIM[0], M.LOGA0_LIM[1] + 1e-9, 0.05) if grid is None else grid
    lo, hi = M.bounds(gal, "mond_a0")

    def neg_at(log_a0):
        def neg(nuis):
            value = M.log_post(np.concatenate([[log_a0], nuis]), gal, "mond_a0")[0]
            return -value if np.isfinite(value) else 1e30
        return neg

    prof = np.empty(grid.size)
    start = np.array([np.log10(M.YD0), np.log10(M.YB0), 1.0, gal.inc])
    fiducial = start.copy()
    order = np.argsort(np.abs(grid - np.log10(M.A0_SI)))  # walk outwards from the canonical value
    sol = {}
    for idx in order:
        near = [sol[j] for j in (idx - 1, idx + 1) if j in sol]
        best = np.inf
        for x0 in near + [fiducial]:
            x0 = np.clip(x0, lo[1:] + 1e-6, hi[1:] - 1e-6)
            res = minimize(neg_at(grid[idx]), x0, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 4000, "maxfev": 4000})
            if res.fun < best:
                best, sol[idx] = res.fun, res.x
        prof[idx] = 2.0 * best
    i = int(np.argmin(prof))
    dchi = prof - prof[i]

    def crossing(side):
        j = i
        while 0 <= j + side < grid.size and dchi[j + side] < 1.0:
            j += side
        if not 0 <= j + side < grid.size:
            return np.inf
        f = (1.0 - dchi[j]) / (dchi[j + side] - dchi[j])
        return abs(grid[j] + side * f * (grid[1] - grid[0]) - grid[i])

    sigma = 0.5 * (crossing(-1) + crossing(+1))
    return float(grid[i]), float(sigma), grid, prof
