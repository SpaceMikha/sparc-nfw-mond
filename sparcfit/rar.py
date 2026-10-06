"""Radial acceleration relation built from the whole sample."""
import numpy as np
from scipy.optimize import curve_fit

from . import models as M


def rar_points(sample, Yd=M.YD0, Yb=M.YB0, max_rel_err=0.1):
    """g_bar and g_obs (m/s^2) for all points passing the quality cuts."""
    gbar, gobs, quality = [], [], []
    to_si = 1e3 / M.KPC_KM
    for gal in sample:
        vb2 = M.vbar2(gal, Yd, Yb)[0]
        keep = (gal.eV / gal.Vobs < max_rel_err) & (vb2 > 0) & (gal.Vobs > 0)
        gbar.append(vb2[keep] / gal.R[keep] * to_si)
        gobs.append(gal.Vobs[keep] ** 2 / gal.R[keep] * to_si)
        quality.append(np.full(keep.sum(), gal.Q))
    return np.concatenate(gbar), np.concatenate(gobs), np.concatenate(quality)


def fit_a0(gbar, gobs, nu=M.nu_rar):
    """One-parameter least-squares fit in log space. Returns (a0, error, residuals)."""
    def model(log_gbar, log_a0):
        return log_gbar + np.log10(nu(10.0 ** (log_gbar - log_a0)))

    x, y = np.log10(gbar), np.log10(gobs)
    popt, pcov = curve_fit(model, x, y, p0=[-10.0])
    resid = y - model(x, *popt)
    a0 = 10.0 ** popt[0]
    return a0, a0 * np.log(10.0) * np.sqrt(pcov[0, 0]), resid
