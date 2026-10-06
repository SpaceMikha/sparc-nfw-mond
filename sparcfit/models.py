"""Mass models, priors and posteriors.

All velocities are in km/s, radii in kpc, accelerations in (km/s)^2/kpc.
Every function is vectorised over a leading "walker" axis: parameters have
shape (W,) and model velocities have shape (W, N).
"""
import numpy as np

G = 4.30091e-6                    # kpc (km/s)^2 / Msun
H0 = 73.0e-3                      # km/s/kpc (73 km/s/Mpc, the SPARC distance scale)
H_CM = 0.671                      # h of the concentration-mass relation
KPC_KM = 3.0856775814913673e16    # km per kpc
A0_SI = 1.2e-10                   # m/s^2
A0 = A0_SI * 1e-3 * KPC_KM        # (km/s)^2 / kpc

YD0, YB0, SIG_Y = 0.5, 0.7, 0.1   # stellar M/L priors (centres, width in dex)
LOGY_LIM = (-1.0, 1.0)             # hard limits on log10 M/L
LOGC_LIM = (0.0, 2.0)
LOGV_LIM = (1.0, 2.9)
LOGA0_LIM = (-12.0, -8.0)         # log10 of a0 in m/s^2
CM_SCATTER = 0.11                 # dex

MODELS = ("nfw", "lcdm", "mond")
PARAMS = {
    "nfw": ("log_c", "log_V200", "log_Yd", "log_Yb", "d", "inc"),
    "lcdm": ("log_c", "log_V200", "log_Yd", "log_Yb", "d", "inc"),
    "mond": ("log_Yd", "log_Yb", "d", "inc"),
    "mond_a0": ("log_a0", "log_Yd", "log_Yb", "d", "inc"),
}


def n_free(model, gal):
    """Number of free parameters; a bulge M/L only counts if there is a bulge."""
    k = {"nfw": 5, "lcdm": 5, "mond": 3, "mond_a0": 4}[model]
    return k + int(gal.has_bulge)


def vbar2(gal, Yd, Yb):
    """Baryonic V^2 at the catalogued distance (Eq. 1), shape (W, N)."""
    Yd = np.atleast_1d(Yd)[:, None]
    Yb = np.atleast_1d(Yb)[:, None]
    return Yd * gal.Vdisk**2 + Yb * gal.Vbul**2 + gal.Vgas * np.abs(gal.Vgas)


def log_m200(log_v200):
    return 3.0 * log_v200 - np.log10(10.0 * G * H0)


def cm_relation(logm):
    """Dutton & Maccio (2014) c200-M200 relation at z = 0 (Eq. 4)."""
    return 0.905 - 0.101 * (logm - 12.0 + np.log10(H_CM))


def vnfw2(R, log_c, log_v200):
    """NFW circular velocity squared (Eq. 3). R has shape (W, N)."""
    c = 10.0 ** np.atleast_1d(log_c)[:, None]
    v200 = 10.0 ** np.atleast_1d(log_v200)[:, None]
    x = R / (v200 / (10.0 * H0))
    mu = lambda t: np.log1p(t) - t / (1.0 + t)
    return v200**2 * mu(c * x) / (x * mu(c))


def nu_simple(y):
    return 0.5 + np.sqrt(0.25 + 1.0 / y)


def nu_rar(y):
    return 1.0 / (1.0 - np.exp(-np.sqrt(y)))


def unpack(theta, model):
    theta = np.atleast_2d(theta)
    return {name: theta[:, j] for j, name in enumerate(PARAMS[model])}


def model_v(theta, gal, model):
    """Model circular velocity at the fitted distance, shape (W, N)."""
    p = unpack(theta, model)
    d = p["d"][:, None]
    vb2 = vbar2(gal, 10.0 ** p["log_Yd"], 10.0 ** p["log_Yb"])
    if model in ("nfw", "lcdm"):
        v2 = d * vb2 + vnfw2(d * gal.R, p["log_c"], p["log_V200"])
    else:
        a0 = A0 if model == "mond" else 10.0 ** p["log_a0"][:, None] * 1e-3 * KPC_KM
        # g_bar = V_bar^2 / R does not depend on the distance factor.
        y = np.maximum(np.abs(vb2) / gal.R / a0, 1e-12)
        v2 = nu_simple(y) * vb2 * d
    return np.sqrt(np.clip(v2, 0.0, None))


def bounds(gal, model):
    """Hard prior limits (lower, upper) for every parameter."""
    sd = gal.e_D / gal.D
    lim = {
        "log_c": LOGC_LIM, "log_V200": LOGV_LIM, "log_a0": LOGA0_LIM,
        "log_Yd": LOGY_LIM, "log_Yb": LOGY_LIM,
        "d": (max(1.0 - 3.0 * sd, 0.05), 1.0 + 3.0 * sd),
        "inc": (max(gal.inc - 3.0 * gal.e_inc, 1.0), min(gal.inc + 3.0 * gal.e_inc, 90.0)),
    }
    lo, hi = zip(*(lim[name] for name in PARAMS[model]))
    return np.array(lo), np.array(hi)


def chi2(theta, gal, model):
    """Data term of Eq. (7); the inclination correction scales data and errors."""
    p = unpack(theta, model)
    s = (np.sin(np.radians(gal.inc)) / np.sin(np.radians(p["inc"])))[:, None]
    res = (gal.Vobs * s - model_v(theta, gal, model)) / (gal.eV * s)
    return np.sum(res**2, axis=1)


def log_prior(theta, gal, model):
    theta = np.atleast_2d(theta)
    p = unpack(theta, model)
    lo, hi = bounds(gal, model)
    inside = np.all((theta >= lo) & (theta <= hi), axis=1)
    lp = -0.5 * ((p["log_Yd"] - np.log10(YD0)) / SIG_Y) ** 2
    lp += -0.5 * ((p["log_Yb"] - np.log10(YB0)) / SIG_Y) ** 2
    lp += -0.5 * ((p["d"] - 1.0) * gal.D / gal.e_D) ** 2
    lp += -0.5 * ((p["inc"] - gal.inc) / gal.e_inc) ** 2
    if model == "lcdm":
        mean = cm_relation(log_m200(p["log_V200"]))
        lp += -0.5 * ((p["log_c"] - mean) / CM_SCATTER) ** 2
    return np.where(inside, lp, -np.inf)


def log_post(theta, gal, model):
    theta = np.atleast_2d(theta)
    lp = log_prior(theta, gal, model)
    out = np.full(theta.shape[0], -np.inf)
    ok = np.isfinite(lp)
    if ok.any():
        out[ok] = lp[ok] - 0.5 * chi2(theta[ok], gal, model)
    return out
