"""Draw the eight figures of the paper from results/ into figs/."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparcfit import models as M  # noqa: E402
from sparcfit.data import load_sample  # noqa: E402
from sparcfit.rar import fit_a0, rar_points  # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "legend.frameon": False})
BLUE, GREEN, RED = "#1f5fa8", "#2a8f4f", "#c0392b"
COL = {"nfw": BLUE, "lcdm": GREEN, "mond": RED}
LAB = {"nfw": "NFW", "lcdm": r"NFW+$\Lambda$CDM", "mond": "MOND"}
QSTYLE = {1: ("o", "#222222"), 2: ("s", "#888888"), 3: ("^", "#cc8800")}
FIGS = ROOT / "figs"


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)


def by_quality(ax, x, y, Q, **kw):
    for q, (marker, colour) in QSTYLE.items():
        m = Q == q
        ax.scatter(x[m], y[m], s=10, marker=marker, color=colour, lw=0, label=f"$Q={q}$", **kw)


def main():
    sample = load_sample()
    gal = {g.name: g for g in sample}
    names = [g.name for g in sample]
    fits = {}
    for row in read_csv(ROOT / "results" / "fits.csv"):
        fits.setdefault(row["galaxy"], {})[row["model"]] = {
            k: float(v) for k, v in row.items() if k not in ("galaxy", "model") and v != ""}
    col = lambda model, key: np.array([fits[n][model][key] for n in names])
    Q = np.array([g.Q for g in sample])
    vflat = np.array([g.Vflat for g in sample])
    sb = np.array([g.SBeff for g in sample])
    lum = np.array([g.L36 for g in sample]) * 1e9
    dbic = col("nfw", "bic") - col("mond", "bic")

    # 1. sample
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    m = vflat > 0
    by_quality(ax[0], lum[m], vflat[m], Q[m])
    ax[0].set(xscale="log", yscale="log", xlabel=r"$L_{[3.6]}$ [$L_\odot$]", ylabel=r"$V_{\rm flat}$ [km s$^{-1}$]")
    ax[0].legend()
    ax[1].hist(np.log10(sb), bins=24, color="#888888")
    ax[1].set(xlabel=r"$\log_{10}\,\Sigma_{\rm eff}$ [$L_\odot$ pc$^{-2}$]", ylabel="Number of galaxies")
    save(fig, "fig_sample.pdf")

    # 2. chi2 CDF
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for model in M.MODELS:
        x = np.sort(col(model, "chi2nu"))
        ax.step(x, np.arange(1, x.size + 1) / x.size, color=COL[model], label=LAB[model], where="post")
    ax.axvline(1, color="k", ls=":", lw=0.8)
    ax.set(xscale="log", xlim=(0.03, 60), ylim=(0, 1), xlabel=r"$\chi^2_\nu$", ylabel="Cumulative fraction")
    ax.legend(loc="upper left")
    save(fig, "fig_chi2cdf.pdf")

    # 3. examples
    chosen = ["NGC2403", "NGC3198", "NGC6503", "DDO154", "IC2574", "UGC00128", "NGC2841", "F583-1", "UGC05750"]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.8))
    for ax, name in zip(axes.flat, chosen):
        g = gal[name]
        ax.errorbar(g.R, g.Vobs, g.eV, fmt="o", ms=2.5, color="k", lw=0.6, zorder=3)
        for model in M.MODELS:
            theta = np.array([fits[name][model][f"{p}_map"] for p in M.PARAMS[model]])
            # Bring the model back to the catalogued distance and inclination.
            back = np.sin(np.radians(theta[-1])) / np.sin(np.radians(g.inc))
            ax.plot(g.R, M.model_v(theta, g, model)[0] * back, color=COL[model], lw=1.2,
                    ls="--" if model == "lcdm" else "-", label=LAB[model])
            if model == "nfw":
                vb2 = theta[4] * M.vbar2(g, 10 ** theta[2], 10 ** theta[3])[0]
                ax.plot(g.R, np.sqrt(np.clip(vb2, 0, None)) * back, color="#999999", ls=":", lw=1.0)
        ax.set_title(name, fontsize=9)
        ax.text(0.97, 0.05, " / ".join(f"{fits[name][m]['chi2nu']:.2f}" for m in M.MODELS),
                transform=ax.transAxes, ha="right", fontsize=7)
        ax.set(xlim=(0, None), ylim=(0, None))
    for ax in axes[-1]:
        ax.set_xlabel(r"$R$ [kpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$V$ [km s$^{-1}$]")
    axes[0, 0].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    save(fig, "fig_examples.pdf")

    # 4. dBIC
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    clipped = np.clip(dbic, -60, 60)
    by_quality(ax[0], vflat[m], clipped[m], Q[m])
    by_quality(ax[1], sb, clipped, Q)
    for a in ax:
        a.axhspan(-6, 6, color="#dddddd", zorder=0)
        a.axhline(0, color="k", lw=0.5)
        a.set_xscale("log")
    ax[0].set(xlabel=r"$V_{\rm flat}$ [km s$^{-1}$]", ylabel=r"$\Delta$BIC (NFW $-$ MOND)")
    ax[1].set(xlabel=r"$\Sigma_{\rm eff}$ [$L_\odot$ pc$^{-2}$]")
    ax[0].legend()
    save(fig, "fig_dbic.pdf")

    # 5. concentration-mass
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    x, y = col("nfw", "log_M200_med"), col("nfw", "log_c_med")
    ax.errorbar(x, y, xerr=[x - col("nfw", "log_M200_p16"), col("nfw", "log_M200_p84") - x],
                yerr=[y - col("nfw", "log_c_p16"), col("nfw", "log_c_p84") - y],
                fmt="o", ms=2.5, color=BLUE, lw=0.4, alpha=0.8)
    grid = np.linspace(8.5, 14.5, 50)
    ax.fill_between(grid, M.cm_relation(grid) - M.CM_SCATTER, M.cm_relation(grid) + M.CM_SCATTER, color="#bbbbbb")
    ax.plot(grid, M.cm_relation(grid), color="k", lw=1)
    ax.set(xlim=(8.5, 14.5), ylim=(-0.1, 2.1), xlabel=r"$\log_{10}\,M_{200}$ [$M_\odot$]", ylabel=r"$\log_{10}\,c$")
    save(fig, "fig_cM.pdf")

    # 6. mass-to-light ratios
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bins = np.linspace(0.1, 1.5, 29)
    for model in ("nfw", "mond"):
        ax.hist(10 ** col(model, "log_Yd_med"), bins=bins, histtype="stepfilled", alpha=0.45,
                color=COL[model], label=LAB[model])
    ax.axvline(M.YD0, color="k", ls=":")
    ax.set(xlabel=r"$\Upsilon_{\rm disk}$ [$M_\odot/L_\odot$]", ylabel="Number of galaxies")
    ax.legend()
    save(fig, "fig_ML.pdf")

    # 7. radial acceleration relation
    gbar, gobs, _ = rar_points(sample)
    a0, _, resid = fit_a0(gbar, gobs, M.nu_rar)
    a0s, _, _ = fit_a0(gbar, gobs, M.nu_simple)
    lx, ly = np.log10(gbar), np.log10(gobs)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ax[0].hexbin(lx, ly, gridsize=45, cmap="Greys", mincnt=1, linewidths=0)
    grid = np.linspace(lx.min(), lx.max(), 200)
    ax[0].plot(grid, grid + np.log10(M.nu_rar(10 ** grid / a0)), color=RED, lw=1.3)
    ax[0].plot(grid, grid + np.log10(M.nu_simple(10 ** grid / a0s)), color=BLUE, lw=1.1, ls="--")
    ax[0].plot(grid, grid, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel(r"$\log_{10}\,g_{\rm obs}$ [m s$^{-2}$]")
    ax[1].scatter(lx, resid, s=1, color="#aaaaaa", lw=0)
    edges = np.linspace(lx.min(), lx.max(), 13)
    mid = 0.5 * (edges[1:] + edges[:-1])
    which = np.digitize(lx, edges[1:-1])
    ax[1].errorbar(mid, [np.median(resid[which == i]) for i in range(12)],
                   [np.std(resid[which == i]) for i in range(12)], fmt="o", ms=3, color=RED, lw=0.8)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(ylim=(-0.6, 0.6), xlabel=r"$\log_{10}\,g_{\rm bar}$ [m s$^{-2}$]", ylabel="Residual [dex]")
    save(fig, "fig_rar.pdf")

    # 8. a0 galaxy by galaxy
    path = ROOT / "results" / "a0.csv"
    if path.exists():
        rows = {r["galaxy"]: r for r in read_csv(path)}
        la = np.array([float(rows[n]["log_a0"]) for n in names])
        sg = np.array([float(rows[n]["sigma_log_a0"]) for n in names])
        ok = sg < 0.3
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
        ax[0].hist(la[ok], bins=np.arange(-11.5, -8.4, 0.125), color="#888888")
        ax[0].axvline(np.log10(M.A0_SI), color=RED, label=r"$a_0=1.2\times10^{-10}$ m s$^{-2}$")
        ax[0].set(xlabel=r"$\log_{10}\,a_0$ [m s$^{-2}$]", ylabel="Number of galaxies")
        ax[0].legend(fontsize=7)
        mm = ok & (vflat > 0)
        ax[1].errorbar(vflat[mm], la[mm], sg[mm], fmt="o", ms=2.5, color="k", lw=0.5)
        ax[1].axhline(np.log10(M.A0_SI), color=RED)
        ax[1].set(xscale="log", xlabel=r"$V_{\rm flat}$ [km s$^{-1}$]", ylabel=r"$\log_{10}\,a_0$ [m s$^{-2}$] (per galaxy)")
        save(fig, "fig_a0.pdf")
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
