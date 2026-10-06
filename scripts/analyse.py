"""Turn results/fits.csv and results/a0.csv into the tables and numbers of the paper.

Writes results/table_fits.csv (the per-galaxy table of the appendix),
results/summary.txt (every headline number, next to the published value) and
results/comparison.csv (galaxy-by-galaxy differences from the published table).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparcfit import models as M  # noqa: E402
from sparcfit.data import load_sample  # noqa: E402
from sparcfit.rar import fit_a0, rar_points  # noqa: E402


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_fits():
    fits = {}
    for row in read_csv(ROOT / "results" / "fits.csv"):
        fits.setdefault(row["galaxy"], {})[row["model"]] = {
            k: float(v) for k, v in row.items() if k not in ("galaxy", "model") and v != ""}
    return fits


def main():
    sample = load_sample()
    fits = load_fits()
    names = [g.name for g in sample]
    col = lambda model, key: np.array([fits[n][model][key] for n in names])
    Q = np.array([g.Q for g in sample])
    N = np.array([g.N for g in sample])
    vflat = np.array([g.Vflat for g in sample])
    sb = np.array([g.SBeff for g in sample])
    lum = np.array([g.L36 for g in sample])
    mhi = np.array([g.MHI for g in sample])
    f_d = np.array([g.f_D for g in sample])
    sig_d = np.array([g.e_D / g.D for g in sample])

    dbic = col("nfw", "bic") - col("mond", "bic")
    dbic_l = col("lcdm", "bic") - col("mond", "bic")
    daic = col("nfw", "aic") - col("mond", "aic")
    c2 = {m: col(m, "chi2nu") for m in M.MODELS}

    lines = []

    def say(label, value, published=""):
        lines.append(f"{label:<58s} {value:<28s} {published}")

    say("QUANTITY", "THIS RUN", "PUBLISHED")
    say("galaxies / points / median points", f"{len(sample)} / {N.sum()} / {np.median(N):.0f}", "175 / 3391 / 14")
    say("quality flags 1 / 2 / 3", " / ".join(str((Q == q).sum()) for q in (1, 2, 3)), "99 / 64 / 12")

    lines.append("\n-- Fit quality (NFW / LCDM / MOND)")
    fmt3 = lambda values, f="{:.2f}": " / ".join(f.format(v) for v in values)
    say("median chi2nu, all", fmt3([np.median(c2[m]) for m in M.MODELS]), "1.68 / 1.85 / 2.10")
    say("median chi2nu, Q=1", fmt3([np.median(c2[m][Q == 1]) for m in M.MODELS]), "1.48 / 1.56 / 1.82")
    say("fraction chi2nu < 1 (%)", fmt3([100 * np.mean(c2[m] < 1) for m in M.MODELS], "{:.0f}"), "34 / 27 / 23")
    say("fraction chi2nu > 5 (%)", fmt3([100 * np.mean(c2[m] > 5) for m in M.MODELS], "{:.0f}"), "21 / 23 / 25")
    say("total chi2", fmt3([col(m, "chi2").sum() for m in M.MODELS], "{:.0f}"), "9969 / 10354 / 13514")
    say("both NFW and MOND chi2nu < 2", str(int(np.sum((c2["nfw"] < 2) & (c2["mond"] < 2)))), "77")
    say("both NFW and MOND chi2nu > 5", str(int(np.sum((c2["nfw"] > 5) & (c2["mond"] > 5)))), "26")
    cost = col("lcdm", "chi2") - col("nfw", "chi2")
    say("LCDM prior cost: median dchi2 / n(>10)", f"{np.median(cost):.1f} / {int(np.sum(cost > 10))}", "0.7 / 9")

    lines.append("\n-- Model comparison (halo / MOND, strong halo / strong MOND, undecided)")

    def tally(delta, mask=slice(None)):
        d = delta[mask]
        return f"{np.sum(d < 0)} / {np.sum(d > 0)}, {np.sum(d < -6)} / {np.sum(d > 6)}, {np.sum(np.abs(d) <= 2)}"

    say("BIC, free NFW, all", tally(dbic), "96 / 79, 63 / 23, 37")
    say("BIC, NFW+LCDM, all", tally(dbic_l), "86 / 89, 60 / 34, 37")
    say("AIC, free NFW, all", tally(daic), "105 / 70, 68 / 18, 34")
    say("BIC, free NFW, Q=1", tally(dbic, Q == 1), "58 / 41, 39 / 16")
    say("BIC, NFW+LCDM, Q=1", tally(dbic_l, Q == 1), "55 / 44, 38 / 22")
    say("AIC, free NFW, Q=1", tally(daic, Q == 1), "65 / 34, 43 / 11")
    say("median dBIC free / constrained", f"{np.median(dbic):+.1f} / {np.median(dbic_l):+.1f}", "-0.9 / +0.2")
    say("very strong (|dBIC|>10) halo / MOND", f"{np.sum(dbic < -10)} / {np.sum(dbic > 10)}", "56 / 12")
    say("summed dBIC", f"{dbic.sum():.0f}", "-2598")
    order = np.argsort(dbic)
    say("most pro-halo", ", ".join(f"{names[i]} ({dbic[i]:+.0f})" for i in order[:4]), "F571-8 -399, IC4202 -363, NGC2403 -355, NGC5907 -193")
    say("most pro-MOND", ", ".join(f"{names[i]} ({dbic[i]:+.0f})" for i in order[::-1][:4]), "IC2574 +994, NGC3109 +119, DDO154 +69, NGC2366 +38")

    lines.append("\n-- Subsamples: n, halo, MOND, median chi2nu NFW / LCDM / MOND")
    subsamples = [
        ("SBeff < 100", sb < 100, "94 45 49 1.47 / 1.79 / 1.80"),
        ("SBeff >= 100", sb >= 100, "81 51 30 1.92 / 2.04 / 2.42"),
        ("Vflat < 100", (vflat > 0) & (vflat < 100), "54 32 22 1.28 / 1.44 / 1.68"),
        ("Vflat >= 100", vflat >= 100, "81 52 29 1.95 / 2.04 / 2.60"),
        ("no flat part", vflat == 0, "40 12 28 1.54 / 1.95 / 1.59"),
        ("gas-dominated (MHI > 0.5 L)", mhi > 0.5 * lum, "73 40 33 1.51 / 1.76 / 1.84"),
    ]
    for label, mask, published in subsamples:
        say(label, f"{mask.sum()} {np.sum(dbic[mask] < 0)} {np.sum(dbic[mask] > 0)} " +
            fmt3([np.median(c2[m][mask]) for m in M.MODELS]), published)
    say("median dBIC, no flat part", f"{np.median(dbic[vflat == 0]):+.1f}", "+2.6")

    lines.append("\n-- Concentration-mass relation (free NFW)")
    logc, logm = col("nfw", "log_c_med"), col("nfw", "log_M200_med")
    off = logc - M.cm_relation(logm)
    say("offset median / std (dex)", f"{np.median(off):+.2f} / {np.std(off):.2f}", "-0.24 / 0.40")
    say("within 2 sigma / below / above (%)", fmt3([100 * np.mean(np.abs(off) <= 0.22), 100 * np.mean(off < -0.22), 100 * np.mean(off > 0.22)], "{:.0f}"), "30 / 53 / 17")
    low = logm < 11
    say("median offset M200<1e11 (n) / above (n)", f"{np.median(off[low]):+.2f} ({low.sum()}) / {np.median(off[~low]):+.2f} ({(~low).sum()})", "-0.25 (42) / -0.24 (133)")

    lines.append("\n-- Nuisance parameters")
    for model, published in (("nfw", "0.50, 0.40-0.56"), ("mond", "0.51, 0.33-0.89")):
        yd = 10 ** col(model, "log_Yd_med")
        p16, p50, p84 = np.percentile(yd, [16, 50, 84])
        say(f"Yd {model.upper()}: median, 16-84%", f"{p50:.2f}, {p16:.2f}-{p84:.2f}", published)
    for model, published in (("nfw", "0.16, 10"), ("mond", "0.21, 28")):
        d = col(model, "d_med")
        say(f"d {model.upper()}: dispersion, n(|pull| > 2 sigma)", f"{np.std(d):.2f}, {int(np.sum(np.abs(d - 1) > 2 * sig_d))}", published)
    precise = (Q == 1) & np.isin(f_d, (2, 3))
    say("Q=1 with TRGB/Cepheid distance: n, median chi2nu MOND / NFW", f"{precise.sum()}, {np.median(c2['mond'][precise]):.1f} / {np.median(c2['nfw'][precise]):.1f}", "21, 2.1 / 1.5")
    pulled = precise & (np.abs(col("mond", "d_med") - 1) > 2 * sig_d)
    say("  of which MOND distance pulled > 2 sigma", ", ".join(np.array(names)[pulled]) or "none", "NGC0891, NGC3198, NGC5907, NGC7814")

    lines.append("\n-- Radial acceleration relation")
    gbar, gobs, q = rar_points(sample)
    a0, err, resid = fit_a0(gbar, gobs, M.nu_rar)
    a0s, errs, _ = fit_a0(gbar, gobs, M.nu_simple)
    say("points", str(gbar.size), "2803")
    say("a0, RAR function (1e-10 m/s2)", f"{a0 * 1e10:.2f} +- {err * 1e10:.2f}", "1.13 +- 0.02")
    say("a0, simple function (1e-10 m/s2)", f"{a0s * 1e10:.2f} +- {errs * 1e10:.2f}", "1.10 +- 0.02")
    say("scatter all / Q=1 (dex)", f"{np.std(resid):.3f} / {np.std(resid[q == 1]):.3f}", "0.144 / 0.128")

    a0_file = ROOT / "results" / "a0.csv"
    if a0_file.exists():
        lines.append("\n-- Acceleration constant galaxy by galaxy")
        rows = {r["galaxy"]: r for r in read_csv(a0_file)}
        la = np.array([float(rows[n]["log_a0"]) for n in names])
        sg = np.array([float(rows[n]["sigma_log_a0"]) for n in names])
        ok = sg < 0.3
        x, s = la[ok], np.maximum(sg[ok], 0.01)
        mean = np.sum(x / s**2) / np.sum(1 / s**2)
        chi2_const = np.sum(((x - mean) / s) ** 2)
        grid = np.linspace(0, 2, 2001)  # intrinsic scatter that brings chi2 to its dof
        red = [np.sum((x - np.sum(x / (s**2 + t**2)) / np.sum(1 / (s**2 + t**2))) ** 2 / (s**2 + t**2)) for t in grid]
        intrinsic = grid[np.argmin(np.abs(np.array(red) - (ok.sum() - 1)))]
        mad = np.median(np.abs(x - np.median(x)))
        say("galaxies with sigma < 0.3 dex", str(ok.sum()), "121")
        say("median a0 (1e-10 m/s2)", f"{10 ** np.median(x) * 1e10:.2f}", "1.02")
        say("MAD / std of log a0 (dex)", f"{mad:.2f} / {np.std(x):.2f}", "0.36 / 0.58")
        say("chi2 of a constant / dof", f"{chi2_const:.0f} / {ok.sum() - 1}", "3120 / 120")
        say("intrinsic scatter (dex)", f"{intrinsic:.2f}", "0.55")
        for label, values, mask, published in (
                ("luminosity", lum, ok, "~0.1, p > 0.3"),
                ("Vflat", vflat, ok & (vflat > 0), "~0.1, p > 0.3"),
                ("surface brightness", sb, ok, "0.33, p = 3e-4")):
            rho, p = spearmanr(values[mask], la[mask])
            say(f"Spearman with {label}", f"{rho:+.2f}, p = {p:.1e}", published)

    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "summary.txt").write_text(text)
    print(text)

    header = ["galaxy", "Q", "N", "chi2nu_nfw", "log_c", "log_M200", "chi2nu_lcdm", "chi2nu_mond", "Yd_mond", "dBIC"]
    table = [[n, Q[i], N[i], f"{c2['nfw'][i]:.2f}", f"{logc[i]:.2f}", f"{logm[i]:.2f}", f"{c2['lcdm'][i]:.2f}",
              f"{c2['mond'][i]:.2f}", f"{10 ** col('mond', 'log_Yd_med')[i]:.2f}", f"{dbic[i]:+.1f}"]
             for i, n in enumerate(names)]
    with open(ROOT / "results" / "table_fits.csv", "w", newline="") as handle:
        csv.writer(handle).writerows([header] + table)

    published = {r["galaxy"]: r for r in read_csv(ROOT / "paper" / "table_published.csv")}
    with open(ROOT / "results" / "comparison.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["galaxy"] + [f"{h}_{s}" for h in header[3:] for s in ("run", "published")] + ["same_preferred_model"])
        diffs = {h: [] for h in header[3:]}
        agree = 0
        for row in table:
            ref = published[row[0]]
            out = [row[0]]
            for j, h in enumerate(header[3:], 3):
                out += [row[j], ref[h]]
                diffs[h].append(float(row[j]) - float(ref[h]))
            same = (float(row[9]) > 0) == (float(ref["dBIC"]) > 0)
            agree += same
            writer.writerow(out + [int(same)])
    lines = ["\n-- Galaxy-by-galaxy agreement with the published table",
             f"same preferred model (sign of dBIC): {agree} of {len(table)}"]
    for h, d in diffs.items():
        d = np.abs(d)
        lines.append(f"{h:<12s} median |diff| {np.median(d):.3f}   90th pct {np.percentile(d, 90):.3f}   max {d.max():.2f}")
    text = "\n".join(lines) + "\n"
    with open(ROOT / "results" / "summary.txt", "a") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
