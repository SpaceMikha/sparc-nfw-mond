"""Compare the forward CCC fits (results/fits_ccc.csv) with the three models of the paper.

Writes results/summary_ccc.txt and results/table_ccc.csv. The CCC model is an
extension of this repository; it is not part of the published paper.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparcfit.data import load_sample  # noqa: E402


def load(name):
    out = {}
    with open(ROOT / "results" / name, newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["galaxy"], {})[row["model"]] = {
                k: float(v) for k, v in row.items() if k not in ("galaxy", "model") and v != ""}
    return out


def main():
    sample = load_sample()
    names = [g.name for g in sample]
    Q = np.array([g.Q for g in sample])
    fits = load("fits.csv")
    for name, models in load("fits_ccc.csv").items():
        fits[name].update(models)
    col = lambda model, key: np.array([fits[n][model][key] for n in names])
    models = ("nfw", "lcdm", "mond", "ccc")
    lines = ["Forward CCC model against the three models of the paper", ""]
    lines.append(f"{'':<34s}" + "".join(f"{m.upper():>9s}" for m in models))
    row = lambda label, f: lines.append(f"{label:<34s}" + "".join(f"{f(m):>9s}" for m in models))
    row("free parameters (no bulge)", lambda m: f"{int(np.min(col(m, 'k')))}")
    row("median chi2nu, all", lambda m: f"{np.median(col(m, 'chi2nu')):.2f}")
    row("median chi2nu, Q=1", lambda m: f"{np.median(col(m, 'chi2nu')[Q == 1]):.2f}")
    row("fraction chi2nu < 2 (%)", lambda m: f"{100 * np.mean(col(m, 'chi2nu') < 2):.0f}")
    row("fraction chi2nu > 10 (%)", lambda m: f"{100 * np.mean(col(m, 'chi2nu') > 10):.0f}")
    row("total chi2", lambda m: f"{col(m, 'chi2').sum():.0f}")
    lines.append("")
    lines.append("BIC head to head (CCC preferred / other preferred, strong CCC / strong other)")
    for other in ("mond", "nfw", "lcdm"):
        delta = col("ccc", "bic") - col(other, "bic")
        lines.append(f"  CCC vs {other.upper():<5s} {np.sum(delta < 0):3d} / {np.sum(delta > 0):3d},"
                     f" {np.sum(delta < -6):3d} / {np.sum(delta > 6):3d}   median dBIC {np.median(delta):+.1f}")
    best = np.argmin(np.array([col(m, "bic") for m in ("nfw", "mond", "ccc")]), axis=0)
    lines.append("")
    lines.append("lowest BIC among free NFW / MOND / CCC: " + " / ".join(str(int(np.sum(best == i))) for i in range(3)))
    rho = col("ccc", "log_rho_t_med")
    lines.append("")
    lines.append(f"turn-off density: median log10(rho_t / g cm^-3) = {np.median(rho):.2f}, scatter {np.std(rho):.2f} dex")
    yd = 10 ** col("ccc", "log_Yd_med")
    lines.append("disk M/L in the CCC fits: median {:.2f}, 16-84% {:.2f}-{:.2f}".format(np.median(yd), *np.percentile(yd, [16, 84])))
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "summary_ccc.txt").write_text(text)
    print(text)
    with open(ROOT / "results" / "table_ccc.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["galaxy", "Q", "N", "chi2nu_nfw", "chi2nu_mond", "chi2nu_ccc", "log_rho_t", "Yd_ccc", "dBIC_ccc_minus_mond", "dBIC_ccc_minus_nfw"])
        for i, n in enumerate(names):
            f = fits[n]
            writer.writerow([n, Q[i], sample[i].N, f"{f['nfw']['chi2nu']:.2f}", f"{f['mond']['chi2nu']:.2f}", f"{f['ccc']['chi2nu']:.2f}",
                             f"{f['ccc']['log_rho_t_med']:.2f}", f"{yd[i]:.2f}",
                             f"{f['ccc']['bic'] - f['mond']['bic']:+.1f}", f"{f['ccc']['bic'] - f['nfw']['bic']:+.1f}"])


if __name__ == "__main__":
    main()
