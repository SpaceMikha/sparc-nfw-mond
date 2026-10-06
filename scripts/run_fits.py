"""Fit every SPARC galaxy with the three models and write results/fits.csv."""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparcfit.data import load_sample  # noqa: E402
from sparcfit.fit import fit  # noqa: E402
from sparcfit.models import MODELS  # noqa: E402


def work(gal):
    return [fit(gal, model) for model in MODELS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--galaxies", nargs="*", help="restrict to these galaxy names")
    args = parser.parse_args()

    sample = load_sample()
    if args.galaxies:
        sample = [g for g in sample if g.name in args.galaxies]

    rows = []
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap(work, sample), 1):
            rows.extend(result)
            print(f"[{i:3d}/{len(sample)}] {result[0]['galaxy']}", flush=True)

    keys = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    out = ROOT / "results" / "fits.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
