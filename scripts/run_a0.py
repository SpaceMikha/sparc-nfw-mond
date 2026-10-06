"""Profile the MOND acceleration constant galaxy by galaxy -> results/a0.csv."""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sparcfit.data import load_sample  # noqa: E402
from sparcfit.fit import profile_a0  # noqa: E402


def work(gal):
    best, sigma, _, _ = profile_a0(gal)
    return {"galaxy": gal.name, "log_a0": best, "sigma_log_a0": sigma}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    sample = load_sample()
    rows = []
    with Pool(args.processes) as pool:
        for i, row in enumerate(pool.imap(work, sample), 1):
            rows.append(row)
            print(f"[{i:3d}/{len(sample)}] {row['galaxy']} {row['log_a0']:.2f} +- {row['sigma_log_a0']:.2f}", flush=True)
    out = ROOT / "results" / "a0.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
