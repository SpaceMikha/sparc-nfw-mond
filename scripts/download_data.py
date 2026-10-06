"""Download the public SPARC files (Lelli, McGaugh & Schombert 2016) into data/."""
import io
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://astroweb.case.edu/SPARC/"
DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA.mkdir(exist_ok=True)
    master = urllib.request.urlopen(BASE + "SPARC_Lelli2016c.mrt").read()
    (DATA / "SPARC_Lelli2016c.mrt").write_bytes(master)
    archive = urllib.request.urlopen(BASE + "Rotmod_LTG.zip").read()
    (DATA / "rotmod").mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(archive)).extractall(DATA / "rotmod")
    print(f"SPARC data written to {DATA}")


if __name__ == "__main__":
    main()
