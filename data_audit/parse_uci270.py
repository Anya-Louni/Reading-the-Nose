"""Parse UCI 270 (Gas Sensor Array Drift at Different Concentrations) from raw files.

Do NOT use ucimlrepo.fetch_ucirepo(id=270). It treats the "gas;concentration" label
token as feature 1 and shifts every column by one, losing both the gas label and the
concentration. Verified 2026-09-07.

Raw line format:  "<gas>;<concentration> 1:<v1> 2:<v2> ... 128:<v128>"
gas codes: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene

Feature layout (128 = 16 sensors x 8 features), per sensor in order:
    DR, |DR|, EMAi(0.001), EMAi(0.01), EMAi(0.1), EMAd(0.001), EMAd(0.01), EMAd(0.1)
Sensor order is 4x TGS2600, 4x TGS2602, 4x TGS2610, 4x TGS2620.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

URL = (
    "https://archive.ics.uci.edu/static/public/270/"
    "gas+sensor+array+drift+dataset+at+different+concentrations.zip"
)

GAS_NAMES = {
    1: "ethanol",
    2: "ethylene",
    3: "ammonia",
    4: "acetaldehyde",
    5: "acetone",
    6: "toluene",
}

GAS_SMILES = {
    "ethanol": "CCO",
    "ethylene": "C=C",
    "ammonia": "N",
    "acetaldehyde": "CC=O",
    "acetone": "CC(C)=O",
    "toluene": "Cc1ccccc1",
}

SENSOR_TYPES = ["TGS2600"] * 4 + ["TGS2602"] * 4 + ["TGS2610"] * 4 + ["TGS2620"] * 4
FEATURE_NAMES = [
    "DR",
    "absDR",
    "EMAi_0.001",
    "EMAi_0.01",
    "EMAi_0.1",
    "EMAd_0.001",
    "EMAd_0.01",
    "EMAd_0.1",
]


def download(dest: Path) -> Path:
    """Fetch the raw zip if not already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urlopen(URL) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def load(zip_path: Path) -> pd.DataFrame:
    """Return a tidy frame: batch, gas, gas_name, smiles, concentration_ppmv, f0..f127."""
    rows, meta = [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist(), key=lambda n: int("".join(c for c in n if c.isdigit()))):
            batch = int("".join(c for c in name if c.isdigit()))
            for line in io.TextIOWrapper(z.open(name), "utf-8"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                gas_str, conc_str = parts[0].split(";")
                gas = int(gas_str)
                # values are "index:value"; index is 1-based and dense in this dataset
                vals = [float(p.split(":", 1)[1]) for p in parts[1:]]
                if len(vals) != 128:
                    raise ValueError(f"{name}: expected 128 features, got {len(vals)}")
                rows.append(vals)
                meta.append((batch, gas, GAS_NAMES[gas], GAS_SMILES[GAS_NAMES[gas]], float(conc_str)))

    cols = [f"{s}_{i % 4}_{f}" for i, s in enumerate(SENSOR_TYPES) for f in FEATURE_NAMES]
    df = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=cols)
    m = pd.DataFrame(meta, columns=["batch", "gas", "gas_name", "smiles", "concentration_ppmv"])
    return pd.concat([m, df], axis=1)


if __name__ == "__main__":
    here = Path(__file__).parent
    df = load(download(here / "cache" / "uci270.zip"))
    print(f"{len(df)} measurements, {df.shape[1] - 5} sensor features")
    summary = df.groupby("gas_name")["concentration_ppmv"].agg(["count", "min", "max", "nunique"])
    print(summary.sort_values("count", ascending=False))
    print("\nper batch:")
    print(df.groupby("batch").size().to_string())
