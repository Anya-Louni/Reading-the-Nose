"""Platform B: Zenodo 15681119, Long-Term Drift Behavior of Electronic Nose.

62-sensor commercial MOX nose, 3 analytes (diacetyl, 2-phenylethanol, ethanol).
The record describes 40 acquisition days; `extracted_features.csv` actually holds
624 measurements over 36 distinct days spanning 355 days. Quote 36, not 40.
CC BY 4.0.

The archive ships `extracted_features.csv` alongside the raw per-day CSVs. We use
it. It is space-separated (not comma, despite the extension) and holds 624 rows
of 62 sensors x 4 feature blocks:

    mean_R<i>            steady-state resistance during exposure
    rel_diff_R<i>        relative change from baseline
    sample_startslope_R<i>  initial transient slope
    rec_lvl_R<i>         recovery level after exposure

That is the same shape of information as UCI 270's steady-state + transient
split, which is what makes a shared encoder with per-platform heads reasonable
even though the sensors themselves have nothing in common.

Ethanol (`EtOH`) is the only analyte shared with platform A, and it is the one
analyte here recorded without a concentration in the Class field. It is the
single point of comparison between the two heads, which is not enough to
identify a mapping between the platforms. Treat any cross-platform claim as
illustrative, not quantitative.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_ENTRY = "extracted_features.csv"

# `Class` values in the archive mapped onto the analyte registry.
CLASS_TO_ANALYTE = {
    "Diacetyl": "diacetyl",
    "Phenylethanol": "2-phenylethanol",
    "EtOH": "ethanol",
}

FEATURE_PREFIXES = ["mean_R", "rel_diff_R", "sample_startslope_R", "rec_lvl_R"]

# The steady-state columns carry a unit suffix ("mean_R1[Ohm]") while the other
# three blocks do not ("rel_diff_R1"). Match the sensor index explicitly so a
# prefix test does not silently drop a whole block, and so that rel_diff_T01 /
# rel_diff_H01 (chamber temperature and humidity) are excluded.
FEATURE_RE = re.compile(
    r"^(?P<block>mean_R|rel_diff_R|sample_startslope_R|rec_lvl_R)(?P<idx>\d+)(\[[^\]]*\])?$"
)


def _feature_cols(columns) -> list[str]:
    return [c for c in columns if FEATURE_RE.match(c)]


def load(zip_path: Path) -> pd.DataFrame:
    """Return a tidy frame: analyte, concentration, date, then 248 sensor features.

    Temperature and humidity columns are dropped. They are chamber conditions,
    not responses to the molecule, and including them would let the model fit
    the acquisition environment instead of the chemistry.
    """
    with zipfile.ZipFile(zip_path) as z:
        entry = next(n for n in z.namelist() if n.endswith(FEATURES_ENTRY))
        raw = z.open(entry).read().decode("utf-8", "replace")

    # The file is space-separated, but the Class field is a quoted string that
    # itself contains a space ("Diacetyl 0.1ppm"). A plain \s+ split shears it in
    # two and silently shifts the trailing columns, so parse with the csv module
    # which honours the quoting.
    reader = csv.reader(io.StringIO(raw), delimiter=" ", quotechar='"',
                        skipinitialspace=True)
    rows = [r for r in reader if r]
    header, body = rows[0], rows[1:]
    df = pd.DataFrame(body, columns=header)

    feature_cols = _feature_cols(df.columns)
    expected = 62 * len(FEATURE_PREFIXES)
    if len(feature_cols) != expected:
        raise ValueError(f"expected {expected} sensor features, found {len(feature_cols)}")

    class_col = next(c for c in df.columns if c.lower() == "class")
    date_col = next(c for c in df.columns if c.lower().startswith("date"))

    # Class is "<substance>" or "<substance> <concentration>ppm".
    substance = df[class_col].str.split().str[0]
    conc = (df[class_col].str.extract(r"([\d.]+)\s*ppm", expand=False)
            .astype(float))

    unknown = set(substance) - set(CLASS_TO_ANALYTE)
    if unknown:
        raise ValueError(f"unmapped Class values: {sorted(unknown)}")

    out = pd.DataFrame({
        "analyte": substance.map(CLASS_TO_ANALYTE),
        "class_raw": df[class_col],
        "concentration_ppm": conc,
        "date": pd.to_datetime(df[date_col], format="%y-%m-%d"),
    })
    out["day_index"] = (out["date"] - out["date"].min()).dt.days
    features = df[feature_cols].astype(np.float64)
    return pd.concat([out, features], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return _feature_cols(df.columns)
