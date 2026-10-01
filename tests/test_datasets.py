"""Loader regression tests.

Both of these cover parsing bugs that were live in this repo and that fail
*silently* rather than loudly. A shifted column produces a plausible-looking
number, so an assertion is the only thing standing between us and a results
table built on mislabelled data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data_audit"))

RAW = ROOT / "data" / "raw"
UCI270 = ROOT / "data_audit" / "cache" / "uci270.zip"
ZENODO = RAW / "zenodo_drift.zip"

pytestmark = pytest.mark.skipif(
    not (UCI270.exists() and ZENODO.exists()),
    reason="raw archives not downloaded",
)


def test_uci270_labels_survive_parsing():
    """ucimlrepo shifts every column by one and loses the label entirely.

    Our parser must recover 6 gases and a concentration for every row.
    """
    from parse_uci270 import load

    df = load(UCI270)
    assert len(df) == 13910
    assert set(df.gas_name) == {"ethanol", "ethylene", "ammonia",
                                "acetaldehyde", "acetone", "toluene"}
    assert df.concentration_ppmv.notna().all()
    assert df.concentration_ppmv.min() > 0
    assert len([c for c in df.columns if c.startswith("TGS")]) == 128
    # counts are stable and worth pinning; a change here means the parse moved
    assert df.gas_name.value_counts().to_dict() == {
        "acetone": 3009, "ethylene": 2926, "ethanol": 2565,
        "acetaldehyde": 1936, "toluene": 1833, "ammonia": 1641,
    }


def test_zenodo_quoted_class_field_is_not_split():
    """Class is `"Diacetyl 0.1ppm"`: a quoted field containing a space.

    Splitting on whitespace shears it in two and shifts the date column, which
    is how this was first written. If the substance names come back as bare
    concentrations or dates, the quoting has regressed.
    """
    from rtn.datasets.zenodo_drift import feature_columns, load

    df = load(ZENODO)
    assert len(df) == 624
    assert set(df.analyte) == {"diacetyl", "2-phenylethanol", "ethanol"}
    assert len(feature_columns(df)) == 62 * 4
    assert df[feature_columns(df)].isna().sum().sum() == 0
    # ethanol is the one analyte with no concentration in the Class field
    assert df.loc[df.analyte == "ethanol", "concentration_ppm"].isna().all()
    assert df.loc[df.analyte != "ethanol", "concentration_ppm"].notna().all()
    assert df.date.dt.year.between(2024, 2025).all()


def test_zenodo_feature_blocks_are_all_present():
    """`mean_R1[Ohm]` has a unit suffix, the other three blocks do not.

    A naive prefix match drops the whole steady-state block and leaves 186
    features instead of 248, which still trains and still looks fine.
    """
    from rtn.datasets.zenodo_drift import feature_columns, load

    cols = feature_columns(load(ZENODO))
    for prefix in ["mean_R", "rel_diff_R", "sample_startslope_R", "rec_lvl_R"]:
        block = [c for c in cols if c.startswith(prefix)]
        assert len(block) == 62, f"{prefix}: expected 62, got {len(block)}"
    # chamber temperature and humidity must not leak in as responses
    assert not any("T01" in c or "H01" in c for c in cols)
