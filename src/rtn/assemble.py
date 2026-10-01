"""Assemble the Stage 2 training set: one example per (molecule, acquisition run).

A training example is a molecule's array signature *as measured in one
acquisition batch* (UCI 270) or *on one day* (Zenodo). Rationale:

- Training on all 13,910 individual measurements would just teach the model the
  per-molecule conditional mean, which is the centroid we would have aggregated
  to anyway, at ~250x the compute.
- Aggregating all the way to one signature per molecule throws away the real
  measurement spread and leaves 6 and 3 training rows, which cannot support a
  train/validation split at all.
- Per (molecule, run) keeps the drift spread as natural augmentation and gives a
  model something to be robust to, while staying small enough to train many
  folds on CPU.

The scaler is fitted inside each fold on training rows only, so a held-out
molecule never contributes to the standardisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rtn.analytes import BY_NAME
from rtn.targets import SignatureScaler


@dataclass
class PlatformData:
    """Per-run signatures for one sensor platform."""

    name: str
    molecules: np.ndarray   # (n,) analyte name per row
    runs: np.ndarray        # (n,) acquisition batch / day index per row
    raw: pd.DataFrame       # (n, d) unstandardised features, aligned to rows
    dim: int

    def __len__(self) -> int:
        return len(self.molecules)

    def signatures(self, scaler: SignatureScaler) -> np.ndarray:
        v = scaler.transform(self.raw)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n


def _collapse(df: pd.DataFrame, feature_cols: list[str],
              molecule_col: str, run_col: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Median-collapse raw measurements to one row per (molecule, run)."""
    g = df.groupby([molecule_col, run_col], observed=True)
    agg = g[feature_cols].median()
    idx = agg.index.to_frame(index=False)
    return (idx[molecule_col].to_numpy(),
            idx[run_col].to_numpy(),
            agg.reset_index(drop=True))


def load_platform_a(uci270_zip: Path) -> PlatformData:
    """UCI 270 only. UCI 251 is folded in separately once its extraction lands."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data_audit"))
    from parse_uci270 import load as load270

    from rtn.targets import feature_columns

    df = load270(uci270_zip)
    cols = feature_columns(df)
    mols, runs, agg = _collapse(df, cols, "gas_name", "batch")
    return PlatformData("A", mols, runs, agg, dim=len(cols))


def load_platform_b(zenodo_zip: Path) -> PlatformData:
    from rtn.datasets.zenodo_drift import feature_columns, load

    df = load(zenodo_zip)
    cols = feature_columns(df)
    mols, runs, agg = _collapse(df, cols, "analyte", "day_index")
    return PlatformData("B", mols, runs, agg, dim=len(cols))


def all_molecules(*platforms: PlatformData) -> list[str]:
    out: set[str] = set()
    for p in platforms:
        out.update(p.molecules.tolist())
    unknown = out - set(BY_NAME)
    if unknown:
        raise ValueError(f"molecules missing from the analyte registry: {sorted(unknown)}")
    return sorted(out)


def load_platform_a2(
    parquet: Path,
    run_key: tuple[str, ...] = ("location", "heater_v", "fan"),
) -> PlatformData:
    """UCI 251 wind tunnel, from the streamed feature table.

    Separate from platform A1 (UCI 270) on purpose. The two share sensor types
    but not feature definitions: UCI 270 ships the authors' pre-extracted DR and
    EMA features, while these are computed in `scripts/extract_uci251.py` from
    the raw traces (max_r, mean_r, p90_r, std_r, area_r, rise_slope, t_to_max,
    recovery). Concatenating them would put incommensurable quantities in one
    output space. A shared encoder with one head each is the honest structure.

    `run_key` defines what counts as one acquisition run. The default treats
    each (location, heater voltage, fan speed) as its own run, which uses all
    the data and lets condition-to-condition variation act as the spread the
    model has to be robust to. Heater voltage changes the sensor operating point
    materially, so check the variance decomposition before trusting this: if
    voltage explains more of the signature than molecule identity does, restrict
    to a single voltage instead.
    """
    df = pd.read_parquet(parquet)
    cols = [c for c in df.columns if c.startswith("s") and "_" in c
            and c.split("_")[0][1:].isdigit()]
    if not cols:
        raise ValueError("no sensor feature columns found in the parquet")

    run = df[list(run_key)].astype(str).agg("|".join, axis=1)
    work = df[["analyte"] + cols].copy()
    work["_run"] = run.to_numpy()
    mols, runs, agg = _collapse(work, cols, "analyte", "_run")
    return PlatformData("A2", mols, runs, agg, dim=len(cols))


def variance_decomposition(parquet: Path) -> pd.DataFrame:
    """How much of the response signature is molecule vs operating condition?

    If heater voltage or fan speed explains as much variance as the analyte
    does, then a per-molecule signature averaged over conditions is a fiction
    and the run definition needs narrowing. Reports eta-squared per factor on
    the standardised, L2-normalised signature.
    """
    from rtn.targets import SignatureScaler

    df = pd.read_parquet(parquet)
    cols = [c for c in df.columns if c.startswith("s") and "_" in c
            and c.split("_")[0][1:].isdigit()]
    sc = SignatureScaler.fit(df, cols)
    sig = sc.transform(df)

    total = ((sig - sig.mean(0)) ** 2).sum()
    rows = []
    for factor in ["analyte", "heater_v", "fan", "location"]:
        between = 0.0
        for _, idx in df.groupby(factor).groups.items():
            g = sig[df.index.get_indexer(idx)]
            between += len(g) * ((g.mean(0) - sig.mean(0)) ** 2).sum()
        rows.append({"factor": factor, "eta_squared": between / total,
                     "n_levels": df[factor].nunique()})
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
