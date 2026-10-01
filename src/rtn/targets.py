"""Stage 2 regression target: the array cross-reactivity signature.

## Why the target is z-scored before L2 normalisation

`scripts/check_target_variants.py` compared candidate representations. Two
results decided the design:

1. On raw features, the 128-dim vector and the 16-dim DR-only vector give
   *identical* cosine geometry (within 0.966, between 0.820, effective
   dimensionality 1.26). The steady-state DR features are orders of magnitude
   larger than the EMA transient features, so they dominate the norm completely
   and the other 112 numbers contribute nothing. Anyone who normalises the raw
   128-dim vector is silently training on DR alone.
2. Effective dimensionality of the raw target is 1.26. A target with roughly one
   degree of freedom is not a regression problem, it is a scalar. Standardising
   each feature before normalising raises it to 3.25 and raises the
   within/between separation margin from 0.148 to 0.620.

So: standardise per feature across the corpus, then L2-normalise per measurement.
The L2 step removes concentration magnitude and leaves the shape of the array
response, which is the cross-reactivity signature the project is about.

## What this target can and cannot support

It is concentration-invariant by construction, so it cannot be used for
concentration regression. That is deliberate: dataset 270 is a sealed chamber
and dataset 251 is an open wind tunnel, so absolute magnitudes are not
comparable across them anyway (see the Phase 0 audit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_SUFFIXES = [
    "DR", "absDR",
    "EMAi_0.001", "EMAi_0.01", "EMAi_0.1",
    "EMAd_0.001", "EMAd_0.01", "EMAd_0.1",
]


@dataclass
class SignatureScaler:
    """Per-feature standardisation fitted on training measurements only.

    Fitting on all measurements would leak the held-out molecule's response
    distribution into every leave-molecule-out fold, so this is fitted inside
    each fold.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "SignatureScaler":
        v = df[columns].to_numpy(dtype=np.float64)
        return cls(mean=v.mean(0), std=v.std(0) + 1e-9, columns=list(columns))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        v = df[self.columns].to_numpy(dtype=np.float64)
        z = (v - self.mean) / self.std
        n = np.linalg.norm(z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return z / n


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if any(c.endswith("_" + s) for s in FEATURE_SUFFIXES)]


def aggregate_per_molecule(
    signatures: np.ndarray,
    molecules: np.ndarray,
    batches: np.ndarray | None = None,
) -> pd.DataFrame:
    """Collapse measurements to one signature per molecule.

    Median, not mean. `check_target_stability.py` found per-molecule cosine
    minima below zero in every gas, i.e. a small number of measurements have
    sign-flipped responses. Those are almost certainly acquisition faults and a
    mean would let them drag the centroid.

    When `batches` is given, the median is taken per batch first and then across
    batches, so a batch with 3,600 measurements does not outvote one with 161.
    Dataset 270 batch sizes range from 161 to 3,613, so this matters.
    """
    df = pd.DataFrame(signatures)
    df["_mol"] = molecules
    if batches is not None:
        df["_batch"] = batches
        per_batch = df.groupby(["_mol", "_batch"]).median(numeric_only=True)
        out = per_batch.groupby(level="_mol").median()
    else:
        out = df.groupby("_mol").median(numeric_only=True)

    # renormalise: the median of unit vectors is not a unit vector
    v = out.to_numpy()
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pd.DataFrame(v / n, index=out.index)


def confusion_matrix(sig: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine similarity between molecule signatures.

    High off-diagonal entries are the sensor confusions this project exists to
    explain, so this is a headline output, not a diagnostic.
    """
    v = sig.to_numpy()
    c = v @ v.T / np.outer(np.linalg.norm(v, axis=1), np.linalg.norm(v, axis=1))
    return pd.DataFrame(c, index=sig.index, columns=sig.index)
