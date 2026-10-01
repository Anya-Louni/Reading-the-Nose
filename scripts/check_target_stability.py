"""Is the array response pattern a learnable function of the molecule at all?

Before committing to the pipeline, check that the cross-reactivity signature we
plan to predict is more consistent within a molecule than it is between
molecules. If within-molecule spread swamps between-molecule spread, the target
is dominated by drift and concentration rather than by chemistry, and no model
of the molecule can succeed. Vergara's dataset is known to have a batch/time
confound (arXiv 2108.08793), so this is not a formality.

Target definition under test: the steady-state response DR of the 16 sensors,
L2-normalised per measurement to remove concentration magnitude, leaving the
shape of the array response.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_audit"))
from parse_uci270 import download, load  # noqa: E402


def response_shape(df: pd.DataFrame) -> np.ndarray:
    """L2-normalised 16-dim steady-state (DR) response per measurement."""
    dr_cols = [c for c in df.columns if c.endswith("_DR") and "abs" not in c]
    assert len(dr_cols) == 16, f"expected 16 DR columns, got {len(dr_cols)}"
    v = df[dr_cols].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    df = load(download(here / "data_audit" / "cache" / "uci270.zip"))
    shapes = response_shape(df)
    gases = df["gas_name"].to_numpy()

    centroids = {g: shapes[gases == g].mean(0) for g in sorted(set(gases))}

    print("Within-molecule cosine similarity to own centroid")
    print("(1.0 = perfectly reproducible signature, 0 = no signal)\n")
    within = {}
    for g, c in centroids.items():
        v = shapes[gases == g]
        cs = v @ c / (np.linalg.norm(v, axis=1) * np.linalg.norm(c))
        within[g] = cs
        print(f"  {g:14s} n={len(v):5d}  mean={cs.mean():.3f}  p10={np.percentile(cs,10):.3f}  min={cs.min():.3f}")

    print("\nBetween-molecule centroid cosine similarity")
    names = sorted(centroids)
    m = np.array([[centroids[a] @ centroids[b] / (np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]))
                   for b in names] for a in names])
    print("      " + " ".join(f"{n[:6]:>6s}" for n in names))
    for i, a in enumerate(names):
        print(f"{a[:6]:>6s} " + " ".join(f"{m[i, j]:6.2f}" for j in range(len(names))))

    off = m[~np.eye(len(names), dtype=bool)]
    all_within = np.concatenate(list(within.values()))
    print(f"\nmean within-molecule similarity : {all_within.mean():.3f}")
    print(f"mean between-molecule similarity: {off.mean():.3f}")
    print(f"separation margin              : {all_within.mean() - off.mean():+.3f}")

    print("\nPer-batch drift of the signature (cosine to the global centroid)")
    for g in names:
        sub = df[df.gas_name == g]
        s = response_shape(sub)
        c = centroids[g]
        by_batch = pd.Series(s @ c / (np.linalg.norm(s, axis=1) * np.linalg.norm(c)),
                             index=sub.batch.to_numpy()).groupby(level=0).mean()
        print(f"  {g:14s} " + " ".join(f"b{b}:{v:.2f}" for b, v in by_batch.items()))

    print("\nNearest-centroid classification accuracy (a floor on learnability):")
    sim = shapes @ np.array([centroids[n] for n in names]).T
    pred = np.array(names)[sim.argmax(1)]
    print(f"  {(pred == gases).mean():.3f} over {len(gases)} measurements, 6 classes")


if __name__ == "__main__":
    main()
