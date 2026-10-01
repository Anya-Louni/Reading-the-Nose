"""Which response representation should be the Stage 2 regression target?

check_target_stability.py showed the steady-state (DR) shape is reproducible but
collapses to roughly two clusters. That may be the real cross-reactivity story,
or it may just be that DR discards the transient dynamics the sensor literature
relies on. This compares candidate targets on separability and on how many
dimensions of the target actually carry molecule-specific variance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_audit"))
from parse_uci270 import download, load  # noqa: E402

FEATS = ["DR", "absDR", "EMAi_0.001", "EMAi_0.01", "EMAi_0.1",
         "EMAd_0.001", "EMAd_0.01", "EMAd_0.1"]


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def evaluate(name: str, v: np.ndarray, gases: np.ndarray) -> dict:
    names = sorted(set(gases))
    cents = np.array([v[gases == g].mean(0) for g in names])

    def cos(a, b):
        return a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))

    within = np.concatenate([
        cos(v[gases == g], cents[i:i + 1]).ravel() for i, g in enumerate(names)
    ])
    cm = cos(cents, cents)
    between = cm[~np.eye(len(names), dtype=bool)]

    acc = (np.array(names)[cos(v, cents).argmax(1)] == gases).mean()

    # how many dimensions of between-molecule variation actually matter
    ev = np.linalg.svd(cents - cents.mean(0), compute_uv=False) ** 2
    ev = ev / ev.sum()
    eff_dim = float(np.exp(-(ev[ev > 0] * np.log(ev[ev > 0])).sum()))

    worst = np.unravel_index(np.argmax(cm - np.eye(len(names)) * 2), cm.shape)
    return {
        "target": name,
        "dim": v.shape[1],
        "within": within.mean(),
        "between": between.mean(),
        "margin": within.mean() - between.mean(),
        "nc_acc": acc,
        "eff_dim": eff_dim,
        "worst_pair": f"{names[worst[0]]}/{names[worst[1]]} {cm[worst]:.3f}",
    }


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    df = load(download(here / "data_audit" / "cache" / "uci270.zip"))
    gases = df["gas_name"].to_numpy()

    def cols(suffixes):
        return [c for c in df.columns
                if any(c.endswith("_" + s) for s in suffixes)]

    variants = {
        "DR only (16d)": cols(["DR"])[:16],
        "DR + absDR (32d)": cols(["DR", "absDR"]),
        "EMA rise (48d)": cols(["EMAi_0.001", "EMAi_0.01", "EMAi_0.1"]),
        "EMA decay (48d)": cols(["EMAd_0.001", "EMAd_0.01", "EMAd_0.1"]),
        "full 128d": [c for c in df.columns if any(c.endswith("_" + f) for f in FEATS)],
    }

    rows = []
    for name, c in variants.items():
        v = df[c].to_numpy(dtype=np.float64)
        assert v.shape[1] == int(name.split("(")[-1].rstrip("d)")) if "(" in name else True
        rows.append(evaluate(name + " raw", v, gases))
        rows.append(evaluate(name + " L2", l2norm(v), gases))
        # per-block standardisation then L2, to stop absDR magnitudes dominating
        z = (v - v.mean(0)) / (v.std(0) + 1e-9)
        rows.append(evaluate(name + " z+L2", l2norm(z), gases))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nBest by nearest-centroid accuracy:")
    print(out.sort_values("nc_acc", ascending=False).head(5).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
