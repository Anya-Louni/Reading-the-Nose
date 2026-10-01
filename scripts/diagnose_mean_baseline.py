"""Why is a *constant* prediction molecule-dependent?

Phase 2 reported the mean baseline at +0.707 on the ammonia fold, +0.809 on
toluene, but -0.180 on 2-phenylethanol. A constant prediction should not care
which molecule is held out, so either the per-fold rescaling is doing something
systematic, or the baseline is degenerate and its direction is noise.

Hypothesis under test: signatures are z-scored across training rows, so the
training molecule centroids very nearly sum to zero. Their mean is then a
near-zero vector whose *direction* is numerical residue rather than signal.
If so, the mean baseline is a random-direction baseline wearing a disguise, its
0.302 average is luck over 8 folds, and "nothing beats the mean baseline" is the
wrong way to state the Phase 2 result.

Prints, per fold:
  - the norm of the mean training signature relative to a single signature
  - how much that norm shrinks relative to what independent vectors would give
  - the cosine of the prediction to the held-out truth
and compares against a genuine random-direction control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.assemble import all_molecules, load_platform_a, load_platform_b  # noqa: E402
from rtn.evaluate import _candidate_signatures, _fit_scalers  # noqa: E402


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> None:
    A = load_platform_a(ROOT / "data_audit" / "cache" / "uci270.zip")
    B = load_platform_b(ROOT / "data" / "raw" / "zenodo_drift.zip")
    platforms = {"A": A, "B": B}
    molecules = all_molecules(A, B)

    rng = np.random.default_rng(0)
    rows = []

    for mol in molecules:
        train = {k: p.molecules != mol for k, p in platforms.items()}
        test = {k: p.molecules == mol for k, p in platforms.items()}
        scalers = _fit_scalers(platforms, train)

        for key, p in platforms.items():
            if test[key].sum() == 0 or key not in scalers:
                continue
            cand = _candidate_signatures(p, scalers[key], train[key])
            names = sorted(cand)
            M = np.stack([cand[n] for n in names])       # unit vectors
            mu = M.mean(0)

            sig = p.signatures(scalers[key])
            sel = test[key] & (p.molecules == mol)
            truth = sig[sel].mean(0)
            truth = truth / np.linalg.norm(truth)

            # If the k centroids were independent unit vectors in d dimensions,
            # the mean would have norm about 1/sqrt(k). Much smaller than that
            # means they are actively cancelling, i.e. the mean is degenerate.
            k = len(names)
            expected_if_independent = 1.0 / np.sqrt(k)

            rand = rng.normal(size=p.dim)
            rows.append({
                "fold": mol,
                "platform": key,
                "k_train": k,
                "norm_mu": np.linalg.norm(mu),
                "norm_if_indep": expected_if_independent,
                "shrink_x": expected_if_independent / max(np.linalg.norm(mu), 1e-12),
                "cos_mean_vs_truth": cos(mu, truth),
                "cos_rand_vs_truth": cos(rand, truth),
                "mean_cos_train_pairs": float(np.mean([
                    cos(M[i], M[j]) for i in range(k) for j in range(i + 1, k)])),
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- summary ---")
    print(f"mean |mu|                        : {df.norm_mu.mean():.4f}")
    print(f"mean |mu| expected if independent: {df.norm_if_indep.mean():.4f}")
    print(f"median shrinkage factor          : {df.shrink_x.median():.1f}x")
    print(f"mean cosine, mean-baseline       : {df.cos_mean_vs_truth.mean():+.4f}")
    print(f"mean cosine, random direction    : {df.cos_rand_vs_truth.mean():+.4f}")
    print(f"std  cosine, mean-baseline       : {df.cos_mean_vs_truth.std():.4f}")
    print(f"std  cosine, random direction    : {df.cos_rand_vs_truth.std():.4f}")

    print("\nrandom-direction control over 2000 draws per fold:")
    out = []
    for mol in molecules:
        train = {k: p.molecules != mol for k, p in platforms.items()}
        test = {k: p.molecules == mol for k, p in platforms.items()}
        scalers = _fit_scalers(platforms, train)
        for key, p in platforms.items():
            if test[key].sum() == 0 or key not in scalers:
                continue
            sig = p.signatures(scalers[key])
            sel = test[key] & (p.molecules == mol)
            truth = sig[sel].mean(0)
            truth = truth / np.linalg.norm(truth)
            draws = rng.normal(size=(2000, p.dim))
            c = draws @ truth / np.linalg.norm(draws, axis=1)
            out.append({"fold": mol, "platform": key,
                        "rand_mean": c.mean(), "rand_p95": np.percentile(c, 95),
                        "actual_mean_baseline": cos(
                            np.stack([v for v in _candidate_signatures(
                                p, scalers[key], train[key]).values()]).mean(0), truth)})
    ctrl = pd.DataFrame(out)
    print(ctrl.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    beat = (ctrl.actual_mean_baseline.abs() > ctrl.rand_p95).mean()
    print(f"\nfolds where |mean baseline| exceeds the random 95th percentile: {beat:.2f}")


if __name__ == "__main__":
    main()
