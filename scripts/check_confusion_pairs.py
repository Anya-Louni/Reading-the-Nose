"""Do the sensor confusion pairs survive common-mode removal?

The Phase 6 "blind spots" framing rests entirely on specific pairs: chemically
distant molecules that the array maps to near-identical signatures. Platform A's
headline pair is ammonia/toluene at 0.77 on the raw target.

Phase 2 showed the raw target carries a large common-mode component on platform
A2, worth 51.5% of centroid variance and correlated with concentration. If a
confusion pair is only close because both molecules load on that shared
component, it is an artefact of concentration and dilution, not a statement
about the sensor's chemical selectivity, and it must not headline the
visualisation.

This recomputes every platform's confusion matrix both ways and reports which
pairs are robust. A pair is only quotable if it is high in *both*.

Unlike the evaluation code this is descriptive, so centring uses all molecules
on the platform; there is no held-out set to leak into.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.analytes import BY_NAME  # noqa: E402
from rtn.assemble import load_platform_a, load_platform_a2, load_platform_b  # noqa: E402
from rtn.targets import SignatureScaler, aggregate_per_molecule  # noqa: E402

# Concentrations as run, for the confound check. UCI 251 ran one per gas.
A2_PPM = {"acetaldehyde": 500, "acetone": 2500, "ammonia": 10000, "benzene": 200,
          "butan-1-ol": 100, "carbon_monoxide": 4000, "ethylene": 500,
          "methane": 1000, "methanol": 200, "toluene": 200}


def tanimoto(a: str, b: str) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return DataStructs.TanimotoSimilarity(
        gen.GetFingerprint(Chem.MolFromSmiles(BY_NAME[a].smiles)),
        gen.GetFingerprint(Chem.MolFromSmiles(BY_NAME[b].smiles)))


def matrices(p) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = SignatureScaler.fit(p.raw, list(p.raw.columns))
    agg = aggregate_per_molecule(p.signatures(sc), p.molecules, p.runs)
    names = list(agg.index)
    V = agg.to_numpy()

    def cosmat(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return pd.DataFrame(M @ M.T, index=names, columns=names)

    return cosmat(V), cosmat(V - V.mean(0))


def top_pairs(cm: pd.DataFrame, k: int = 6):
    names = list(cm.index)
    iu = np.triu_indices(len(names), 1)
    vals = cm.to_numpy()[iu]
    pairs = [(names[i], names[j]) for i, j in zip(*iu)]
    return sorted(zip(vals, pairs), reverse=True)[:k]


def main() -> None:
    plats = {
        "A (UCI 270, sealed chamber)": load_platform_a(
            ROOT / "data_audit" / "cache" / "uci270.zip"),
        "A2 (UCI 251, wind tunnel)": load_platform_a2(
            ROOT / "data" / "processed" / "uci251_features.parquet"),
        "B (Zenodo, commercial nose)": load_platform_b(
            ROOT / "data" / "raw" / "zenodo_drift.zip"),
    }

    survivors = []
    for label, p in plats.items():
        raw, cen = matrices(p)
        names = list(raw.index)
        print(f"\n{'=' * 74}\n{label}  ({len(names)} molecules)\n{'=' * 74}")
        off = lambda m: m.to_numpy()[~np.eye(len(m), dtype=bool)]  # noqa: E731
        print(f"mean off-diagonal   raw {off(raw).mean():+.3f}   "
              f"centred {off(cen).mean():+.3f}")

        print("\ntop confusions, raw target:")
        for v, (a, b) in top_pairs(raw):
            print(f"   {v:+.3f}  {a} / {b}   (centred {cen.loc[a, b]:+.3f}, "
                  f"Tanimoto {tanimoto(a, b):.2f})")

        print("\ntop confusions, common mode removed:")
        for v, (a, b) in top_pairs(cen):
            keep = "ROBUST" if raw.loc[a, b] > 0.5 and v > 0.5 else ""
            print(f"   {v:+.3f}  {a} / {b}   (raw {raw.loc[a, b]:+.3f}, "
                  f"Tanimoto {tanimoto(a, b):.2f})  {keep}")
            if keep:
                survivors.append((label, a, b, raw.loc[a, b], v, tanimoto(a, b)))

        if p.name == "A2":
            iu = np.triu_indices(len(names), 1)
            dppm = np.array([abs(np.log10(A2_PPM[names[i]]) - np.log10(A2_PPM[names[j]]))
                             for i, j in zip(*iu)])
            for tag, m in (("raw", raw), ("centred", cen)):
                r = np.corrcoef(m.to_numpy()[iu], dppm)[0, 1]
                print(f"\n  r(pair similarity, |log10 conc difference|) {tag}: {r:+.3f}")
                print("    (strongly negative means pairs look similar mainly "
                      "because they were run at similar concentrations)")

    print(f"\n{'=' * 74}\nPAIRS QUOTABLE IN THE VISUALISATION (> 0.5 both ways)\n{'=' * 74}")
    if not survivors:
        print("none")
    for label, a, b, r, c, t in sorted(survivors, key=lambda x: -x[4]):
        print(f"  {c:+.3f} centred / {r:+.3f} raw   {a} vs {b}"
              f"   Tanimoto {t:.2f}   [{label.split(' ')[0]}]")


if __name__ == "__main__":
    main()
