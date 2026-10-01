"""Export the odour-task layer for the visualisation.

The odour task is the primary experiment now, so this supplies the main
explanatory layer. Three blocks:

  validation  750 attention + 750 IG observations as (sanity, faithfulness)
              points, which is the project's headline finding plotted directly
  showcase    a handful of molecules carrying THREE maps each: attention from
              the trained model, attention from the same architecture with its
              weights reinitialised, and integrated gradients
  table       the Phase 2 baseline numbers quoted in the UI

The showcase is the argument in one picture. On these molecules the trained and
untrained attention maps are near-identical (Spearman up to 1.000) while the
attention still passes a faithfulness test. Showing both maps side by side is
more convincing than any summary statistic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rtn.datasets.openpom import load as load_openpom  # noqa: E402
from rtn.explain import (  # noqa: E402
    attention_scores, integrated_gradients, randomize_weights)
from rtn.odor_eval import scaffold_folds  # noqa: E402
from run_odor_phase4 import fit_model  # noqa: E402

OUT = ROOT / "webapp" / "data"

# Picked from artifacts/odor_phase4_checks.csv: attention is faithful AND
# reproduces almost exactly on an untrained model, while IG stays clean.
SHOWCASE = [
    "CC(CC(C)(C)C#N)c1ccccc1",
    "COCC(C)Cc1ccccc1",
    "COC(=O)Cc1ccc(C(C)(C)C)cc1",
    "O=C(O)C(=O)Cc1ccccc1",
]


def depiction(smiles: str) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    c = mol.GetConformer()
    return {
        "atoms": [{"symbol": a.GetSymbol(),
                   "x": round(c.GetAtomPosition(a.GetIdx()).x, 4),
                   "y": round(c.GetAtomPosition(a.GetIdx()).y, 4)}
                  for a in mol.GetAtoms()],
        "bonds": [{"a": b.GetBeginAtomIdx(), "b": b.GetEndAtomIdx(),
                   "order": int(b.GetBondTypeAsDouble()),
                   "aromatic": b.GetIsAromatic()}
                  for b in mol.GetBonds()],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = pd.read_csv(ROOT / "artifacts" / "odor_phase4_checks.csv")

    # ---- validation scatter -------------------------------------------
    validation = []
    for _, r in checks.iterrows():
        if pd.isna(r.sanity_spearman):
            continue
        validation.append({
            "m": 0 if r.method == "attention" else 1,
            "s": round(float(r.sanity_spearman), 3),   # sanity: high = spurious
            "g": round(float(r.gap), 4),               # faithfulness gap
            "f": bool(r.faithful),
            "n": int(r.n_atoms),
        })

    # ---- showcase: trained vs untrained attention, plus IG -------------
    print("training a model for the showcase maps...", flush=True)
    graphs, descriptors, _ = load_openpom(ROOT / "data" / "raw" / "openpom_gslf.csv")
    folds = scaffold_folds(graphs, 3)
    test = set(folds[1].tolist())
    rest = np.array([i for i in range(len(graphs)) if i not in test])
    rng = np.random.default_rng(1235)
    perm = rng.permutation(rest)
    n_val = max(1, int(0.12 * len(perm)))
    model, val_auc = fit_model([graphs[i] for i in perm[n_val:]],
                               [graphs[i] for i in perm[:n_val]],
                               len(descriptors), seed=0, epochs=60)
    print(f"  val macro AUROC {val_auc:.4f}", flush=True)
    rnd = randomize_weights(model, 0)

    from scipy.stats import spearmanr

    showcase = []
    for smi in SHOWCASE:
        att = attention_scores(model, smi, "odor").atom_scores
        att_r = attention_scores(rnd, smi, "odor").atom_scores
        ig = integrated_gradients(model, smi, "odor").atom_scores
        row = checks[(checks.smiles == smi) & (checks.method == "attention")]
        igrow = checks[(checks.smiles == smi) &
                       (checks.method == "integrated_gradients")]
        showcase.append({
            "smiles": smi,
            **depiction(smi),
            "attention": [round(float(v), 4) for v in att],
            "attention_untrained": [round(float(v), 4) for v in att_r],
            "ig": [round(float(v), 4) for v in ig],
            "rho_trained_vs_untrained": round(
                float(spearmanr(att, att_r).statistic), 3),
            "attention_faithful": bool(row.faithful.iloc[0]) if len(row) else None,
            "attention_gap": round(float(row.gap.mean()), 4) if len(row) else None,
            "ig_rho": round(float(igrow.sanity_spearman.mean()), 3) if len(igrow) else None,
            "ig_gap": round(float(igrow.gap.mean()), 4) if len(igrow) else None,
        })
        print(f"  {smi}  rho(trained, untrained) = "
              f"{showcase[-1]['rho_trained_vs_untrained']:+.3f}", flush=True)

    # ---- Phase 2 table -------------------------------------------------
    p2 = pd.read_csv(ROOT / "artifacts" / "odor_phase2_results.csv")
    table = {}
    for split in p2.split.unique():
        s = p2[p2.split == split]
        table[split] = [
            {"model": m,
             "auroc": round(float(s[s.model == m].macro_auroc.mean()), 4),
             "ap": round(float(s[s.model == m].macro_ap.mean()), 4)}
            for m in ["constant", "ecfp_mlp", "gcn", "gat", "gat_sensor_pretrained"]
            if (s.model == m).any()
        ]

    att = checks[checks.method == "attention"].dropna(subset=["sanity_spearman"])
    ig_ = checks[checks.method == "integrated_gradients"].dropna(subset=["sanity_spearman"])
    stats = {
        "n_obs": int(len(att)),
        "attention": {
            "sanity_mean": round(float(att.sanity_spearman.mean()), 3),
            "fail_rate": round(float((att.sanity_spearman > 0.8).mean()), 3),
            "faithful_rate": round(float(att.faithful.mean()), 3),
            "faithful_and_spurious": round(
                float(((att.faithful) & (att.sanity_spearman > 0.8)).mean()), 3),
        },
        "ig": {
            "sanity_mean": round(float(ig_.sanity_spearman.mean()), 3),
            "fail_rate": round(float((ig_.sanity_spearman > 0.8).mean()), 3),
            "faithful_rate": round(float(ig_.faithful.mean()), 3),
            "faithful_and_spurious": round(
                float(((ig_.faithful) & (ig_.sanity_spearman > 0.8)).mean()), 3),
        },
    }

    bundle = {"validation": validation, "showcase": showcase,
              "table": table, "stats": stats}
    path = OUT / "odor_layer.json"
    path.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"\nwrote {path} ({path.stat().st_size / 1e3:.0f} KB)")
    print(f"  validation points {len(validation)}, showcase {len(showcase)}")
    print(f"  attention faithful-and-spurious {stats['attention']['faithful_and_spurious']:.1%}"
          f" vs IG {stats['ig']['faithful_and_spurious']:.1%}")


if __name__ == "__main__":
    main()
