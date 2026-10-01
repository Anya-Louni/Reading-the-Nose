"""Export everything the Smell Galaxy app needs, as one JSON bundle.

Framing (locked): the visualisation is about the **sensor's blind spots**, not
the model's confidence. Phase 2 found no generalisation to unseen molecules, so
a viz built on model confidence would be built on the one claim that failed.
What is real and well evidenced is the confusion geometry: chemically distant
molecules that a cheap array maps to near-identical signatures.

The GAT is used only where it demonstrably works: producing the embedding that
lays out the 4,975-molecule OpenPOM background (validation macro AUC 0.80 on
odor descriptors). It is not used to make any claim about sensor response.

Bundle contents:
  meta         provenance and the numbers quoted in the UI
  galaxy       3D coordinates for the OpenPOM background + the analytes
  confusions   per-platform pairwise cosine, raw and common-mode removed
  pairs        the five quotable confusion pairs, with caveat flags
  molecules    2D depiction coordinates + per-atom IG attribution + verdicts
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

from rtn.analytes import ANALYTES, BY_NAME  # noqa: E402
from rtn.assemble import load_platform_a, load_platform_a2, load_platform_b  # noqa: E402
from rtn.datasets.openpom import load as load_openpom  # noqa: E402
from rtn.model import GraphEncoder  # noqa: E402
from rtn.molgraph import featurize  # noqa: E402
from rtn.targets import SignatureScaler, aggregate_per_molecule  # noqa: E402

OUT = ROOT / "webapp" / "data"

# Locked in the Phase 4 / confusion-pair review. `clean` means the pair is not
# confounded with concentration; only UCI 270 qualifies, because UCI 251 ran a
# single concentration per gas.
QUOTABLE_PAIRS = [
    ("ammonia", "toluene", "A", True),
    ("carbon_monoxide", "ethylene", "A2", False),
    ("butan-1-ol", "toluene", "A2", False),
    ("butan-1-ol", "methanol", "A2", False),
    ("methane", "methanol", "A2", False),
]

# Molecules whose IG attribution passed BOTH the Adebayo randomisation check and
# the deletion faithfulness test. Everything else is shown without a heatmap.
IG_VALIDATED = {"ethanol", "acetaldehyde", "acetone", "butan-1-ol"}
IG_EXCLUDED = {
    "diacetyl": "FAILS the weight-randomisation sanity check (rho +0.943): "
                "nearly the same map comes out of an untrained model",
    "toluene": "borderline on the sanity check (rho +0.702)",
    "2-phenylethanol": "not faithful; masking its top-ranked atoms degrades the "
                       "prediction no faster than masking random ones",
    "benzene": "all six atoms are equivalent by symmetry, so any per-atom "
               "attribution must be uniform",
}


def embed_corpus(encoder, graphs, batch_size: int = 256) -> np.ndarray:
    from torch_geometric.loader import DataLoader
    encoder.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size):
            out.append(encoder(batch).cpu().numpy())
    return np.concatenate(out)


def project_3d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """PCA to 50 dims, then t-SNE to 3. UMAP is not installed; t-SNE is enough
    for a layout whose only job is to be legible."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    Xp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    Y = TSNE(n_components=3, random_state=seed, perplexity=30,
             init="pca", max_iter=1000).fit_transform(Xp)
    Y = Y - Y.mean(0)
    return Y / np.abs(Y).max() * 100.0


def depiction(smiles: str) -> dict:
    """2D coordinates and bonds from RDKit, so the browser needs no chem library."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atoms = [{"symbol": a.GetSymbol(),
              "x": round(conf.GetAtomPosition(a.GetIdx()).x, 4),
              "y": round(conf.GetAtomPosition(a.GetIdx()).y, 4),
              "aromatic": a.GetIsAromatic()}
             for a in mol.GetAtoms()]
    bonds = [{"a": b.GetBeginAtomIdx(), "b": b.GetEndAtomIdx(),
              "order": int(b.GetBondTypeAsDouble()),
              "aromatic": b.GetIsAromatic()}
             for b in mol.GetBonds()]
    return {"atoms": atoms, "bonds": bonds}


def confusion_blocks() -> dict:
    plats = {
        "A": ("UCI 270, sealed chamber", load_platform_a(
            ROOT / "data_audit" / "cache" / "uci270.zip")),
        "A2": ("UCI 251, open wind tunnel", load_platform_a2(
            ROOT / "data" / "processed" / "uci251_features.parquet")),
        "B": ("Zenodo 15681119, commercial nose", load_platform_b(
            ROOT / "data" / "raw" / "zenodo_drift.zip")),
    }
    out = {}
    for key, (label, p) in plats.items():
        sc = SignatureScaler.fit(p.raw, list(p.raw.columns))
        agg = aggregate_per_molecule(p.signatures(sc), p.molecules, p.runs)
        names = list(agg.index)
        V = agg.to_numpy()

        def cos(M):
            M = M / np.linalg.norm(M, axis=1, keepdims=True)
            return np.round(M @ M.T, 4).tolist()

        out[key] = {"label": label, "molecules": names,
                    "raw": cos(V), "centered": cos(V - V.mean(0))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading OpenPOM corpus...", flush=True)
    graphs, descriptors, _ = load_openpom(ROOT / "data" / "raw" / "openpom_gslf.csv")

    encoder = GraphEncoder(hidden=128, layers=3, heads=4, attention=True)
    encoder.load_state_dict(torch.load(ROOT / "artifacts" / "encoder_pretrained_seed0.pt",
                                       map_location="cpu"))
    print(f"embedding {len(graphs)} background molecules...", flush=True)
    Xbg = embed_corpus(encoder, graphs)

    analyte_graphs = [featurize(a.smiles, name=a.name).data for a in ANALYTES]
    Xan = embed_corpus(encoder, analyte_graphs)

    print("projecting to 3D...", flush=True)
    Y = project_3d(np.concatenate([Xbg, Xan]))
    Ybg, Yan = Y[:len(Xbg)], Y[len(Xbg):]

    # dominant descriptor per background molecule, for colouring
    counts = pd.Series({d: sum(1 for g in graphs if g.y[0, i] > 0)
                        for i, d in enumerate(descriptors)})
    top_desc = list(counts.sort_values(ascending=False).head(12).index)
    bg = []
    for g, xyz in zip(graphs, Ybg):
        labels = [d for i, d in enumerate(descriptors) if g.y[0, i] > 0]
        primary = next((d for d in top_desc if d in labels), "other")
        bg.append({"s": g.smiles, "p": [round(float(v), 2) for v in xyz],
                   "c": primary, "n": int(g.x.shape[0])})

    analytes = []
    for a, xyz in zip(ANALYTES, Yan):
        analytes.append({
            "name": a.name, "smiles": a.smiles, "heavy_atoms": a.heavy_atoms,
            "pos": [round(float(v), 2) for v in xyz],
            "platforms": list(a.platform), "sources": list(a.sources),
            "in_openpom": a.in_openpom, "note": a.note,
        })

    # IG attributions, averaged over seeds
    attr = pd.read_csv(ROOT / "artifacts" / "phase4_attributions.csv")
    ig = (attr[attr.method == "integrated_gradients"]
          .groupby(["molecule", "atom_idx"]).score.mean())

    molecules = {}
    for a in ANALYTES:
        d = depiction(a.smiles)
        scores = None
        if a.name in ig.index.get_level_values(0):
            s = ig.loc[a.name].sort_index().to_numpy()
            if len(s) == len(d["atoms"]):
                scores = [round(float(v), 4) for v in s]
        molecules[a.name] = {
            **d,
            "smiles": a.smiles,
            "ig": scores,
            "ig_validated": a.name in IG_VALIDATED,
            "ig_caveat": IG_EXCLUDED.get(a.name),
        }

    pairs = []
    conf = confusion_blocks()
    for x, y, plat, clean in QUOTABLE_PAIRS:
        names = conf[plat]["molecules"]
        i, j = names.index(x), names.index(y)
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        tan = DataStructs.TanimotoSimilarity(
            gen.GetFingerprint(Chem.MolFromSmiles(BY_NAME[x].smiles)),
            gen.GetFingerprint(Chem.MolFromSmiles(BY_NAME[y].smiles)))
        pairs.append({
            "a": x, "b": y, "platform": plat,
            "raw": conf[plat]["raw"][i][j],
            "centered": conf[plat]["centered"][i][j],
            "tanimoto": round(tan, 3),
            "concentration_clean": clean,
        })

    bundle = {
        "meta": {
            "n_background": len(bg),
            "n_analytes": len(analytes),
            "encoder": "GATv2, 3 layers, 128 hidden, 4 heads, pretrained on "
                       "OpenPOM GS-LF (validation macro AUC 0.803)",
            "framing": "sensor blind spots, not model confidence",
            "attention_note": "Attention maps exist but are not shown. On "
                              "held-out molecules they correlate +0.79 on average "
                              "with maps from a weight-randomised model and 60.4% "
                              "fail that check outright, so they are largely "
                              "reproducible from an untrained network. Atom "
                              "heatmaps are integrated gradients, and only where "
                              "they pass both the randomisation and faithfulness "
                              "checks.",
            "descriptor_palette": top_desc,
        },
        "galaxy": {"background": bg, "analytes": analytes},
        "confusions": conf,
        "pairs": pairs,
        "molecules": molecules,
    }

    path = OUT / "galaxy.json"
    path.write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"  background {len(bg)}, analytes {len(analytes)}, pairs {len(pairs)}")
    print(f"  IG heatmaps shown for: {sorted(IG_VALIDATED)}")


if __name__ == "__main__":
    main()
