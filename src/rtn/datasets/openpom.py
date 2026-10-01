"""Stage 1 pretraining corpus: OpenPOM curated GS-LF.

4,983 molecules from GoodScents + Leffingwell, 138 binary odor descriptors,
mean 4.9 labels per molecule. Source:
https://github.com/ARY2260/openpom (curated_GS_LF_merged_4983.csv)

The thirteen sensor analytes are removed here, not at train time, so there is no
way to accidentally pretrain on a molecule that a later fold holds out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from rtn.analytes import pretrain_exclusions
from rtn.molgraph import canonical_smiles, featurize

CSV_URL = (
    "https://raw.githubusercontent.com/ARY2260/openpom/main/"
    "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
)
META_COLS = ["nonStereoSMILES", "descriptors"]


def download(dest: Path) -> Path:
    from urllib.request import urlopen

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urlopen(CSV_URL) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def load(csv_path: Path, exclude_analytes: bool = True) -> tuple[list[Data], list[str], pd.DataFrame]:
    """Return (graphs, descriptor_names, dropped_rows).

    `dropped_rows` records what was removed and why, so the writeup can state the
    corpus size honestly rather than quoting the headline 4,983.
    """
    df = pd.read_csv(csv_path)
    descriptors = [c for c in df.columns if c not in META_COLS]

    excluded = pretrain_exclusions() if exclude_analytes else set()
    graphs, dropped = [], []

    # Descriptor names contain spaces ("black currant"), so index positionally
    # rather than by attribute.
    labels = df[descriptors].to_numpy(dtype="float32")

    for i, (smiles, text) in enumerate(zip(df["nonStereoSMILES"], df["descriptors"])):
        canon = canonical_smiles(smiles)
        if canon is None:
            dropped.append((smiles, "rdkit_parse_failed"))
            continue
        if canon in excluded:
            dropped.append((smiles, "sensor_analyte_excluded"))
            continue
        mg = featurize(smiles)
        if mg is None:
            dropped.append((smiles, "featurize_failed"))
            continue
        mg.data.y = torch.from_numpy(labels[i]).unsqueeze(0)
        mg.data.descriptors_text = text
        graphs.append(mg.data)

    return graphs, descriptors, pd.DataFrame(dropped, columns=["smiles", "reason"])
