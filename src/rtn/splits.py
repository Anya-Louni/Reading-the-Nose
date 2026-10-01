"""Evaluation splits.

Headline: leave-molecule-out. Train on N-1 molecules, test on the held-out one,
repeat for all N. This is the only split that tests whether molecular structure
generalises rather than whether the model memorised a per-molecule response.

Secondary: a random split over measurements, reported for comparison because it
is the number most papers quote. It must be batch-aware. Rodriguez-Lujan et al.
(arXiv 2108.08793) show dataset 270's batch structure confounds gas identity
with acquisition time; a naive random split puts measurements from the same
batch on both sides and inflates the number substantially. `random_split` here
splits by batch, not by measurement, and says so in its name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_molecule: str | None = None

    def __repr__(self) -> str:
        return (f"Fold({self.name}, train={len(self.train_idx)}, "
                f"test={len(self.test_idx)})")


def leave_molecule_out(molecules: np.ndarray) -> Iterator[Fold]:
    """One fold per unique molecule.

    With 13 molecules every fold trains on 12. That is a small training set and
    the writeup must not present per-fold numbers as if they were stable; report
    the distribution across folds, not just the mean.
    """
    for m in sorted(set(molecules)):
        test = np.flatnonzero(molecules == m)
        train = np.flatnonzero(molecules != m)
        yield Fold(f"lmo:{m}", train, test, held_out_molecule=m)


def leave_batch_out(batches: np.ndarray) -> Iterator[Fold]:
    """One fold per acquisition batch. Isolates drift rather than structure.

    Useful as a control: if leave-batch-out is easy and leave-molecule-out is
    hard, the model is fitting chemistry poorly but handling drift fine, which
    is a different failure than the reverse.
    """
    for b in sorted(set(batches)):
        test = np.flatnonzero(batches == b)
        train = np.flatnonzero(batches != b)
        yield Fold(f"lbo:batch{b}", train, test)


def batchwise_random_split(
    batches: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Fold:
    """Random split at the batch level, never at the measurement level."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(batches)))
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_batches = set(uniq[:n_test].tolist())
    mask = np.array([b in test_batches for b in batches])
    return Fold(f"random(seed={seed})", np.flatnonzero(~mask), np.flatnonzero(mask))


def scaffold_groups(smiles: list[str]) -> dict[str, list[int]]:
    """Bemis-Murcko scaffold grouping.

    Retained because the Phase 0 spec named it as the fallback split. On this
    analyte set it is close to useless: most of the molecules are acyclic and
    collapse to a single empty scaffold. Do not use it as the headline split
    here; it is kept for the OpenPOM stage-1 corpus, where it is meaningful.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        key = "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        groups.setdefault(key, []).append(i)
    return groups
