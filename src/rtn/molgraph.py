"""SMILES to molecular graph featurization.

Atoms are nodes, bonds are edges. Feature choices follow the Phase 1 spec:
atoms carry element, formal charge, hybridization, aromaticity and degree;
bonds carry bond type, conjugation and ring membership.

Design notes for this project specifically:

- The sensor analytes include methane (1 atom, 0 bonds) and ammonia (1 atom,
  0 bonds). Single-node graphs with an empty edge index are legal here and must
  stay legal all the way through the model. Anything that assumes at least one
  edge will break on those two. See `test_molgraph.py`.
- Hydrogens stay implicit. Making them explicit would triple the node count on
  the small analytes without adding information the sensor task can use, and it
  would make attention maps harder to read against standard 2D structures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# Elements covering both the sensor analytes and the OpenPOM corpus.
# "other" catches the long tail (P, Si, halogens beyond the listed ones, etc.).
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se"]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

FORMAL_CHARGES = [-2, -1, 0, 1, 2]
DEGREES = [0, 1, 2, 3, 4, 5]
NUM_HS = [0, 1, 2, 3, 4]


def _one_hot(value, choices: list, allow_other: bool = True) -> list[float]:
    """One-hot with an optional trailing 'other' slot."""
    vec = [0.0] * (len(choices) + (1 if allow_other else 0))
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        if allow_other:
            vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ELEMENTS)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalDegree(), DEGREES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass() * 0.01),
        ]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(ELEMENTS) + 1 + len(FORMAL_CHARGES) + 1 + len(HYBRIDIZATIONS) + 1 + len(DEGREES) + 1 + len(NUM_HS) + 1 + 3
BOND_DIM = len(BOND_TYPES) + 1 + 2


@dataclass(frozen=True)
class MolGraph:
    """Convenience wrapper so callers can keep the RDKit mol for rendering."""

    smiles: str
    data: Data
    mol: Chem.Mol


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def featurize(smiles: str, name: str | None = None) -> MolGraph | None:
    """Build a PyG Data object from SMILES. Returns None if RDKit cannot parse it.

    Edges are stored in both directions. Single-atom molecules get an edge_index
    of shape (2, 0) and an edge_attr of shape (0, BOND_DIM), which is what the
    downstream GAT layers expect for an isolated node.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feats, feats]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = Chem.MolToSmiles(mol)
    data.name = name if name is not None else data.smiles
    return MolGraph(smiles=data.smiles, data=data, mol=mol)


def ecfp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """ECFP4 bit vector for the Phase 2 fingerprint baseline."""
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.asarray(gen.GetFingerprint(mol), dtype=np.float32)
