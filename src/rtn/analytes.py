"""The analyte registry.

Thirteen molecules with public MOX array measurements, across two sensor
platforms. See data_audit/PHASE0_DATA_AUDIT.md for how this set was arrived at
and why it is not larger.

Platform A (Figaro TGS26xx, UCSD): UCI 270, UCI 251, UCI 361.
Platform B (62-sensor commercial nose): Zenodo 15681119.

The two platforms share no sensors, so the model gets a shared graph encoder and
one output head per platform. Ethanol is the only molecule measured on both,
which makes it the single point of comparison between the heads. Worth reporting
as such rather than treating it as a bridge that identifies the two scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Analyte:
    name: str
    smiles: str
    heavy_atoms: int
    sources: tuple[str, ...]
    platform: tuple[str, ...]
    in_openpom: bool
    note: str = ""


ANALYTES: list[Analyte] = [
    Analyte("methane", "C", 1, ("uci251", "uci361"), ("A",), False,
            "single node, no bonds"),
    Analyte("ammonia", "N", 1, ("uci270", "uci251"), ("A",), False,
            "single node, no bonds"),
    Analyte("carbon_monoxide", "[C-]#[O+]", 2, ("uci251", "uci361"), ("A",), False,
            "charge-separated triple bond, atypical valence"),
    Analyte("methanol", "CO", 2, ("uci251",), ("A",), True),
    Analyte("ethylene", "C=C", 2, ("uci270", "uci251", "uci361"), ("A",), False),
    Analyte("ethanol", "CCO", 3, ("uci270", "uci361", "zenodo"), ("A", "B"), True,
            "only molecule present on both sensor platforms"),
    Analyte("acetaldehyde", "CC=O", 3, ("uci270", "uci251"), ("A",), True),
    Analyte("acetone", "CC(C)=O", 4, ("uci270", "uci251"), ("A",), True),
    Analyte("butan-1-ol", "CCCCO", 5, ("uci251",), ("A",), True),
    Analyte("benzene", "c1ccccc1", 6, ("uci251",), ("A",), False),
    Analyte("diacetyl", "CC(=O)C(C)=O", 6, ("zenodo",), ("B",), True),
    Analyte("toluene", "Cc1ccccc1", 7, ("uci270", "uci251"), ("A",), True),
    Analyte("2-phenylethanol", "OCCc1ccccc1", 9, ("zenodo",), ("B",), True),
]

BY_NAME = {a.name: a for a in ANALYTES}
BY_SMILES = {a.smiles: a for a in ANALYTES}

# Molecules in scope for Phase 4 attention claims.
#
# Cutoff is >= 3 heavy atoms: ethanol, acetaldehyde, acetone, butan-1-ol,
# benzene, diacetyl, toluene, 2-phenylethanol. Methane, ammonia, carbon
# monoxide, methanol and ethylene are excluded - a one- or two-node graph has no
# substructure to attend to and forcing a narrative there is exactly the
# overstatement the Phase 4 sanity checks exist to catch.
#
# Note the boundary is soft at the bottom. On ethanol (C-C-O) and acetaldehyde
# (C-C=O) every atom is arguably part of the functional group, so an attention
# map there says less than it does on toluene or 2-phenylethanol. Report the
# >= 5 heavy-atom subset separately if the small ones look degenerate.
INTERPRETABLE = tuple(a.name for a in ANALYTES if a.heavy_atoms >= 3)
INTERPRETABLE_STRICT = tuple(a.name for a in ANALYTES if a.heavy_atoms >= 5)

PLATFORM_A = tuple(a.name for a in ANALYTES if "A" in a.platform)
PLATFORM_B = tuple(a.name for a in ANALYTES if "B" in a.platform)


def pretrain_exclusions() -> set[str]:
    """Canonical SMILES to strip from the OpenPOM pretraining corpus.

    Eight of the thirteen analytes appear in OpenPOM. Leaving them in would leak
    the held-out molecule of every leave-molecule-out fold into stage 1. The
    alternative is retraining stage 1 once per fold, which costs 13x the compute
    for no scientific gain, so all thirteen come out up front and the pretraining
    corpus is fixed across folds.
    """
    from rtn.molgraph import canonical_smiles

    out = set()
    for a in ANALYTES:
        c = canonical_smiles(a.smiles)
        if c is not None:
            out.add(c)
    return out
