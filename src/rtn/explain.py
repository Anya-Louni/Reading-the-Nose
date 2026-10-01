"""Phase 4: interpolation-only explainability.

Scope, set deliberately and narrowly:

- **Interpolation only.** The model is trained on all 13 analytes and explained
  on those same molecules. Phase 2 showed no generalisation to unseen molecules,
  so any attention map presented as explaining an *unseen* prediction would be
  explaining noise. What is asked here is only "what does the model look at for
  a molecule it was fitted on".
- **Molecules with >= 3 heavy atoms** (`analytes.INTERPRETABLE`). Methane and
  ammonia are single nodes with no substructure to attend to.
- Every claim is checked before it is made: attention is cross-checked against
  integrated gradients, run through the Adebayo model-randomisation sanity
  check, and scored for faithfulness by deletion against a random-masking
  control. A map that fails those is reported as failing, not quietly dropped.

Attribution methods:

- `attention_scores`   mean incoming attention per atom, averaged over heads and
                       layers. The quantity the project pitch is about.
- `integrated_gradients` path integral of the gradient of the prediction with
                       respect to atom features, from an all-zero baseline.
                       Independent of the attention mechanism, so agreement
                       between the two is evidence and disagreement is a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from rtn.analytes import BY_NAME
from rtn.molgraph import featurize


@dataclass
class Attribution:
    molecule: str
    method: str
    atom_scores: np.ndarray      # (n_atoms,), normalised to sum 1
    atom_symbols: list[str]
    seed: int


def _batch(molecule: str, device: str = "cpu"):
    """Accepts either an analyte name from the registry or a raw SMILES string.

    The sensor task only ever explains 13 named molecules; the odour task
    explains arbitrary corpus members. Resolving the registry first keeps every
    existing call site working unchanged.
    """
    smiles = BY_NAME[molecule].smiles if molecule in BY_NAME else molecule
    mg = featurize(smiles, name=molecule)
    if mg is None:
        raise ValueError(f"could not featurize {molecule!r}")
    return next(iter(DataLoader([mg.data], batch_size=1))).to(device), mg.mol


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.abs(v)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / max(len(v), 1))


def attention_scores(model, molecule: str, task: str,
                     device: str = "cpu", seed: int = 0,
                     reduce: str = "source") -> Attribution:
    """Per-atom attention, averaged over heads and layers.

    ## Aggregate by SOURCE, never by destination

    `GATv2Conv` applies softmax over the edges arriving at each destination
    node. The incoming alpha for a node therefore sums to exactly 1.0, for every
    node, in every graph, in every model, trained or randomly initialised. It is
    a normalisation constant.

    An earlier version of this function summed by destination. It produced a
    perfectly uniform map on every molecule, which was reported as the finding
    "the learned attention is uniform to float32 precision". That was wrong: it
    measured the softmax constraint, not the model. The giveaway was a quantity
    constant to seven decimals across every molecule and seed, which is the
    signature of an identity rather than a measurement.

    `reduce="source"` sums the outgoing alpha per node: how much the rest of the
    graph attends to this atom. That is the quantity the interpretability claim
    is about, and it varies (spread ~0.35 per layer on 2-phenylethanol).

    `reduce="destination"` is retained only so the broken behaviour can be
    reproduced when checking the retraction, and is never the right choice for
    an interpretation.

    Verified independently against raw PyG layer output in
    `scripts/verify_attention_raw.py`.
    """
    if reduce not in ("source", "destination"):
        raise ValueError("reduce must be 'source' or 'destination'")

    batch, mol = _batch(molecule, device)
    model.eval()
    with torch.no_grad():
        _, _, attns = model(batch, task=task, return_attention=True)

    n = batch.x.shape[0]
    acc = np.zeros(n)
    row = 0 if reduce == "source" else 1
    for edge_index, alpha in attns:
        a = alpha.mean(dim=-1).detach().cpu().numpy()   # average heads
        idx = edge_index[row].detach().cpu().numpy()
        np.add.at(acc, idx, a)
    acc /= max(len(attns), 1)

    return Attribution(molecule, "attention", _normalise(acc),
                       [a.GetSymbol() for a in mol.GetAtoms()], seed)


def attention_edge_scores(model, molecule: str, task: str,
                          device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Raw per-edge attention, averaged over heads, concatenated across layers.

    Returned as (edge_index, alpha) so edge-level claims can be made without
    going through any node aggregation at all.
    """
    batch, _ = _batch(molecule, device)
    model.eval()
    with torch.no_grad():
        _, _, attns = model(batch, task=task, return_attention=True)
    ei = np.concatenate([e.detach().cpu().numpy() for e, _ in attns], axis=1)
    al = np.concatenate([a.mean(dim=-1).detach().cpu().numpy() for _, a in attns])
    return ei, al


def integrated_gradients(model, molecule: str, task: str, steps: int = 64,
                         device: str = "cpu", seed: int = 0) -> Attribution:
    """IG on atom features against an all-zero baseline, summed per atom.

    The target is a direction, not a scalar, so the scalar being attributed is
    the cosine between the prediction at the interpolated input and the model's
    own prediction on the full molecule. That keeps the attribution about the
    predicted signature rather than an arbitrary output coordinate.
    """
    batch, mol = _batch(molecule, device)
    model.eval()
    with torch.no_grad():
        reference = model(batch, task=task).detach()
    reference = reference / reference.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    x_full = batch.x.clone()
    total = torch.zeros_like(x_full)
    for k in range(1, steps + 1):
        scaled = x_full * (k / steps)
        batch.x = scaled.clone().requires_grad_(True)
        out = model(batch, task=task)
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        (out * reference).sum().backward()
        total = total + batch.x.grad.detach()
    batch.x = x_full

    ig = (x_full * total / steps).sum(dim=-1).cpu().numpy()
    return Attribution(molecule, "integrated_gradients", _normalise(ig),
                       [a.GetSymbol() for a in mol.GetAtoms()], seed)


# ---------------------------------------------------------------------------
# sanity check (Adebayo et al., "Sanity Checks for Saliency Maps")
# ---------------------------------------------------------------------------

def randomize_weights(model, seed: int = 0):
    """Return a copy of the model with freshly reinitialised parameters.

    Cascading randomisation of the whole network. If an attribution map is
    essentially unchanged under this, it is a function of the input and the
    architecture rather than of anything the model learned, and it does not
    explain the prediction.

    Implementation note, learned the hard way: do NOT reinitialise by looping
    over raw parameters and zeroing everything 1-dimensional. LayerNorm's
    `weight` is 1-D, and zeroing it makes every LayerNorm emit exactly zero, so
    the network becomes dead rather than random. Integrated gradients then
    returns all-zero attributions, the normaliser falls back to uniform, and
    every rank correlation comes out NaN. A dead model is not a control.

    Instead ask each module to reinitialise itself, which respects the intended
    distribution for every parameter type.
    """
    import copy

    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    n_reset = 0
    for module in rnd.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            n_reset += 1
    if n_reset == 0:
        raise RuntimeError("no module exposed reset_parameters; randomisation is a no-op")

    # Guard against the failure above recurring silently.
    for name, p in rnd.named_parameters():
        if "norm" in name.lower() and "weight" in name and float(p.detach().abs().sum()) == 0.0:
            raise RuntimeError(f"{name} reinitialised to all zeros; model would be dead")
    return rnd


def sanity_check(model, molecule: str, task: str, method, seed: int = 0) -> dict:
    """Compare an attribution on the trained model against a randomised one.

    Reports Spearman rank correlation. Near 1 means the explanation survives
    destroying the weights, i.e. it is not explaining the model.
    """
    from scipy.stats import spearmanr

    trained = method(model, molecule, task, seed=seed)
    rand = method(randomize_weights(model, seed), molecule, task, seed=seed)
    n = len(trained.atom_scores)

    # Benzene is fully symmetric: every atom is equivalent, so any per-atom
    # attribution must be constant and a rank correlation is undefined. That is
    # a property of the molecule, not a failure of the method, and it is why
    # benzene carries no atom-level story in the writeup.
    if n < 3 or np.ptp(trained.atom_scores) == 0 or np.ptp(rand.atom_scores) == 0:
        return {"molecule": molecule, "method": trained.method,
                "spearman": float("nan"), "n_atoms": n,
                "verdict": "undefined (attribution is constant; molecule is symmetric)"}
    rho = spearmanr(trained.atom_scores, rand.atom_scores).statistic
    return {"molecule": molecule, "method": trained.method,
            "spearman": float(rho), "n_atoms": len(trained.atom_scores),
            "verdict": "FAILS (map survives weight randomisation)" if rho > 0.8
                       else "passes"}


# ---------------------------------------------------------------------------
# faithfulness: deletion curve
# ---------------------------------------------------------------------------

def deletion_curve(model, molecule: str, task: str, order: np.ndarray,
                   device: str = "cpu") -> np.ndarray:
    """Mask atoms one at a time in the given order; track prediction similarity.

    Masking is zeroing the atom's feature vector. Atoms cannot literally be
    deleted from a molecular graph without changing the connectivity into
    something that is not a molecule, so this is the standard substitute and is
    applied identically to the attribution order and the random control.

    Returns cosine of the masked prediction to the unmasked one, at each step.
    """
    batch, _ = _batch(molecule, device)
    model.eval()
    with torch.no_grad():
        full = model(batch, task=task)
        full = full / full.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        x0 = batch.x.clone()
        out = [1.0]
        for k in range(1, len(order) + 1):
            batch.x = x0.clone()
            batch.x[order[:k]] = 0.0
            p = model(batch, task=task)
            p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            out.append(float((p * full).sum()))
        batch.x = x0
    return np.asarray(out)


def deletion_auc(model, molecule: str, task: str, scores: np.ndarray,
                 n_random: int = 50, seed: int = 0) -> dict:
    """Area under the deletion curve, attribution order vs random control.

    Lower AUC is better: masking the atoms the explanation calls important
    should destroy the prediction faster than masking random atoms. The number
    that matters is the gap, not the absolute AUC, because a small molecule
    reaches full masking in a handful of steps regardless of order.
    """
    n = len(scores)
    if n < 3:
        return {"molecule": molecule, "n_atoms": n, "auc_attr": float("nan"),
                "auc_random": float("nan"), "gap": float("nan")}

    attr_order = np.argsort(-scores)
    auc_attr = deletion_curve(model, molecule, task, attr_order).mean()

    rng = np.random.default_rng(seed)
    aucs = [deletion_curve(model, molecule, task,
                           rng.permutation(n)).mean() for _ in range(n_random)]
    auc_rand = float(np.mean(aucs))

    return {"molecule": molecule, "n_atoms": n,
            "auc_attr": float(auc_attr), "auc_random": auc_rand,
            "auc_random_std": float(np.std(aucs)),
            "gap": auc_rand - float(auc_attr),
            "faithful": bool(auc_attr < auc_rand - np.std(aucs))}
