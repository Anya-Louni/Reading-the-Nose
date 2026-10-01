"""Phase 2 on the odour-descriptor task: same pipeline, 383x the molecules.

## Why this task exists

This is the *second* experiment in the project, not the first. The sensor
cross-reactivity task was run first as the more novel and harder test, and
failed cleanly: no model beat a constant prediction under leave-molecule-out,
and the GAT's attention came out uniform to floating-point precision. That
result stands and is not being replaced.

The open question that failure left is whether the method is wrong or whether 13
molecules with a median of 3 heavy atoms was simply too little data to learn
from. Running the identical pipeline on 4,975 molecules answers it. If the graph
models now separate cleanly from the baselines, the negative was about data. If
they still do not, it was about the method.

Odour prediction from molecular graphs is not itself novel; OpenPOM's own paper
does it. Nothing here should be presented as a new task.

## What is reused unchanged

The featurizer, the `NoseNet` encoder and heads, the scaffold grouping, and the
whole explainability and sanity-check harness. This module adds only the
multi-label evaluation loop and the baselines in the ordering the sensor task
used.

## The pretraining arm had to change

"GAT + OpenPOM pretraining" is circular here: those encoder weights were fitted
on this exact corpus, so using them would be training on the test set. The arm
is replaced by transfer in the opposite direction, an encoder pretrained on the
13-molecule sensor task, which asks whether the sensor signal carries anything
useful about odour. That is a real question and a fair slot in the same table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from rtn.model import NoseNet
from rtn.splits import scaffold_groups

MODELS = ("constant", "ecfp_mlp", "gcn", "gat", "gat_sensor_pretrained")


@dataclass
class OdorResult:
    model: str
    split: str
    fold: int
    seed: int
    macro_auroc: float
    macro_ap: float
    n_test: int
    n_descriptors_scored: int
    epochs_run: int = 0


def macro_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, int]:
    """Macro AUROC and macro average precision over descriptors present in the fold.

    A descriptor with no positives (or no negatives) in the test fold is skipped
    rather than scored as 0 or 0.5; with 138 descriptors and the rarest holding
    31 positives corpus-wide, some will always be absent from a given fold, and
    imputing a value for them would quietly move the mean.
    """
    aurocs, aps = [], []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if 0 < col.sum() < len(col):
            aurocs.append(roc_auc_score(col, y_score[:, j]))
            aps.append(average_precision_score(col, y_score[:, j]))
    if not aurocs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aurocs)), float(np.mean(aps)), len(aurocs)


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def scaffold_folds(graphs, k: int = 3, seed: int = 0) -> list[np.ndarray]:
    """Group-disjoint scaffold folds, greedily balanced by size.

    Bemis-Murcko scaffolds are meaningful on this corpus, unlike on the 13
    analytes where they collapsed to two groups. Largest groups are placed
    first into whichever fold is currently smallest, so no fold is dominated by
    the single acyclic group.
    """
    groups = scaffold_groups([g.smiles for g in graphs])
    order = sorted(groups, key=lambda key: (-len(groups[key]), key))
    folds: list[list[int]] = [[] for _ in range(k)]
    for key in order:
        target = min(range(k), key=lambda i: len(folds[i]))
        folds[target].extend(groups[key])
    rng = np.random.default_rng(seed)
    return [np.array(sorted(f)) for f in folds] if k > 1 else [np.arange(len(graphs))]


def random_folds(n: int, k: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(a) for a in np.array_split(idx, k)]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class EcfpMlpMulti(torch.nn.Module):
    """Flat fingerprint baseline, the control for whether graph structure helps."""

    def __init__(self, n_out: int, n_bits: int = 2048, hidden: int = 512,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_bits, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def ecfp_matrix(graphs) -> torch.Tensor:
    from rtn.molgraph import ecfp
    return torch.tensor(np.stack([ecfp(g.smiles) for g in graphs]))


def _pos_weight(y: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    pos = y.sum(0).clamp(min=1.0)
    return ((len(y) - pos) / pos).clamp(max=cap)


def train_eval_graph(
    train_g, val_g, test_g, n_out: int, attention: bool,
    init_encoder: Path | None = None,
    epochs: int = 60, patience: int = 8, batch_size: int = 128,
    lr: float = 1e-3, seed: int = 0, device: str = "cpu",
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NoseNet(n_odor=n_out, attention=attention,
                    hidden=128, layers=3, heads=4, dropout=0.1).to(device)
    if init_encoder is not None and init_encoder.exists():
        model.encoder.load_state_dict(torch.load(init_encoder, map_location=device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = _pos_weight(torch.cat([g.y for g in train_g])).to(device)

    tl = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_g, batch_size=256)
    best, best_state, best_ep, ran = -np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        ran = epoch
        model.train()
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                model(batch, task="odor"), batch.y, pos_weight=pw).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                ps.append(torch.sigmoid(model(batch, task="odor")).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
        auc, _, _ = macro_scores(np.concatenate(ys), np.concatenate(ps))
        if auc > best:
            best, best_ep = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_ep >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(test_g, batch_size=256):
            out.append(torch.sigmoid(model(batch.to(device), task="odor")).cpu().numpy())
    return np.concatenate(out), ran


def train_eval_ecfp(train_g, val_g, test_g, n_out: int, epochs: int = 120,
                    patience: int = 12, seed: int = 0,
                    device: str = "cpu") -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    Xtr, Xva, Xte = (ecfp_matrix(g).to(device) for g in (train_g, val_g, test_g))
    ytr = torch.cat([g.y for g in train_g]).to(device)
    yva = np.concatenate([g.y.numpy() for g in val_g])

    net = EcfpMlpMulti(n_out).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = _pos_weight(ytr).to(device)
    best, best_state, best_ep, ran = -np.inf, None, 0, 0

    for epoch in range(1, epochs + 1):
        ran = epoch
        net.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(net(Xtr), ytr, pos_weight=pw).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            auc, _, _ = macro_scores(yva, torch.sigmoid(net(Xva)).cpu().numpy())
        if auc > best:
            best, best_ep = auc, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch - best_ep >= patience:
            break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xte)).cpu().numpy(), ran


def constant_prediction(train_g, test_g, n_out: int) -> np.ndarray:
    """Predict each descriptor's training base rate for every molecule.

    Identical for all molecules, so AUROC is 0.5 by construction. It is the
    floor, and its macro average precision is the prevalence any real model has
    to beat on the imbalanced descriptors.
    """
    y = torch.cat([g.y for g in train_g]).numpy()
    rate = y.mean(0)
    return np.tile(rate, (len(test_g), 1))
