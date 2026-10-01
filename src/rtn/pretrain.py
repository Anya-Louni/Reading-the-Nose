"""Stage 1: pretrain the graph encoder on OpenPOM odor descriptors.

4,975 molecules (the 13 sensor analytes are already excluded by the loader),
138 binary descriptors, multi-label. The point is not the odor model itself but
the encoder weights: the sensor task has 8-13 molecules and cannot teach a graph
encoder anything on its own.

Split is scaffold-based, which is meaningful on this corpus even though it is
degenerate on the analyte set.

Labels are heavily imbalanced (the rarest descriptor has 31 positives out of
4,975), so the loss is positive-weighted and the reported metric is macro AUC,
not accuracy. Accuracy here would sit above 0.96 by predicting all-negative.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from rtn.model import NoseNet
from rtn.splits import scaffold_groups


@dataclass
class PretrainResult:
    macro_auc: float
    epochs_run: int
    best_epoch: int
    history: list[dict] = field(default_factory=list)


def scaffold_split(graphs, val_frac: float = 0.15, seed: int = 0):
    """Group-disjoint scaffold split. Largest groups go to train first."""
    groups = scaffold_groups([g.smiles for g in graphs])
    rng = np.random.default_rng(seed)
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))

    n_val_target = int(len(graphs) * val_frac)
    val_idx: list[int] = []
    # skip the largest group (usually acyclic) so validation is not all one scaffold
    for k in rng.permutation(keys[1:]):
        if len(val_idx) >= n_val_target:
            break
        val_idx.extend(groups[k])
    val = set(val_idx)
    return ([g for i, g in enumerate(graphs) if i not in val],
            [g for i, g in enumerate(graphs) if i in val])


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean AUC over descriptors that have both classes present."""
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if 0 < col.sum() < len(col):
            aucs.append(roc_auc_score(col, y_score[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def pretrain(
    graphs,
    n_descriptors: int = 138,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 12,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
    **encoder_kwargs,
) -> tuple[NoseNet, PretrainResult]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, val = scaffold_split(graphs, seed=seed)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val, batch_size=256)

    model = NoseNet(n_odor=n_descriptors, **encoder_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    y_all = torch.cat([g.y for g in train])
    pos = y_all.sum(0).clamp(min=1.0)
    pos_weight = ((len(y_all) - pos) / pos).clamp(max=50.0).to(device)

    best, best_state, best_epoch, history = -np.inf, None, 0, []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in tl:
            batch = batch.to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(batch, task="odor"), batch.y, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * batch.num_graphs
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(device)
                ps.append(torch.sigmoid(model(batch, task="odor")).cpu().numpy())
                ys.append(batch.y.cpu().numpy())
        auc = macro_auc(np.concatenate(ys), np.concatenate(ps))
        history.append({"epoch": epoch, "loss": total / len(train), "val_macro_auc": auc})

        if auc > best:
            best, best_epoch = auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  loss {total/len(train):.4f}  "
                  f"val macro AUC {auc:.4f}  best {best:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, PretrainResult(best, len(history), best_epoch, history)
