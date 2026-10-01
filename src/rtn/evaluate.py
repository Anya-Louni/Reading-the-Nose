"""Phase 2/3 evaluation: baselines and graph models under both split types.

## Models

| key              | what it isolates |
|------------------|------------------|
| `mean`           | sanity floor: predict the mean training signature |
| `ecfp_mlp`       | flat molecular descriptor, no graph structure |
| `gcn`            | graph structure without attention |
| `gat`            | attention, trained from scratch |
| `gat_pretrained` | attention plus Stage 1 OpenPOM transfer |

`gcn` and `gat` share the encoder implementation, pooling, loss and training
loop, differing only in the convolution, so the comparison isolates attention
rather than confounding it with everything else.

## Metrics

- `cosine`: cosine similarity between predicted and true signature. Primary.
- `rank`: rank of the true molecule among all candidate signatures, ordered by
  cosine to the prediction. 1 is perfect. This is the metric that actually
  matters for the project's question, because a prediction can have decent
  cosine to the truth while still being closer to a *different* molecule, which
  is precisely the cross-reactivity failure we are trying to characterise.
- `top1`: fraction of folds with rank 1.

Both metrics are computed against signatures built from *training* rows only,
so the held-out molecule's own measurements never enter the candidate set
construction beyond the target being scored.

## A caution about fold counts

Leave-molecule-out over 8 molecules gives 8 numbers. The spread across folds is
large and the mean is not a stable estimate. Report the per-fold values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from rtn.assemble import PlatformData
from rtn.model import NoseNet, signature_loss
from rtn.molgraph import ecfp, featurize
from rtn.targets import SignatureScaler


@dataclass
class FoldResult:
    model: str
    split: str
    fold: str
    platform: str
    cosine: float
    rank: int
    n_candidates: int
    seed: int


@dataclass
class EvalConfig:
    epochs: int = 300
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    pretrained_dir: Path | None = None
    center_molecules: bool = False
    """Subtract the mean of the *training* molecule centroids before scoring.

    On UCI 251 (open wind tunnel) the molecule signatures share a large
    common-mode component: mean off-diagonal cosine is +0.583 and the leading
    principal axis of the centroid cloud holds 51.5% of the variance and
    correlates with log concentration at r = +0.58. Every gas was run at a
    single concentration there, so concentration is confounded with identity.

    That common mode is shared by all molecules, so predicting it says nothing
    about chemistry, but it inflates the cosine of every model including the
    constant baseline. Centring removes it and leaves the molecule-specific
    residual, which is the cross-reactivity signature the project is about:
    mean off-diagonal falls to -0.058, matching UCI 270's -0.094.

    The offset is computed from training molecules only, so it is leak-free.
    Report both settings; the gap between them is how much of the headline
    number is common mode rather than structure.
    """


# --------------------------------------------------------------------------
# shared fold plumbing
# --------------------------------------------------------------------------

def _fit_scalers(platforms: dict[str, PlatformData],
                 train_mask: dict[str, np.ndarray]) -> dict[str, SignatureScaler]:
    """Fit one scaler per platform on that platform's training rows only."""
    out = {}
    for key, p in platforms.items():
        rows = p.raw.loc[train_mask[key]]
        if len(rows) == 0:
            continue
        out[key] = SignatureScaler.fit(rows, list(p.raw.columns))
    return out


def _candidate_signatures(p: PlatformData, scaler: SignatureScaler,
                          mask: np.ndarray) -> dict[str, np.ndarray]:
    """Mean signature per molecule over the given rows, renormalised."""
    sig = p.signatures(scaler)
    out = {}
    for m in sorted(set(p.molecules[mask])):
        sel = mask & (p.molecules == m)
        v = sig[sel].mean(0)
        n = np.linalg.norm(v)
        out[m] = v / (n if n else 1.0)
    return out


def _score(pred: np.ndarray, truth: np.ndarray,
           candidates: dict[str, np.ndarray], true_name: str) -> tuple[float, int, int]:
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    c = cos(pred, truth)
    pool = dict(candidates)
    pool[true_name] = truth
    sims = sorted(((cos(pred, v), k) for k, v in pool.items()), reverse=True)
    rank = next(i for i, (_, k) in enumerate(sims, 1) if k == true_name)
    return c, rank, len(pool)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def _predict_1nn(train_names: list[str], train_sig: np.ndarray,
                 target: str) -> np.ndarray:
    """Predict the signature of the structurally most similar training molecule.

    Tanimoto on ECFP4, no learning at all. This is the sharpest test of the
    project's premise: if structurally similar molecules produce similar array
    responses, copying the nearest neighbour's signature should beat a constant.
    If it does not, the structure-to-response relationship is not there to learn
    and no amount of model capacity will find it.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from rtn.analytes import BY_NAME

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda n: gen.GetFingerprint(Chem.MolFromSmiles(BY_NAME[n].smiles))
    tgt = fp(target)
    sims = [DataStructs.TanimotoSimilarity(tgt, fp(n)) for n in train_names]
    return train_sig[int(np.argmax(sims))]


def _predict_mean(train_sig: np.ndarray) -> np.ndarray:
    v = train_sig.mean(0)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _train_torch(
    model: torch.nn.Module,
    batches: list[tuple],
    cfg: EvalConfig,
) -> torch.nn.Module:
    """Generic loop. `batches` holds (graph_batch, task, target) triples."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    model.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = sum(signature_loss(model(b, task=t), y) for b, t, y in batches)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    return model


class EcfpMlp(torch.nn.Module):
    """Flat fingerprint baseline. No graph structure at all."""

    def __init__(self, dims: dict[str, int], n_bits: int = 2048, hidden: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_bits, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleDict(
            {k: torch.nn.Linear(hidden, d) for k, d in dims.items()})

    def forward(self, x, task: str = "platform_a"):
        return self.heads[task](self.trunk(x))


# --------------------------------------------------------------------------
# fold runner
# --------------------------------------------------------------------------

TASK_FOR = {"A": "platform_a", "A2": "platform_a2", "B": "platform_b"}


def _build_net(platforms, attention, cfg):
    """NoseNet sized for whatever platforms are present in this run."""
    extra = {TASK_FOR[k]: p.dim for k, p in platforms.items()
             if TASK_FOR[k] not in ("platform_a", "platform_b")}
    return NoseNet(
        n_platform_a=platforms["A"].dim if "A" in platforms else 128,
        n_platform_b=platforms["B"].dim if "B" in platforms else None,
        head_dims=extra or None,
        attention=attention,
        hidden=cfg.hidden, layers=cfg.layers, heads=cfg.heads,
        dropout=cfg.dropout,
    )


def _graph_batch(names: list[str], device: str):
    from rtn.analytes import BY_NAME
    graphs = [featurize(BY_NAME[n].smiles, name=n).data for n in names]
    return next(iter(DataLoader(graphs, batch_size=len(graphs)))).to(device)


def _ecfp_matrix(names: list[str], device: str) -> torch.Tensor:
    from rtn.analytes import BY_NAME
    return torch.tensor(
        np.stack([ecfp(BY_NAME[n].smiles) for n in names]), device=device)


def run_fold(
    platforms: dict[str, PlatformData],
    train_mask: dict[str, np.ndarray],
    test_mask: dict[str, np.ndarray],
    split_name: str,
    fold_name: str,
    cfg: EvalConfig,
    models: tuple[str, ...],
) -> list[FoldResult]:
    """Train every model on this fold's training rows and score the held-out rows."""
    scalers = _fit_scalers(platforms, train_mask)
    results: list[FoldResult] = []

    # per-platform training targets, one row per molecule
    train_targets: dict[str, tuple[list[str], np.ndarray]] = {}
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for key, p in platforms.items():
        if key not in scalers or train_mask[key].sum() == 0:
            continue
        cand = _candidate_signatures(p, scalers[key], train_mask[key])
        candidates[key] = cand
        names = sorted(cand)
        train_targets[key] = (names, np.stack([cand[n] for n in names]))

    # optional common-mode removal, using training molecules only
    offsets: dict[str, np.ndarray] = {}
    if cfg.center_molecules:
        for key, (names, tgt) in train_targets.items():
            off = tgt.mean(0)
            offsets[key] = off
            centred = tgt - off
            centred /= np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
            train_targets[key] = (names, centred)
            candidates[key] = dict(zip(names, centred))

    # held-out truth, one row per (platform, molecule)
    truths: list[tuple[str, str, np.ndarray]] = []
    for key, p in platforms.items():
        if key not in scalers or test_mask[key].sum() == 0:
            continue
        sig = p.signatures(scalers[key])
        for m in sorted(set(p.molecules[test_mask[key]])):
            sel = test_mask[key] & (p.molecules == m)
            v = sig[sel].mean(0)
            if key in offsets:
                v = v - offsets[key]
            n = np.linalg.norm(v)
            truths.append((key, m, v / (n if n else 1.0)))
    if not truths:
        return results

    for seed in cfg.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        preds: dict[str, dict[tuple[str, str], np.ndarray]] = {m: {} for m in models}

        # --- mean baseline -------------------------------------------------
        if "mean" in models:
            for key, (_, tgt) in train_targets.items():
                mu = _predict_mean(tgt)
                for _, m, _ in truths:
                    preds["mean"][(key, m)] = mu

        # --- ECFP 1-NN -----------------------------------------------------
        if "ecfp_1nn" in models:
            for key, (names, tgt) in train_targets.items():
                for _, m, _ in truths:
                    preds["ecfp_1nn"][(key, m)] = _predict_1nn(names, tgt, m)

        # --- ECFP + MLP ----------------------------------------------------
        if "ecfp_mlp" in models:
            dims = {TASK_FOR[k]: p.dim for k, p in platforms.items()}
            net = EcfpMlp(dims, dropout=cfg.dropout).to(cfg.device)
            batches = []
            for key, (names, tgt) in train_targets.items():
                batches.append((_ecfp_matrix(names, cfg.device), TASK_FOR[key],
                                torch.tensor(tgt, dtype=torch.float, device=cfg.device)))
            _train_torch(net, batches, cfg)
            net.eval()
            with torch.no_grad():
                for key, m, _ in truths:
                    x = _ecfp_matrix([m], cfg.device)
                    preds["ecfp_mlp"][(key, m)] = (
                        net(x, task=TASK_FOR[key]).cpu().numpy()[0])

        # --- graph models --------------------------------------------------
        for key_model in ("gcn", "gat", "gat_pretrained"):
            if key_model not in models:
                continue
            net = _build_net(platforms, key_model != "gcn", cfg).to(cfg.device)

            if key_model == "gat_pretrained":
                if cfg.pretrained_dir is None:
                    continue
                w = cfg.pretrained_dir / f"encoder_pretrained_seed{seed}.pt"
                if not w.exists():
                    continue
                net.encoder.load_state_dict(torch.load(w, map_location=cfg.device))

            batches = []
            for key, (names, tgt) in train_targets.items():
                batches.append((_graph_batch(names, cfg.device), TASK_FOR[key],
                                torch.tensor(tgt, dtype=torch.float, device=cfg.device)))
            _train_torch(net, batches, cfg)
            net.eval()
            with torch.no_grad():
                for key, m, _ in truths:
                    b = _graph_batch([m], cfg.device)
                    preds[key_model][(key, m)] = (
                        net(b, task=TASK_FOR[key]).cpu().numpy()[0])

        # --- score ---------------------------------------------------------
        for model_name in models:
            for key, m, truth in truths:
                p = preds[model_name].get((key, m))
                if p is None:
                    continue
                c, rank, n_cand = _score(p, truth, candidates.get(key, {}), m)
                results.append(FoldResult(model_name, split_name, fold_name,
                                          key, c, rank, n_cand, seed))
    return results
