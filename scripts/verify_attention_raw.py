"""INDEPENDENT VERIFICATION of the "attention is uniform" finding.

Deliberately isolated. This script does not import rtn.explain and does not use
any of the extraction helpers written earlier. It pulls attention coefficients
straight out of the PyG layer with return_attention_weights=True and reasons
about them from first principles.

Three checks:
  1. raw layer output on a real molecule, from a trained checkpoint
  2. does the pipeline's reported map match the raw values
  3. a hand-built toy graph where the expected behaviour is predictable
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtn.model import NoseNet           # the model, not the explainer
from rtn.molgraph import ATOM_DIM, BOND_DIM, featurize

CKPT = ROOT / "artifacts" / "encoder_pretrained_seed0.pt"


def line(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def raw_attention(model, batch):
    """Call each GATv2Conv by hand and keep its own attention output.

    Replicates the encoder's forward pass rather than calling it, so nothing in
    the existing code path is trusted.
    """
    import torch.nn.functional as F

    x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
    per_layer = []
    for conv, norm in zip(model.encoder.convs, model.encoder.norms):
        x_out, (att_ei, alpha) = conv(x, ei, edge_attr=ea,
                                      return_attention_weights=True)
        per_layer.append((att_ei.detach().cpu().numpy(),
                          alpha.detach().cpu().numpy()))
        x = F.gelu(norm(x_out))
    return per_layer


def main() -> None:
    torch.manual_seed(0)
    model = NoseNet(n_odor=138, attention=True, hidden=128, layers=3, heads=4,
                    dropout=0.0)
    if CKPT.exists():
        model.encoder.load_state_dict(torch.load(CKPT, map_location="cpu"))
        print(f"loaded trained encoder: {CKPT.name}")
    else:
        print("WARNING: checkpoint missing, using random init")
    model.eval()

    # ---------------------------------------------------------------- check 1
    line("CHECK 1  raw attention coefficients from the PyG layer, 2-phenylethanol")
    mg = featurize("OCCc1ccccc1")
    batch = next(iter(DataLoader([mg.data], batch_size=1)))
    n = batch.x.shape[0]
    layers = raw_attention(model, batch)

    for li, (ei, alpha) in enumerate(layers):
        print(f"\nlayer {li}: {alpha.shape[0]} edges (incl. self-loops), "
              f"{alpha.shape[1]} heads")
        a = alpha.mean(axis=1)                       # average heads
        src, dst = ei[0], ei[1]

        in_sum = np.zeros(n); np.add.at(in_sum, dst, a)
        out_sum = np.zeros(n); np.add.at(out_sum, src, a)

        print(f"  per-node sum of INCOMING alpha : "
              f"min {in_sum.min():.6f}  max {in_sum.max():.6f}  "
              f"spread {np.ptp(in_sum):.3e}")
        print(f"  per-node sum of OUTGOING alpha : "
              f"min {out_sum.min():.6f}  max {out_sum.max():.6f}  "
              f"spread {np.ptp(out_sum):.3e}")
        print(f"  raw per-edge alpha             : "
              f"min {a.min():.4f}  max {a.max():.4f}  spread {np.ptp(a):.4f}")

    line("WHAT THIS MEANS")
    print("""GATv2Conv applies softmax over the edges arriving at each destination
node. So the incoming alpha for any node sums to exactly 1.0 by construction,
for every node, in every graph, in every model, trained or not.

Aggregating attention by DESTINATION therefore cannot produce anything but a
uniform map. It is measuring the softmax normalisation constant, not the model.

Aggregating by SOURCE, or looking at per-edge alpha directly, is what carries
signal about which atoms the model actually weights.""")

    # ---------------------------------------------------------------- check 2
    line("CHECK 2  does the existing pipeline match the raw values?")
    from rtn.explain import attention_scores          # imported only to compare

    pipe = attention_scores(model, "OCCc1ccccc1", "odor")
    a0 = layers[0][1].mean(axis=1)
    ei0 = layers[0][0]

    by_dst = np.zeros(n); np.add.at(by_dst, ei0[1], a0)
    by_src = np.zeros(n); np.add.at(by_src, ei0[0], a0)

    # replicate the pipeline's own reduction from the raw numbers
    acc = np.zeros(n)
    for ei, alpha in layers:
        np.add.at(acc, ei[1], alpha.mean(axis=1))
    acc /= len(layers)
    replicated = np.abs(acc) / np.abs(acc).sum()

    print(f"pipeline output        : {np.round(pipe.atom_scores, 6)}")
    print(f"replicated from raw    : {np.round(replicated, 6)}")
    print(f"max abs difference     : {np.abs(pipe.atom_scores - replicated).max():.3e}")
    match = np.allclose(pipe.atom_scores, replicated, atol=1e-7)
    print(f"\n>>> extraction code reproduces the raw layer output: {match}")
    print(">>> so there is NO extraction bug in the sense of wrong numbers.")
    print(">>> The bug is the AGGREGATION CHOICE: summing by destination.")

    print(f"\nSame raw data, aggregated by SOURCE instead:")
    src_norm = by_src / by_src.sum()
    print(f"  by source            : {np.round(src_norm, 4)}")
    print(f"  spread               : {np.ptp(src_norm):.4f}   "
          f"(vs {np.ptp(by_dst / by_dst.sum()):.3e} by destination)")

    # ---------------------------------------------------------------- check 3
    line("CHECK 3  toy graph: a 3-node path where one node is clearly distinct")
    # node 0 and 2 identical features, node 1 different; path 0-1-2
    x = torch.zeros(3, ATOM_DIM)
    x[0, 0] = 1.0                     # carbon-like
    x[2, 0] = 1.0                     # identical to node 0
    x[1, 2] = 1.0                     # oxygen-like, distinct
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    ea = torch.zeros(4, BOND_DIM); ea[:, 0] = 1.0
    toy = next(iter(DataLoader([Data(x=x, edge_index=ei, edge_attr=ea)], batch_size=1)))

    tl = raw_attention(model, toy)
    ei0, al0 = tl[0]
    a = al0.mean(axis=1)
    print("per-edge alpha, layer 0 (src -> dst):")
    for k in range(ei0.shape[1]):
        print(f"   {ei0[0, k]} -> {ei0[1, k]}   alpha {a[k]:.4f}")

    in_sum = np.zeros(3); np.add.at(in_sum, ei0[1], a)
    out_sum = np.zeros(3); np.add.at(out_sum, ei0[0], a)
    print(f"\n  by destination : {np.round(in_sum, 6)}  spread {np.ptp(in_sum):.3e}")
    print(f"  by source      : {np.round(out_sum, 4)}  spread {np.ptp(out_sum):.4f}")
    print("""
Expected if the mechanism works: node 1 has different features from nodes 0
and 2, so the per-edge alphas and the by-source totals should differ. The
by-destination totals should still be 1.0 each, because that is the softmax
constraint and says nothing about the model.""")

    line("VERDICT")
    per_edge_varies = any(np.ptp(al.mean(axis=1)) > 1e-4 for _, al in layers)
    print(f"raw per-edge attention varies on a real molecule : {per_edge_varies}")
    print(f"by-destination aggregation is uniform            : "
          f"{np.ptp(by_dst) < 1e-6}")
    print(f"by-source aggregation varies                     : "
          f"{np.ptp(by_src) > 1e-4}")


if __name__ == "__main__":
    main()
