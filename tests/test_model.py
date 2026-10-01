"""Regression tests for the cases this analyte set makes unavoidable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtn.analytes import ANALYTES  # noqa: E402
from rtn.model import NoseNet, signature_loss  # noqa: E402
from rtn.molgraph import BOND_DIM, featurize  # noqa: E402

SINGLE_NODE = ["C", "N"]  # methane, ammonia


def test_single_node_graphs_featurize():
    for s in SINGLE_NODE:
        mg = featurize(s)
        assert mg is not None
        assert mg.data.x.shape[0] == 1
        assert mg.data.edge_index.shape == (2, 0)
        assert mg.data.edge_attr.shape == (0, BOND_DIM)


@pytest.mark.parametrize("attention", [True, False])
def test_forward_on_single_node_graph_is_finite(attention):
    """GATv2Conv's default fill_value='mean' produces NaN here. Guard it."""
    model = NoseNet(attention=attention, hidden=32, layers=2, heads=2)
    for s in SINGLE_NODE:
        batch = next(iter(DataLoader([featurize(s).data], batch_size=1)))
        out = model(batch, task="platform_a")
        assert torch.isfinite(out).all(), f"non-finite output for {s!r}"


@pytest.mark.parametrize("attention", [True, False])
def test_forward_on_all_analytes_is_finite(attention):
    model = NoseNet(attention=attention, hidden=32, layers=2, heads=2)
    graphs = [featurize(a.smiles).data for a in ANALYTES]
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    out = model(batch, task="platform_a")
    assert out.shape == (len(graphs), 128)
    assert torch.isfinite(out).all()


def test_mixed_batch_with_and_without_edges():
    """A batch mixing methane with toluene is the realistic case and the one
    most likely to break edge_attr handling."""
    model = NoseNet(hidden=32, layers=2, heads=2)
    graphs = [featurize(s).data for s in ["C", "Cc1ccccc1", "N", "OCCc1ccccc1"]]
    batch = next(iter(DataLoader(graphs, batch_size=4)))
    out = model(batch, task="odor")
    assert out.shape == (4, 138)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_finite_gradients():
    model = NoseNet(hidden=32, layers=2, heads=2)
    graphs = [featurize(s).data for s in ["C", "N", "Cc1ccccc1"]]
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    pred = model(batch, task="platform_a")
    target = torch.nn.functional.normalize(torch.randn(3, 128), dim=-1)
    signature_loss(pred, target).backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_attention_weights_returned_for_every_layer():
    model = NoseNet(hidden=32, layers=3, heads=2)
    batch = next(iter(DataLoader([featurize("Cc1ccccc1").data], batch_size=1)))
    _, node_h, att = model(batch, task="platform_a", return_attention=True)
    assert len(att) == 3
    for edge_index, weights in att:
        assert edge_index.shape[0] == 2
        assert weights.shape[0] == edge_index.shape[1]
    assert node_h.shape[0] == 7  # toluene heavy atoms


def test_attention_on_single_node_graph_does_not_crash():
    """Phase 4 will ask for attention on every analyte, methane included."""
    model = NoseNet(hidden=32, layers=2, heads=2)
    batch = next(iter(DataLoader([featurize("C").data], batch_size=1)))
    out, node_h, att = model(batch, task="platform_a", return_attention=True)
    assert torch.isfinite(out).all()
    # only the added self-loop should be present
    for edge_index, weights in att:
        assert edge_index.shape[1] == 1
        assert torch.isfinite(weights).all()


def test_signature_loss_is_zero_for_identical_direction():
    v = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
    assert signature_loss(v, v).abs().item() < 1e-5
    assert signature_loss(v * 3.0, v).abs().item() < 1e-5  # scale invariant


def test_attention_by_destination_is_the_softmax_constant():
    """Regression guard for the retracted 'attention is uniform' finding.

    GATv2 softmaxes over incoming edges, so summing alpha by destination gives
    exactly 1.0 per node in any model. Aggregating that way produced a uniform
    map that was mistakenly reported as a property of the trained model. This
    pins both behaviours so the distinction cannot quietly regress.
    """
    import numpy as np

    from rtn.explain import attention_scores

    model = NoseNet(n_odor=138, attention=True, hidden=32, layers=2, heads=2,
                    dropout=0.0)
    smi = "OCCc1ccccc1"

    by_dst = attention_scores(model, smi, "odor", reduce="destination")
    assert np.ptp(by_dst.atom_scores) < 1e-6, (
        "destination aggregation must be flat; if this fails the softmax "
        "assumption behind the retraction has changed")

    by_src = attention_scores(model, smi, "odor", reduce="source")
    assert np.ptp(by_src.atom_scores) > 1e-4, (
        "source aggregation must vary; a flat map here would mean the "
        "attention really is degenerate")


def test_raw_edge_attention_varies():
    """The finding that matters: per-edge alpha is not constant."""
    import numpy as np

    from rtn.explain import attention_edge_scores

    model = NoseNet(n_odor=138, attention=True, hidden=32, layers=2, heads=2,
                    dropout=0.0)
    ei, alpha = attention_edge_scores(model, "OCCc1ccccc1", "odor")
    assert ei.shape[0] == 2 and alpha.shape[0] == ei.shape[1]
    assert np.ptp(alpha) > 0.01, f"per-edge attention spread only {np.ptp(alpha):.2e}"
