"""Graph encoder and task heads.

One shared GATv2 encoder, three heads:
  - `odor`      : 138 binary odor descriptors (Stage 1 pretraining, OpenPOM)
  - `platform_a`: Figaro TGS26xx array signature (UCI 270 / 251 / 361)
  - `platform_b`: 62-sensor commercial nose signature (Zenodo 15681119)

The two sensor heads are separate because the platforms share no sensors, so
their response vectors live in unrelated spaces. Only ethanol is measured on
both, which is not enough to identify a mapping between them.

## The single-node problem

Methane and ammonia are one heavy atom with no bonds. `add_self_loops=True` is
what makes them work at all: without it those nodes receive no messages and the
encoder returns whatever the initial features pool to.

`fill_value=0.0` is set explicitly for the synthetic self-loop edge features.
PyG 2.8 does handle the default `fill_value="mean"` on a zero-edge graph without
producing NaN (checked), so this is a semantic choice rather than a workaround:
a self-loop is not a bond, and averaging real bond features into it would tell
the model that methane's self-loop looks like an average bond. Zero says "no
bond", which is true.

tests/test_model.py covers the zero-edge path end to end, including attention
extraction and the backward pass. Run it after touching the encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool, global_max_pool, global_mean_pool

from rtn.molgraph import ATOM_DIM, BOND_DIM


class GraphEncoder(nn.Module):
    """GATv2 encoder that can also run as a plain GCN for the Phase 2 ablation.

    `attention=False` gives the GCN baseline through the same code path, so the
    comparison isolates attention rather than confounding it with a different
    implementation, pooling scheme or training loop.
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        attention: bool = True,
        atom_dim: int = ATOM_DIM,
        bond_dim: int = BOND_DIM,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = atom_dim
        for _ in range(layers):
            if attention:
                conv = GATv2Conv(
                    in_dim,
                    hidden // heads,
                    heads=heads,
                    edge_dim=bond_dim,
                    add_self_loops=True,
                    fill_value=0.0,  # a self-loop is not a bond; see module docstring
                    dropout=dropout,
                )
            else:
                conv = GCNConv(in_dim, hidden, add_self_loops=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden))
            in_dim = hidden

        self.out_dim = hidden * 3  # mean + max + sum pooling

    def forward(self, data, return_attention: bool = False):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        attentions = []

        for conv, norm in zip(self.convs, self.norms):
            if self.attention:
                if return_attention:
                    x, (att_ei, att_w) = conv(x, ei, edge_attr=ea,
                                              return_attention_weights=True)
                    attentions.append((att_ei, att_w))
                else:
                    x = conv(x, ei, edge_attr=ea)
            else:
                x = conv(x, ei)
            x = F.gelu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Sum pooling is included alongside mean/max because the sensor response
        # scales with the amount of reducible material, which is a size-extensive
        # property that mean pooling deliberately discards.
        g = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch),
        ], dim=-1)

        if return_attention:
            return g, x, attentions
        return g


class Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class NoseNet(nn.Module):
    """Shared encoder plus the three task heads."""

    def __init__(
        self,
        n_odor: int = 138,
        n_platform_a: int = 128,
        n_platform_b: int | None = None,
        head_dims: dict[str, int] | None = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(**encoder_kwargs)
        d = self.encoder.out_dim
        self.heads = nn.ModuleDict({"odor": Head(d, n_odor),
                                    "platform_a": Head(d, n_platform_a)})
        if n_platform_b is not None:
            self.heads["platform_b"] = Head(d, n_platform_b)
        # Arbitrary extra sensor platforms. Needed because UCI 270 and UCI 251
        # cannot share a head: same sensor types, but 270 ships pre-extracted
        # DR/EMA features while 251's features are computed here from raw
        # traces, so the two output spaces are not commensurable.
        for name, dim in (head_dims or {}).items():
            self.heads[name] = Head(d, dim)

    def forward(self, data, task: str = "odor", return_attention: bool = False):
        if return_attention:
            g, node_h, att = self.encoder(data, return_attention=True)
            return self.heads[task](g), node_h, att
        return self.heads[task](self.encoder(data))

    def embed(self, data) -> torch.Tensor:
        """Graph embedding for the Phase 6 galaxy view."""
        return self.encoder(data)


def signature_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine loss on the response signature.

    The target is L2-normalised by construction (see rtn.targets), so direction
    is the whole content of it and a plain MSE would spend capacity fitting a
    magnitude that carries no information.
    """
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
