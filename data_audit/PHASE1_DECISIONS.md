# Phase 1 Decisions Log

Decisions taken after the Phase 0 audit, with the evidence for each.

## Scope decision, 2026-09-07

- **Option C, two-stage transfer.** Stage 1 pretrains the GAT on OpenPOM GS-LF
  odor descriptors, Stage 2 transfers to MOX array response regression with
  leave-molecule-out CV as the headline.
- **Zenodo 15681119 included with a separate output head.** Adds diacetyl and
  2-phenylethanol, the two most structurally interesting analytes. Total 13
  molecules across two sensor platforms.

## Pretraining corpus: exclude the analytes up front

8 of the 13 analytes appear in OpenPOM (methanol, ethanol, acetaldehyde,
acetone, butan-1-ol, toluene, diacetyl, 2-phenylethanol). Leaving them in leaks
each leave-molecule-out fold's held-out molecule into Stage 1.

Two fixes: retrain Stage 1 per fold (13x the compute), or remove all 13 once.
Chose the latter. The pretraining corpus is fixed at **4,975 molecules** across
all folds, and every quoted number refers to that, not the headline 4,983.

Corpus stats: 138 descriptors, mean 4.88 labels/molecule, rarest descriptor has
31 positives, median 12 heavy atoms (vs median 3 for the analytes). Every
molecule featurizes without error.

## Target definition: z-score per feature, then L2 normalise

From `scripts/check_target_variants.py`:

| Target | within | between | margin | eff. dim |
|---|---|---|---|---|
| raw 128d, L2 | 0.967 | 0.819 | 0.148 | 1.42 |
| raw DR-only 16d, L2 | 0.967 | 0.819 | 0.148 | 1.42 |
| z-scored 128d, L2 | 0.582 | -0.037 | **0.620** | **3.25** |

Two things forced this choice:

1. **The raw 128-dim and raw 16-dim targets are numerically identical.** The
   steady-state DR features are orders of magnitude larger than the EMA
   transient features and dominate the norm completely. Normalising the raw
   128-dim vector silently trains on DR alone and throws away all the transient
   dynamics. This is an easy mistake to make and worth stating in the writeup.
2. **Effective dimensionality of the raw target is 1.26.** That is a scalar, not
   a regression target. Standardising first raises it to 3.25.

Aggregation is median-of-batch-medians, not mean. Per-molecule cosine minima go
below zero in every gas (a handful of sign-flipped measurements, almost
certainly acquisition faults) and batch sizes range from 161 to 3,613, so a
plain mean is doubly wrong.

The scaler is fitted per fold on training measurements only. Fitting on
everything would leak the held-out molecule's response distribution.

## The target is learnable, and the confusion structure is the finding

`scripts/check_target_stability.py`: within-molecule cosine to own centroid is
0.967 with per-batch drift between 0.83 and 0.99. The signature is reproducible.
Drift is present but does not swamp chemistry.

Platform A confusion matrix (UCI 270, 6 molecules, z+L2 target):

|  | acetald | acetone | ammonia | ethanol | ethylene | toluene |
|---|---|---|---|---|---|---|
| acetaldehyde | 1.00 | 0.19 | 0.40 | 0.02 | -0.53 | 0.47 |
| acetone | | 1.00 | -0.49 | 0.13 | -0.54 | -0.34 |
| ammonia | | | 1.00 | -0.47 | -0.07 | **0.77** |
| ethanol | | | | 1.00 | -0.17 | -0.49 |
| ethylene | | | | | 1.00 | -0.32 |
| toluene | | | | | | 1.00 |

**Ammonia and toluene sit at 0.77.** NH3, one heavy atom, inorganic base, versus
an aromatic hydrocarbon with seven. About as chemically different as this
analyte set gets, and the array gives nearly the same signature. That is the
concrete instance of the question the project was built to answer, and it is a
better headline example than anything in the original spec.

Note this pair only appears once the target is standardised. On the raw target
the apparent worst pair is acetaldehyde/ethanol at 0.995, which is an artifact
of DR dominance, and is also a chemically boring pair (ethanol oxidises to
acetaldehyde, and a MOX sensor oxidises the analyte at its surface, so similar
surface species are expected). Reporting that pair would have been a mistake.

## Splits

- Headline: leave-molecule-out, 13 folds. Every fold trains on 12 molecules.
  Report the distribution across folds, never just the mean.
- Secondary random split is **batch-wise, not measurement-wise**. Dataset 270's
  batches confound gas identity with acquisition time (arXiv 2108.08793); a
  measurement-level random split puts the same batch on both sides and inflates
  the number.
- Leave-batch-out kept as a control to separate drift failure from structure
  failure.
- Scaffold split is **not** usable as the spec's fallback here: the 13 analytes
  fall into exactly 2 Bemis-Murcko groups (10 acyclic, 3 benzene). Retained only
  for the Stage 1 corpus, where it is meaningful.

## Interpretability scope

Attention maps are only claimed for analytes with **>= 5 heavy atoms**:
butan-1-ol, benzene, diacetyl, toluene, 2-phenylethanol. Below that every atom
is effectively the functional group and a heatmap carries no information a
composition vector would not. Phase 4 plausibility claims are restricted to this
subset and the writeup must say so.

## Model

Shared GATv2 encoder (3 layers, 128 hidden, 4 heads, edge features) with three
heads: odor (138), platform_a, platform_b. `attention=False` swaps in GCN
through the same code path so the Phase 2 ablation isolates attention rather
than confounding it with a different training loop.

Pooling is mean + max + sum. Sum is included because sensor response scales with
the amount of reducible material, which is size-extensive and which mean pooling
discards by construction.

Loss is cosine, not MSE: the target is unit-norm by construction, so direction
carries all the information.

`add_self_loops=True` is what makes methane and ammonia work at all.
tests/test_model.py covers the zero-edge path including attention extraction and
the backward pass.
