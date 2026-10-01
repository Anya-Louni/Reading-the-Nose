# Phase 2 Results (final): 13 molecules, three sensor platforms

Run: 6 models, 13 leave-molecule-out folds, 3 batch-wise random splits, 3 seeds,
57 fold-platform-seed observations per model. Reported twice, with and without
removal of the shared common-mode component.

| platform | source | rows | dim | molecules |
|---|---|---|---|---|
| A | UCI 270 (sealed chamber) | 57 | 128 | 6 |
| A2 | UCI 251 (open wind tunnel) | 885 | 576 | 10 |
| B | Zenodo 15681119 (commercial nose) | 106 | 248 | 3 |

Union 13 molecules. Three heads, one shared encoder: A and A2 share sensor types
but not feature definitions (UCI 270 ships the authors' DR/EMA features, UCI 251
features are computed here from raw traces), so they cannot share an output
space.

Stage 1 pretraining on OpenPOM GS-LF (4,975 molecules, 138 descriptors,
scaffold split): validation macro AUC 0.803 / 0.809 / 0.790.

## Table 1: raw target

### Leave-molecule-out

| model | cosine | std | rank | top-1 |
|---|---|---|---|---|
| constant | **0.425** | 0.363 | 4.47 | **0.105** |
| ECFP 1-NN | 0.262 | 0.499 | 4.37 | 0.000 |
| ECFP + MLP | 0.329 | 0.326 | 3.97 | 0.053 |
| GCN | 0.389 | 0.386 | 4.02 | 0.000 |
| GAT | 0.352 | 0.429 | 4.09 | 0.000 |
| GAT + pretraining | 0.382 | 0.376 | 4.14 | 0.035 |

### Batch-wise random split

| model | cosine | std | rank | top-1 |
|---|---|---|---|---|
| constant | 0.406 | 0.460 | 4.63 | 0.123 |
| ECFP 1-NN | 0.791 | 0.215 | 1.23 | 0.860 |
| ECFP + MLP | 0.791 | 0.215 | 1.23 | 0.860 |
| GCN | 0.791 | 0.215 | 1.21 | 0.860 |
| GAT | 0.791 | 0.215 | 1.25 | 0.842 |
| GAT + pretraining | 0.791 | 0.215 | 1.23 | 0.860 |

## Table 2: common-mode removed

The offset is the mean of the *training* molecule centroids, so it is leak-free.

### Leave-molecule-out, with bootstrap CIs over folds (n = 13)

| model | cosine | 95% CI | above zero? |
|---|---|---|---|
| constant | -0.205 | [-0.317, -0.102] | below |
| ECFP 1-NN | +0.020 | [-0.179, +0.229] | no |
| **ECFP + MLP** | **+0.233** | **[+0.089, +0.388]** | **yes** |
| **GCN** | **+0.172** | **[+0.012, +0.344]** | **yes** |
| GAT | +0.068 | [-0.077, +0.209] | no |
| GAT + pretraining | +0.091 | [-0.062, +0.237] | no |

### Per platform (centered, leave-molecule-out)

| model | A (6 mol) | A2 (10 mol) | B (3 mol) |
|---|---|---|---|
| constant | +0.027 | **-0.402** | -0.035 |
| ECFP 1-NN | +0.349 | -0.190 | +0.034 |
| ECFP + MLP | +0.177 | +0.223 | +0.316 |
| GCN | +0.143 | +0.053 | +0.573 |
| GAT | +0.194 | -0.014 | +0.191 |
| GAT + pretraining | +0.164 | -0.044 | +0.447 |

## The A2 question, answered

Platform A2's constant baseline scored +0.536 on the raw target, far above the
other platforms, and the open question was whether that reflected genuine
structure or common-mode inflation.

**It was common mode.** Centering collapses it from +0.536 to -0.402, a swing of
0.94. There is no residual platform-A2 advantage; the raw number was measuring a
component every molecule shares.

Supporting evidence, all from `scripts/`: the leading principal axis of the A2
molecule centroids holds 51.5% of their variance and correlates with log
concentration at r = +0.58, and every gas in UCI 251 was run at a single
concentration (100 ppm butanol to 10,000 ppm ammonia). Centering drops A2's mean
off-diagonal centroid cosine from +0.583 to -0.058, matching platform A's -0.094.

## Two things that change how the tables should be read

**The constant baseline is only a valid floor on the raw target.** After
centering it becomes actively anti-informative, -0.205 with CI [-0.317, -0.102].
This is mechanical: the offset is the mean of the training centroids, so the
centered training centroids very nearly cancel and their mean is a small residual
pointing away from the held-out molecule. In the centered setting the correct
floor is zero, which is what a random direction scores (+0.012 over 2,000 draws
per fold, from the earlier diagnostic).

**The random split measures memorisation, not chemistry.** All five learned
models score identically to three decimals: 0.791 raw, 0.535 centered. That
includes ECFP 1-NN, which does no learning at all and simply retrieves the
held-out molecule's own training signature, because on this split that molecule
is in the training set. Five mechanisms with nothing in common agreeing exactly
is a lookup table, not a model of chemistry. This number should never be quoted
as evidence for the method.

## Summary paragraph (for the README / paper draft)

> We tested whether a graph attention network can predict low-cost metal-oxide
> sensor array responses from molecular structure, using every public MOX array
> dataset with identified single-molecule analytes: 13 molecules across three
> sensor platforms. Under leave-molecule-out cross-validation, the headline
> evaluation, no model beat a constant prediction on the raw response signature
> (best learned model 0.389 cosine against 0.425 for the constant). Removing the
> component shared by all molecules, which we show is driven substantially by
> concentration and accounts for 51.5% of the variance between molecule
> signatures on the wind-tunnel dataset, reveals a small but statistically
> reliable structure-to-response relationship: an ECFP fingerprint with an MLP
> reaches +0.233 cosine (95% CI [+0.089, +0.388]) and a GCN +0.172 ([+0.012,
> +0.344]), while the GAT (+0.068) and the GAT pretrained on 4,975 OpenPOM
> odorants (+0.091) do not separate from zero. Graph attention therefore does
> not earn its complexity here, and neither does odor-descriptor pretraining;
> a 2048-bit fingerprint captures what little structural signal exists at least
> as well. On a batch-wise random split all five learned models score identically
> (0.791 cosine), including a nearest-neighbour baseline that performs no
> learning, confirming that the split measures memorisation of per-molecule
> response patterns rather than generalisation. We report this as a negative
> result. The dominant limitation is the data, not the method: only 13 molecules
> with public array measurements exist, their median size is 3 heavy atoms, and
> in the largest dataset analyte identity is fully confounded with concentration.

## Limitations to carry into the writeup

1. **13 molecules, median 3 heavy atoms.** Five have one or two. This bounds
   everything and is a property of the MOX literature, which uses small
   calibration gases by design.
2. **Concentration is confounded with identity in UCI 251.** Every gas was run
   at one concentration. Not fixable post hoc.
3. **Platform B folds are degenerate.** Three molecules, so each fold trains on
   two whose signatures are near-antipodal. Its apparent GCN result (+0.573)
   rests on 3 folds and should not be quoted.
4. **Precision is low.** Standard deviations of 0.2 to 0.5 on 13 folds. The
   ECFP-vs-GAT paired differences (+0.06 to +0.16) all have CIs spanning zero,
   so the claim is "attention does not clear zero", not "ECFP beats the GAT".
5. **A single feature-extraction choice for UCI 251.** The 8 per-sensor
   statistics were defined once, before seeing results, and not tuned. A
   different reduction of the raw traces could change A2's numbers.
