# Odour task, Phase 2: baseline table over 4,975 molecules

The second experiment in this project, run to diagnose the first. See
`SCOPE_CHANGE.md` for why it exists and `PHASE2_RESULTS.md` for the sensor task
it is diagnosing.

Setup: OpenPOM GS-LF, 4,975 molecules, 138 descriptors, multi-label. 3 folds x 3
seeds, early stopping on a validation slice carved from the training portion.
Same featurizer, encoder, heads and baseline ordering as the sensor task.

Metrics are macro AUROC and macro average precision, averaged over descriptors
that have both classes present in the test fold. A descriptor absent from a fold
is skipped rather than imputed; with the rarest descriptor holding 31 positives
corpus-wide, some are always absent and imputing would quietly move the mean.

## Read this before the table

**This is not a competitive odour-prediction result and the numbers below must
not be read as one.** OpenPOM's published ensemble is stronger than any single
model here, the task is theirs and not novel, and no hyperparameter search was
run. 0.75 macro AUROC is not "a better odour predictor"; it is what this
deliberately untuned pipeline scores on a hard split.

The claim being made is narrower and does not depend on beating anyone:

> Using OpenPOM's own dataset and setup, graph structure matters a great deal and
> attention does not.

That comparison is internal to the table. It is a statement about which
architectural ingredient carries the signal, measured against baselines run under
identical conditions, and it holds regardless of where the absolute numbers sit
relative to the state of the art.

## Table

### Scaffold split (headline)

| model | macro AUROC | sd | macro AP | sd |
|---|---|---|---|---|
| constant | 0.5000 | 0.0000 | 0.0353 | 0.0022 |
| ECFP + MLP | 0.6483 | 0.0241 | 0.0967 | 0.0103 |
| GCN | 0.7486 | 0.0383 | 0.1439 | 0.0238 |
| GAT | 0.7533 | 0.0423 | 0.1450 | 0.0235 |
| GAT + sensor transfer | 0.7538 | 0.0365 | 0.1476 | 0.0208 |

### Random split (secondary)

| model | macro AUROC | sd | macro AP | sd |
|---|---|---|---|---|
| constant | 0.5000 | 0.0000 | 0.0354 | 0.0003 |
| ECFP + MLP | 0.7623 | 0.0033 | 0.1790 | 0.0025 |
| GCN | 0.8587 | 0.0029 | 0.2587 | 0.0076 |
| GAT | 0.8598 | 0.0052 | 0.2605 | 0.0171 |
| GAT + sensor transfer | 0.8586 | 0.0032 | 0.2643 | 0.0134 |

## The question this was run to answer

**It was the data, not the method.**

On the sensor task, all five models landed within 0.002 of each other on the easy
split and none beat a constant on the hard one. Here, on a scaffold-disjoint
split, graph structure beats the flat fingerprint by **+0.100 AUROC (GCN) and
+0.105 (GAT), winning 9 of 9 fold-seed pairs**, with bootstrap CIs of
[+0.088, +0.114] and [+0.092, +0.120]. Nowhere near zero.

The pipeline is unchanged. The featurizer, the encoder, the heads, the training
loop and the baseline ordering are the same objects that produced the null. Given
enough molecules they separate cleanly, so the 13-molecule negative was a
statement about the dataset, not about graph neural networks.

## Attention still does not earn its place

This is the second finding, and it survives the change of dataset.

| comparison | scaffold AUROC | 95% CI | folds won |
|---|---|---|---|
| GCN over fingerprint | +0.1003 | [+0.0876, +0.1143] | 9/9 |
| **GAT over GCN** | **+0.0046** | **[+0.0007, +0.0085]** | **7/9** |

Attention clears zero on scaffold AUROC, but only just, and it does not clear
zero on any other measure:

| comparison | mean | 95% CI |
|---|---|---|
| GAT - GCN, scaffold AP | +0.0011 | [-0.0012, +0.0032] |
| GAT - GCN, random AUROC | +0.0011 | [-0.0006, +0.0030] |
| GAT - GCN, random AP | +0.0017 | [-0.0056, +0.0087] |

Put on one scale, on the scaffold split:

    fingerprint over constant  +0.1483
    graph structure over fingerprint  +0.1003
    attention over graph structure    +0.0046

Attention buys about 4.5% of what graph structure buys, on one of four measures.
The honest statement is that **graph structure is what matters and attention is
close to free of effect**, which is exactly what the sensor task suggested and
what the uniform attention maps in Phase 4 showed from the other direction. The
baseline ordering was designed to separate those two claims and it did.

## Sensor-task transfer carries nothing

| comparison | mean | 95% CI | folds won |
|---|---|---|---|
| GAT + sensor - GAT, scaffold | +0.0006 | [-0.0062, +0.0083] | 4/9 |
| GAT + sensor - GAT, random | -0.0012 | [-0.0032, +0.0006] | 4/9 |

An encoder fitted on the 13-molecule sensor task transfers no usable signal to
odour prediction. Consistent with the sensor task having failed to learn a
structure-to-response relationship in the first place.

This arm replaced the "GAT + OpenPOM pretraining" slot from the sensor table,
which is circular here. The first attempt at it leaked: the sensor encoder was
warm-started from OpenPOM weights, so the arm was OpenPOM-pretrained then
sensor-finetuned then evaluated on OpenPOM, and scored an implausible 0.747
against 0.645 for every other model in a smoke test. Fixed by training the
sensor encoder from scratch (`use_pretrained=False`), with the reason recorded at
both call sites.

## Caveats

1. **The scaffold folds are unbalanced.** One Bemis-Murcko group dominates the
   corpus, so fold 1 trains on 2,416 and tests on 2,230 while fold 2 trains on
   3,170 and tests on 1,373. This is inherent to scaffold splitting and it makes
   the split genuinely hard, but the folds are not interchangeable and the
   scaffold standard deviations (0.024 to 0.042) are correspondingly larger than
   the random ones (0.003 to 0.005). Read the scaffold spread as fold-to-fold
   variation, not seed noise.
2. **This is not a competitive result.** Restated here because it is the caveat
   most likely to be dropped in summary: see the section above the table. The
   claim is "graph structure matters, attention does not, measured inside
   OpenPOM's own setup", never "we built a better odour predictor".
3. **No hyperparameter search was run.** The architecture and learning rate were
   carried over from the sensor task unchanged, deliberately, so the two tables
   are comparable. A tuned GAT might separate from a GCN; that was not tested,
   and the claim here is only about the untuned comparison.
4. **3 folds, 3 seeds.** Adequate for the +0.10 effect, marginal for the +0.0046
   one. The attention result is **"no meaningful effect detected", not "proven
   absent"**, and that distinction is not to be softened in any summary of this
   work. Three folds cannot resolve an effect that small; what the data support
   is that attention's contribution is far below graph structure's and is
   indistinguishable from zero on three of four measures.

## Summary paragraph (for the README / paper draft)

> The sensor cross-reactivity task was run first, as the more novel and harder
> test, and failed cleanly: across 13 molecules under leave-molecule-out no model
> beat a constant prediction, and the graph attention network's attention weights
> could not be trusted as explanations, which the mandatory sanity checks caught. To determine whether that negative reflected the method or the sample
> size, we ran the identical pipeline on odour-descriptor prediction over the
> 4,975-molecule OpenPOM corpus. It reflected the sample size. On a
> scaffold-disjoint split, graph models beat an ECFP fingerprint baseline by
> +0.100 macro AUROC (GCN) and +0.105 (GAT), winning 9 of 9 fold-seed pairs. The
> second finding survives the change of dataset: attention adds +0.0046 AUROC
> over a plain GCN, roughly 4.5% of what graph structure itself contributes, and
> does not separate from zero on macro average precision or on the random split.
> An encoder transferred from the sensor task carries no usable odour signal
> (+0.0006, CI [-0.0062, +0.0083]). Odour prediction from molecular graphs is not
> a novel task, and this is not a competitive result: OpenPOM's own ensemble is
> stronger, and the claim here is the internal comparison, that graph structure
> carries the signal and attention does not, not a claim to a better odour
> predictor. With three folds the attention comparison supports "no meaningful
> effect detected" rather than "proven absent". The contribution is the
> discipline applied to the task, including a documented near-failure where a
> chemically perfect and highly faithful attribution turned out to survive
> randomising the model's weights.
