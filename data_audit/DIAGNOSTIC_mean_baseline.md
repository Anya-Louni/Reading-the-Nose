# Diagnostic: why is a constant prediction molecule-dependent?

Phase 2 flagged that the mean baseline scores +0.707 on the ammonia fold and
+0.809 on toluene, but -0.180 on 2-phenylethanol. A constant should not care
which molecule is held out. Script: `scripts/diagnose_mean_baseline.py`.

## Two hypotheses, both wrong

**H1: the mean baseline is a random direction in disguise.** Signatures are
z-scored across training rows, so the training molecule centroids should nearly
cancel, leaving a near-zero vector whose direction is numerical residue. If so,
its 0.302 average is luck across 8 folds.

Half right. The cancellation is real: the mean training signature has norm 0.20
against the 0.53 expected if the centroids were independent unit vectors, a 2.2x
shrinkage. But the direction is not noise:

| | mean cosine to truth | std |
|---|---|---|
| mean baseline | **+0.302** | 0.377 |
| random direction (2,000 draws/fold) | +0.012 | 0.058 |

In 8 of 9 fold-platform combinations the baseline exceeds the 95th percentile of
the random control. It carries genuine signal. **H1 rejected**, and the Phase 2
statement "no model beats the mean baseline" stands as a meaningful claim.

**H2: it scores high where the held-out molecule has a near-duplicate in the
training set.** Ammonia and toluene sit at cosine 0.77 on the sensor array, and
each is in the other's training set, which would drag the training mean toward
the held-out target.

| fold | mean baseline | similarity to nearest training molecule | nearest |
|---|---|---|---|
| 2-phenylethanol | -0.180 | 0.885 | ethanol |
| acetaldehyde | +0.248 | 0.352 | toluene |
| acetone | +0.263 | 0.752 | acetaldehyde |
| ammonia | +0.707 | 0.668 | toluene |
| diacetyl | +0.739 | 0.326 | ethanol |
| ethanol (A) | -0.185 | 0.552 | acetaldehyde |
| ethanol (B) | +0.099 | 0.861 | 2-phenylethanol |
| ethylene | +0.217 | 0.382 | ethanol |
| toluene | +0.809 | 0.681 | ammonia |

Pearson r = **-0.297**, the wrong sign, with clear counterexamples in both
directions: 2-phenylethanol has the second-highest neighbour similarity (0.885)
and the worst score, diacetyl has almost the lowest (0.326) and the second-best.
**H2 rejected.**

## What the diagnostic did establish

1. The baseline is real signal, not noise. The Phase 2 comparison is valid.
2. The constant prediction is a heavily shrunken residual vector, a direct
   consequence of fitting the scaler inside each fold. This is the correct thing
   to do for leakage reasons and should not be changed, but it means the "mean"
   baseline is not the naive object its name suggests.
3. The molecule-dependence is real and is **not** explained by structural or
   sensor-response near-duplication. With n = 9 and platform B contributing only
   two training molecules per fold, the geometry is too degenerate to attribute
   further. Leaving it unexplained is the honest position.

## The actionable finding

**Platform B leave-molecule-out folds are degenerate.** Three molecules total, so
holding one out leaves two, and those two are close to antipodal:

| fold | training molecules | mean pairwise cosine | norm of their mean |
|---|---|---|---|
| 2-phenylethanol | diacetyl, ethanol | -0.984 | 0.091 |
| ethanol | diacetyl, 2-phenylethanol | -0.987 | 0.080 |
| diacetyl | ethanol, 2-phenylethanol | -0.737 | 0.363 |

A two-molecule training set with near-antipodal targets cannot support learning.
Three of the nine fold-platform combinations in the Phase 2 table are therefore
uninformative and inflate its variance. The results script now breaks the
leave-molecule-out table out by platform so this is visible rather than buried
in a pooled average. Platform A (6 molecules now, 11 once UCI 251 lands) is the
only one that can carry the headline claim.

## Change made as a result

Added an **ECFP 1-NN baseline**: predict the signature of the structurally
nearest training molecule by Tanimoto on ECFP4, with no learning at all. This is
the sharpest available test of the project's premise. If structurally similar
molecules produce similar array responses, copying the nearest neighbour should
beat a constant; if it does not, the structure-to-response relationship is not
present to be learned and model capacity is irrelevant.

Early indication on 8 molecules, one seed: 1-NN scores 0.306 cosine against the
constant's 0.302, but has the best mean rank of any model (2.11 vs 2.78). Not
conclusive at this sample size. It goes into the 13-molecule table.

A useful side effect: on the random split, 1-NN retrieves the held-out
molecule's own training signature (Tanimoto 1.0 against itself) and scores 0.603,
identical to every learned model. That is independent confirmation that the
random-split number is memorisation rather than generalisation.
