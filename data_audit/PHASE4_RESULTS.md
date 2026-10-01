# Phase 4 Results: interpolation-only explainability (sensor task)

> **Corrected 2026-09-08.** The original version of this document reported "the
> learned attention is uniform" as its headline. That was an artifact of
> aggregating attention by destination node, which measures the GATv2 softmax
> normalisation constant rather than the model. See
> `RETRACTION_attention_uniform.md`. All attention numbers below are from the
> corrected by-source aggregation. Integrated-gradients results are unchanged,
> as they never used attention.

Model trained on all 13 analytes (interpolation by construction), 3 seeds,
explained on the 8 molecules with >= 3 heavy atoms.

## Attention: real maps, but they largely survive weight randomisation

Attention spread per molecule-seed: median 0.136, max 0.324. Not uniform.

Corrected per-atom attention:

| molecule | attention |
|---|---|
| ethanol (CCO) | C 0.239, **C 0.428**, O 0.333 |
| acetone (CC(C)=O) | C 0.170, **C 0.465**, C 0.170, O 0.194 |
| toluene (Cc1ccccc1) | C 0.125, **C 0.245**, ring 0.117-0.137 |

Adebayo sanity check, Spearman between trained and weight-randomised maps:

| method | mean | sd |
|---|---|---|
| attention | **+0.488** | 0.238 |
| integrated gradients | +0.311 | 0.609 |

Attention correlates +0.49 on average with the map from an untrained model. On
the larger odour corpus, where this has real power, the same figure is +0.79 with
60.4% of 750 molecule-seed observations failing outright. See `ODOR_PHASE4_RESULTS.md`; that is the
version of this result to quote, since 8 molecules cannot support the claim.

Faithfulness (fraction beating the random-masking control):

| method | all 8 | >= 5 heavy atoms |
|---|---|---|
| attention | 0.583 | 0.467 |
| integrated gradients | 0.833 | 0.733 |

## Integrated gradients does produce structure

The second attribution method, run as the cross-check the plan required, gives
non-uniform and reproducible attributions.

| molecule | per-atom IG | top atom |
|---|---|---|
| ethanol (CCO) | C 0.175, C 0.314, **O 0.511** | hydroxyl O |
| butan-1-ol (CCCCO) | C 0.182, C 0.169, C 0.163, C 0.172, **O 0.313** | hydroxyl O |
| acetaldehyde (CC=O) | C 0.141, **C 0.474**, O 0.385 | carbonyl C |
| acetone (CC(C)=O) | C 0.149, **C 0.487**, C 0.149, O 0.214 | carbonyl C |
| diacetyl (CC(=O)C(C)=O) | C 0.043, **C 0.358**, O 0.099, **C 0.357**, C 0.043, O 0.100 | both carbonyl C |
| toluene (Cc1ccccc1) | C 0.087, **C 0.212**, ring 0.135-0.144 | ipso ring C |

Diacetyl is a useful internal check: the molecule is symmetric about its centre
and IG returns 0.358 and 0.357 for the two equivalent carbonyl carbons, and
0.043 for both terminal methyls. Nothing enforced that.

Benzene returns an exactly uniform map from both methods, which is correct: all
six atoms are equivalent, so any per-atom attribution must be constant. Its rank
correlations are undefined rather than failed, and it carries no atom-level
story.

## The checks, and what they disqualify

| molecule | IG sanity (Spearman vs randomised) | IG faithful? | quotable |
|---|---|---|---|
| ethanol | -0.500 pass | yes (gap +0.167) | **yes** |
| acetaldehyde | +0.500 pass | yes (gap +0.198) | **yes** |
| acetone | +0.481 pass | yes (gap +0.246) | **yes** |
| butan-1-ol | -0.100 pass | yes (gap +0.046) | **yes** |
| toluene | +0.702 borderline | yes (gap +0.053) | no |
| diacetyl | **+0.943 FAILS** | yes (gap +0.414) | no |
| 2-phenylethanol | +0.150 pass | **no** (gap -0.004) | no |
| benzene | undefined (symmetric) | n/a | no |

Overall IG faithfulness: 0.833 of (molecule, seed) cases beat the random-masking
control, 0.733 on the >= 5 heavy-atom subset.

**Diacetyl is the cautionary case and belongs in the writeup.** Its attribution
is the most chemically satisfying of the set, perfectly symmetric and landing on
exactly the two carbonyl carbons, and it has the largest faithfulness gap
(+0.414). It also fails the Adebayo randomisation check at +0.943: nearly the
same map comes out of a model whose weights have been reinitialised. A
chemically beautiful, highly faithful, and demonstrably meaningless explanation.
Without the sanity check it would have been the figure in the poster.

**Attention and IG partly agree** once attention is aggregated correctly: mean
Spearman across molecules is positive, and identical on acetaldehyde and acetone
(+1.000, both 3-4 atom molecules where the rankings coincide trivially). The
previously reported -0.092 was comparing IG against a constant and is withdrawn.
The reason to prefer IG is the sanity check, not the agreement level.

## Plausibility, with citations

Restricted to the four molecules that pass both checks: ethanol, acetaldehyde,
acetone, butan-1-ol. Every chemistry claim below is looked up and cited rather
than assumed.

Established behaviour of SnO2-type MOX sensors, which is what all three
platforms use:

1. Sensing proceeds by the analyte reacting with chemisorbed oxygen species
   (O-, O2-, O2^2-) on the oxide surface, which releases trapped electrons back
   into the conduction band and changes resistance. The reactive site is
   therefore wherever the molecule is oxidised, not the molecule as a whole.
   (Degler, Wagner & Weimar, *Sens. Actuators B* 2022, "Current state of
   knowledge on the metal oxide based gas sensing mechanism".)
2. SnO2 surfaces show strong and preferential adsorption of hydroxyl-bearing
   molecules. (Frontiers in Chemistry 2020, 8:321, "Recent Advances of
   SnO2-Based Sensors for Detecting Volatile Organic Compounds".)
3. In a systematic functional-group study across alkanes, alcohols, aldehydes,
   ketones, acids and esters, aldehydes produce strong responses while alkanes
   produce nearly none at typical operating temperature, and alcohols show only
   weak dependence on chain length. (Hahn et al., *Sens. Actuators B* 2001,
   SOMMSA approach.)

Against that, the four quotable attributions are consistent:

- **Ethanol and butan-1-ol**: IG puts the most weight on the hydroxyl oxygen
  (0.511 and 0.313), matching point 2.
- **Acetaldehyde and acetone**: IG puts the most weight on the carbonyl carbon
  (0.474 and 0.487), matching point 1, since the carbonyl carbon is the site of
  oxidative attack.
- **Chain carbons are consistently the least weighted**: butan-1-ol's four
  carbons sit at 0.163-0.182 against the oxygen's 0.313; acetone's two methyls
  at 0.149 against the carbonyl carbon's 0.487. Matches point 3, alkane-like
  carbon contributing little.
- Butan-1-ol's near-flat carbon chain (0.163 to 0.182 across four carbons) is
  consistent with the weak chain-length dependence reported for alcohols.

This is agreement in the qualitative direction only. Four molecules, one
attribution method, no quantitative comparison to measured functional-group
sensitivities. It is a plausibility check that passed, not evidence that the
model learned chemistry.

## Bugs found and fixed during this phase

`randomize_weights` originally reinitialised by looping over parameters and
zeroing everything 1-dimensional. LayerNorm's `weight` is 1-D, so zeroing it
made every LayerNorm emit exactly zero and the network was dead rather than
random. Integrated gradients then returned all-zero attributions, the normaliser
fell back to uniform, and every IG sanity correlation came out NaN. A dead model
is not a control, and the mandatory sanity check was silently vacuous. Fixed by
asking each module to `reset_parameters()`, with an assertion that no
normalisation weight ends up all-zero.

## What Phase 4 means for the project

The pitch was "attention weights explain why a cheap sensor confuses two
molecules". Phase 2 showed attention does not help predict the response. Phase 4
shows the attention maps that exist correlate +0.49 with maps from an untrained
model on this small set, and +0.79 with 60.4% outright failures on the odour
corpus where the check has power. So attention is not a trustworthy explanation
here, but for a different and better-evidenced reason than the retracted
uniformity claim.

What survives is narrower and still worth showing: integrated gradients on the
interpolation model recovers chemically sensible reactive sites on four small
oxygen-containing molecules, and the sanity check caught one case (diacetyl)
where a compelling explanation was spurious. That, plus the sensor confusion
geometry from Phase 2, is the honest content for Phase 6.
