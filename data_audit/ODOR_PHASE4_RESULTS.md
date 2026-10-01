# Odour task, Phase 4: explainability with enough molecules to mean something

> **This document was rewritten after an independent verification found a bug in
> the attention aggregation.** The earlier version reported "the attention is
> uniform to floating-point precision" on both tasks. That was wrong and is
> retracted. See `RETRACTION_attention_uniform.md` for what happened and what it
> invalidated. Everything below uses the corrected aggregation.

250 held-out molecules from a scaffold-disjoint test fold x 3 seeds = 750
molecule-seed observations, on models at validation macro AUROC 0.8585 / 0.8600 /
0.8558.
Harness is `rtn.explain`; `attention_scores` now aggregates by source, verified
against raw PyG layer output in `scripts/verify_attention_raw.py`.

## 1. Attention is not uniform, but it does not survive its sanity check

Attention produces real, varied per-atom maps:

| | median | 90th pct | max | uniform (spread < 1e-6) |
|---|---|---|---|---|
| attention spread | 0.0666 | 0.101 | 0.163 | **0 of 750** |
| IG spread | 0.197 | 0.355 | 0.797 | 0 of 750 |

Median attention spread is 0.88x the size of a single uniform weight, so the
variation is substantial, not marginal.

**Then it fails the Adebayo randomisation check most of the time.**

| method | n | mean rho | fail (rho > 0.8) | near zero (\|rho\| < 0.3) |
|---|---|---|---|---|
| attention | 750 | **+0.793** | **60.4%** | 1.6% |
| integrated gradients | 750 | +0.033 | 6.4% | 40.3% |

Spearman +0.79 on average between the attention map from a trained model and the
attention map from the same architecture with its weights reinitialised. Only
1.6% of attention maps are unrelated to their randomised counterpart.

The attention map is largely a function of the graph and the input features, not
of anything the model learned. This is the standard critique of
attention-as-explanation, measured here rather than asserted.

## 2. Faithfulness does not rescue it

Deletion AUC against a random-masking control:

| method | gap | 95% CI | faithful |
|---|---|---|---|
| attention | +0.0286 | [+0.0263, +0.0309] | 54.9% |
| integrated gradients | **+0.0785** | [+0.0757, +0.0813] | **98.0%** |

Attention is faithful on 54.9% of molecules. That is well above chance and would
look like a pass on its own. But crossing the two checks:

| attention outcome | share of 750 |
|---|---|
| **faithful AND fails sanity (rho > 0.8)** | **36.3%** (272/750) |
| faithful AND passes sanity | 18.7% |

**More than a third of all attention maps are faithful and meaningless
simultaneously.** They predict their own model's behaviour under masking while
being reproducible from an untrained network. For integrated gradients the same
quantity is 6.4% (48/750).

This is the diacetyl case from the sensor task, generalised: on the sensor task
it was one molecule in eight for IG and looked like a fluke. Here it is a
measured rate, and it is four times worse for attention than for gradients.

**If you validate an explanation with a faithfulness metric alone, you will ship
spurious maps at roughly 1 in 3 for attention and 1 in 13 for integrated
gradients, and the faithfulness number will not tell you which ones.** That is
the single most useful result this project produced.

The two independent corrected runs agree closely, which is the reason to trust
these as point estimates rather than one-off draws:

| run | n | attention fail | attention faithful | IG fail | IG faithful |
|---|---|---|---|---|---|
| 120 mol x 2 seeds | 240 | 50.8% | 59.6% | 7.5% | 99.2% |
| **250 mol x 3 seeds** | **750** | **60.4%** | **54.9%** | **6.4%** | **98.0%** |

## 3. Cross-method agreement

Spearman(attention, IG) = +0.273 mean, +0.286 median over 750 observations. The
two methods partially agree, which is expected now that both produce real maps.
The disagreement that remains is not resolvable from this data alone, and the
sanity check is the reason to prefer IG rather than the agreement level.

## 4. What this means alongside Phase 2

Phase 2 found the GAT beats a GCN by +0.0046 macro AUROC, which is a real but
tiny effect. The attention weights are non-uniform, so that is no longer
explained by degenerate attention. The two results sit together as: the model
does compute differentiated attention, that attention is mostly determined by
structure rather than learning, and it buys almost nothing in accuracy over
simple neighbourhood averaging.

## What carries forward

- **Integrated gradients is the method to use.** 98.0% faithful, 6.4% sanity
  failure. Attention is not a substitute.
- **Attention maps may be shown, with the failure rate attached.** They are not
  uniform, so there is something to display, but 60.4% of them are reproducible
  from an untrained model and any presentation must say so.
- The Phase 4 hypothesis about the sensor task's training regime is **neither
  confirmed nor refuted**; the evidence previously used to refute it was the
  uniformity artifact and is withdrawn.

## Caveats

1. 250 molecules x 3 seeds, one scaffold fold, matching the size of the
   retracted run. Point estimates are stable across the two independent
   corrected runs (see the table above), but both draw from the same scaffold
   fold, so fold-to-fold variation is not captured.
2. One architecture and one hyperparameter setting throughout. A different head
   count might behave differently; not tested.
3. Deletion masks atom features to zero rather than removing atoms, applied
   identically to the attribution order and the control.
4. The randomisation control reinitialises every module via `reset_parameters()`.
   An earlier version zeroed all 1-D parameters, which killed LayerNorm and made
   the control a dead network rather than a random one; fixed and guarded.
