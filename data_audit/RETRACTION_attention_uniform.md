# Retraction: "the learned attention is uniform"

Reported 2026-09-07 on the sensor task and 2026-09-08 on the odour task.
Retracted 2026-09-08 after an independent verification of the attention
extraction code.

## The claim, as made

> The learned attention is uniform to float32 precision. Maximum spread within
> any (molecule, seed) is 1.99e-08 against 0.53 for integrated gradients. There
> is no attention map to interpret.

Repeated on the odour task as "uniform on 100% of 750 molecule-seed
observations, max spread 9.00e-09", and used to argue that dataset size could not
explain the collapse.

## What was actually happening

`GATv2Conv` applies softmax over the edges arriving at each **destination** node.
The incoming attention coefficients for any node therefore sum to exactly 1.0 —
in every graph, in every model, trained or randomly initialised. It is the
softmax normalisation constant.

`attention_scores` aggregated with `np.add.at(acc, dst, alpha)`, summing by
destination. It could not have produced anything except a flat map. The reported
"spread" of 1e-08 was float32 rounding error in a quantity that is 1.0 by
construction.

Raw per-edge attention on 2-phenylethanol, from the same checkpoint, via
`return_attention_weights=True`:

    per-node sum of INCOMING alpha : min 1.000000  max 1.000000  spread 1.2e-07
    per-node sum of OUTGOING alpha : min 0.859510  max 1.211330  spread 3.5e-01
    raw per-edge alpha             : min 0.2137    max 0.6583    spread 0.4446

The attention varies by a factor of three across edges. It was never uniform.

Toy confirmation, a 3-node path with nodes 0 and 2 given identical features and
node 1 made distinct:

    by destination : [1.000, 1.000, 1.000]   spread 3e-08
    by source      : [0.758, 1.484, 0.758]   spread 0.726

By-source is symmetric in the two equivalent nodes and singles out the distinct
one, which is what a functioning mechanism should do.

## What was and was not a bug

- **The extraction was numerically correct.** Replicating the pipeline's
  reduction from raw layer output matches its published numbers to 0.000e+00.
  Nothing read the wrong tensor.
- **The aggregation choice was the bug.** Summing a softmax over its own
  normalisation axis measures the constraint, not the model.

## What this invalidated

Withdrawn entirely:

- "The learned attention is uniform" on both tasks.
- All attention faithfulness figures computed before the fix (0.208 sensor,
  20.4% odour) and all attention sanity figures (+0.020 sensor, -0.003 odour).
  These ranked tied values.
- The claim that uniform attention explained why the GAT only beats a GCN by
  +0.0046.
- The refutation of the sensor-task training-regime hypothesis. That hypothesis
  is now neither supported nor refuted; the evidence used against it is gone.
- The instruction in the visualisation README that "no attention map is shown
  because there is nothing to draw". The reason was wrong even though showing
  attention maps unqualified would still be a mistake, for the new reason below.

Unaffected, because none of it used attention:

- Every Phase 2 performance number on both tasks, including GAT vs GCN
  (+0.0046) and graph structure vs fingerprint (+0.100).
- All integrated-gradients results on both tasks.
- The sensor task's negative result and the confusion-pair analysis.
- The diacetyl case study.

## What replaced it

The corrected finding is stronger than the artifact. Attention maps are real and
varied, and they fail the Adebayo randomisation check on 50.8% of held-out
molecules (mean rho +0.78). A third of all attention maps are simultaneously
faithful and reproducible from an untrained network. See
`ODOR_PHASE4_RESULTS.md`.

## Process failure, for the record

Two things should have caught this before it was reported twice.

1. A quantity constant to seven decimal places across every molecule, every seed
   and two unrelated datasets is the signature of an identity, not a
   measurement. That should have prompted the question "what is mathematically
   forced to be constant here" before it prompted a writeup.
2. The finding was convenient. It explained the GAT-vs-GCN result neatly and made
   a striking headline, and it was reported as a headline rather than stress
   tested. A result that tidy deserved more suspicion, not less.

The fix carries two regression tests (`test_attention_by_destination_is_the_
softmax_constant`, `test_raw_edge_attention_varies`) that pin both behaviours,
and `attention_scores` now takes an explicit `reduce` argument with the
destination option retained only for reproducing the retracted numbers.
