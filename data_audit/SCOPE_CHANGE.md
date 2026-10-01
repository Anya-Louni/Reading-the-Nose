# Scope change: odour prediction becomes the primary task

Date: 2026-09-07. Rationale recorded here so the writeup does not later
misrepresent the order in which things happened.

## What changed

The 13-molecule sensor cross-reactivity task is no longer the headline. It stays
in full as a **labelled case study**: the Phase 2 tables, the Phase 4 attention
and integrated-gradient findings, the diacetyl near-failure, and the confusion
pairs are all retained and none of that work is discarded.

The new primary task is **odour descriptor prediction on the OpenPOM corpus**,
4,975 molecules and 138 descriptors, run through the same infrastructure with no
rebuild: same featurizer, same encoder and heads, same baseline ordering, same
explainability and sanity-check harness.

## Why, stated plainly for the writeup

The sensor task was run **first**, as the more novel and harder test. It failed
cleanly and the failure is well characterised: under leave-molecule-out across
13 molecules no model beat a constant prediction, and the GAT's attention came
out unable to pass the mandatory sanity checks as explanations.

That failure leaves one question open: was the **method** wrong, or was 13
molecules with a median of 3 heavy atoms simply **too little data**? The odour
task exists to answer that and nothing else. It is the same pipeline on 383x the
molecules. If the graph models now separate from the baselines, the negative was
about data. If they still do not, it was about the method.

The writeup must say this in this order. The odour task was not the original
plan and must not be presented as though it were.

## Novelty, honestly

Odour prediction from molecular graphs is **not novel**. OpenPOM's own paper does
exactly this, and its published ensemble is stronger than anything here. The
differentiator is not the task; it is the rigour applied to it:

- every explanation put through the Adebayo weight-randomisation check and a
  deletion faithfulness test before any claim is made about it,
- near-failures documented rather than dropped, diacetyl being the clearest case:
  a chemically perfect, highly faithful attribution that turned out to survive
  destroying the model's weights,
- a negative result reported straight rather than buried.

Phase 9 framing follows from that. This is a portfolio piece about method
discipline, not a claim to a new prediction task.

## Constraints carried over

- The mandatory sanity check is **not** skipped because the dataset is larger.
  It is the project's actual differentiator.
- The scaffold split is now viable and meaningful; on the 13 analytes it
  collapsed to two Bemis-Murcko groups and was useless.
- The "GAT + OpenPOM pretraining" arm from the sensor table is **circular** on
  this task and has been replaced by transfer in the opposite direction, an
  encoder fitted on the sensor task and applied to odour. See the leak note in
  `scripts/run_odor_phase2.py`: the first attempt warm-started that encoder from
  OpenPOM weights, which put the corpus back in through the side door and showed
  up immediately as an implausible smoke-test score.

## Visualisation

The existing galaxy and blind-spots build is kept. Odour-prediction results
become the main explanatory layer, with the sensor-confusion work as a clearly
separated and explicitly labelled small-sample stress test on the same page.
