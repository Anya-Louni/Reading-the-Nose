# Phase 0 Data Audit

Project: Reading the Nose. Status: complete, awaiting a scope decision before Phase 1.
Date: 2026-09-07.

## 1. Summary

I searched UCI, Zenodo, IEEE DataPort, arXiv and journal data articles for public
MOX / e-nose array datasets with identified single-molecule analytes.

Result: the union of **sensor-compatible** public datasets gives **11 unique molecules**,
not the 12-15 target. Adding one sensor-incompatible dataset gets to 13.

More importantly, the 11 molecules are mostly tiny. Median size is 3 heavy atoms.
That is a problem for the method, not just for the sample size. Details in section 5.

## 2. Datasets evaluated

| # | Dataset | Source | Sensors | Analytes | License | Verdict |
|---|---------|--------|---------|----------|---------|---------|
| 1 | Gas Sensor Array Drift at Different Concentrations (Vergara 2012) | UCI id 270 | 16 Figaro: 4x TGS2600, 4x TGS2602, 4x TGS2610, 4x TGS2620 | 6 | CC BY 4.0 | **Core** |
| 2 | Gas Sensor Arrays in Open Sampling Settings (Vergara 2013) | UCI id 251 | 9 modules x 8 Figaro: TGS2611, TGS2612, TGS2610, TGS2600, 2x TGS2602, 2x TGS2620 = 72 | 10 | CC BY 4.0 | **Core** |
| 3 | Twin Gas Sensor Arrays (Fonollosa 2016) | UCI id 361 | 8 Figaro: TGS2611, TGS2612, TGS2610, TGS2602 | 4 | CC BY 4.0 | Include (adds 0 molecules, adds sensor-replica info) |
| 4 | Gas Sensors for Home Activity Monitoring | UCI id 362 | same 8-sensor module as #2 | wine, banana (mixtures) | CC BY 4.0 | Reject: analytes are not single molecules |
| 5 | Gas Sensor Array Temperature Modulation | UCI id 487 | 7x Figaro TGS3870-A04, 7x FIS SB-500-12 | CO only | CC BY 4.0 | Reject: 1 analyte, different sensor family |
| 6 | Gas Sensor Array under Dynamic Gas Mixtures | UCI | 16 Figaro | ethylene, methane, CO (mixtures) | CC BY 4.0 | Optional: adds 0 molecules |
| 7 | Long-Term Drift Behavior of Electronic Nose (2025) | Zenodo 15681119 | 62 MOX, commercial e-nose, not Figaro-matched | diacetyl, 2-phenylethanol, ethanol | CC BY 4.0 | Conditional: adds 2 molecules but no sensor overlap |
| 8 | SmellNet (MIT, 2025) | GitHub MIT-MI/SmellNet | MQ-3, MQ-5, MQ-9, BME680, WSP2110, MP503, Grove V2 (12 ch) | 50 foods | CC BY 4.0 | Reject for the graph task: substances are natural products, not molecules. Keep as a citation for the cheap MQ-series framing |

## 3. Molecule union (sensor-compatible core: datasets 1+2+3)

| Molecule | SMILES | Heavy atoms | In 270 | In 251 | In 361 |
|---|---|---|---|---|---|
| Methane | C | 1 | | x | x |
| Ammonia | N | 1 | x | x | |
| Carbon monoxide | [C-]#[O+] | 2 | | x | x |
| Methanol | CO | 2 | | x | |
| Ethylene | C=C | 2 | x | x | x |
| Ethanol | CCO | 3 | x | | x |
| Acetaldehyde | CC=O | 3 | x | x | |
| Acetone | CC(C)=O | 4 | x | x | |
| 1-Butanol | CCCCO | 5 | | x | |
| Benzene | c1ccccc1 | 6 | | x | |
| Toluene | Cc1ccccc1 | 7 | x | x | |

**11 unique molecules.** Adding Zenodo #7: diacetyl (CC(=O)C(C)=O, 6 heavy atoms) and
2-phenylethanol (OCCc1ccccc1, 9) gives 13, at the cost of sensor incompatibility.

## 4. Sensor compatibility

Good news. Datasets 1, 2 and 3 all use Figaro TGS26xx sensors and come from the same
lab (Vergara / Fonollosa / Huerta, BioCircuits Institute, UCSD).

- Common to all three: TGS2610, TGS2602
- Common to 1 and 2: TGS2600, TGS2602, TGS2610, TGS2620
- Only in 2 and 3: TGS2611, TGS2612

Recommended harmonised output space: the 4-sensor-type vector
{TGS2600, TGS2602, TGS2610, TGS2620} from datasets 1 and 2, averaging over replicate
units of the same type. Dataset 3 can add TGS2611/TGS2612 as a partially observed
extension, or be held out as a sensor-replica robustness check.

Caveat that must go in the writeup: dataset 1 is a sealed 60 ml chamber, dataset 2 is
an open wind tunnel with turbulence and advection. Absolute response magnitudes are
not comparable across the two. Harmonisation has to be per-dataset standardisation of
the response vector, so the model learns the shape of the array response (the
cross-reactivity pattern) rather than absolute conductance. That is well aligned with
the project's question, but it does mean joint absolute concentration-response
regression across both datasets is off the table.

## 5. The blocking problem

Median molecule size in the core set is 3 heavy atoms. Five of the 11 molecules have
1 or 2 heavy atoms.

Consequences:

1. A GAT over a 1 to 3 node graph is a lookup table on element composition. There is
   no substructure for attention to select. Only benzene, toluene, 1-butanol and (if
   included) diacetyl and 2-phenylethanol have enough structure for an attention map
   to say anything.
2. The Phase 2 falsification test is close to guaranteed to fire. An ECFP+MLP baseline
   will match a GAT on 11 mostly-tiny molecules, because there is nothing that graph
   structure encodes here that a fingerprint does not.
3. The Phase 4 plausibility check ("does it attend to the hydroxyl on ethanol") is not
   really answerable. Ethanol is C-C-O. Every atom is the functional group.
4. The Phase 6 smell galaxy would have 11 points. That is a labelled scatterplot, not
   a galaxy.

This is a design problem, not a data-collection problem. Further dataset searching
does not fix it, because the MOX gas-sensor literature deliberately uses small
volatile calibration gases.

## 6. Options

**A. Proceed as specified, 11 molecules.**
Honest small-N feasibility study. Leave-molecule-out CV over 11 folds. Expect the
ECFP baseline to tie or win. Publication probability drops to near zero. The galaxy
view has to be dropped or replaced. Cheapest path, weakest artefact.

**B. Pivot the headline task to odor-descriptor prediction.**
Train on the OpenPOM curated GS-LF set (4,983 molecules, ~138 odor descriptors,
GoodScents + Leffingwell) or Pyrfume (>20,000 odorants, >40 archives). Real graphs,
real attention, real galaxy. But it drops the cheap-sensor angle, and it is close to
the tutorial-level project the spec was trying to avoid, since OpenPOM already exists.

**C. Two-stage: pretrain on odorants, transfer to the sensor array. Recommended.**
- Stage 1: train the GAT on OpenPOM GS-LF (4,983 molecules, multi-label odor
  descriptors). This gives a learned molecular embedding with enough data for
  attention to be meaningful and for the sanity and faithfulness tests to have power.
- Stage 2: freeze or fine-tune that encoder onto the 11-molecule MOX response
  regression, with leave-molecule-out CV as the headline number.
- The pitch stays intact: does structure learned from human odor perception transfer
  to what a cheap sensor actually measures, and where do the two disagree. That
  question is unexplored and is a stronger story than either half alone.
- Baselines still work: mean, ECFP+MLP, GCN, plus a from-scratch GAT with no
  pretraining, which directly measures whether transfer helps.
- The galaxy view gets 4,983 points coloured by odor descriptor, with the 11 sensor
  molecules highlighted as the ones with ground truth. Better payoff shot, and it uses
  the descriptor labels the spec already wanted for colouring.
- Cost: one extra training stage. Still fits a T4.

**D. Reframe to confusion-pair prediction.** Predict which molecule pairs the array
cannot separate, rather than the response vector. Same 11-molecule limit, so it does
not solve the core problem. Not recommended on its own.

## 7. Practical findings for Phase 1

- `ucimlrepo.fetch_ucirepo(id=270)` **mis-parses the dataset**. The raw format is
  libsvm-style `gas;concentration 1:v1 2:v2 ...`. ucimlrepo treats the label token as
  feature 1 and shifts every column by one, so both the gas label and the
  concentration are lost. Do not use it. Parse the raw `batch1..10.dat` files from the
  static zip instead. Verified parser: `parse_uci270.py`.
- Verified counts from the raw files (13,910 total):
  ethanol 2565 (2.5-600 ppmv), ethylene 2926 (2.5-300), ammonia 1641 (2.5-1000),
  acetaldehyde 1936 (2.5-300), acetone 3009 (10-1000), toluene 1833 (1-230).
  These minima are lower than the UCI landing page states. Trust the files.
- Dataset 251 is large: 18,000 measurements x 260 s x 100 Hz x 75 channels, roughly
  35 GB uncompressed. It needs a streaming feature-extraction pass, not a full load.
  Budget a session for it. Free Colab disk will not hold it uncompressed.
- Drift: Rodriguez-Lujan et al. 2022 (Sens. Actuators B, arXiv 2108.08793) show the
  batch structure of dataset 270 confounds gas identity with time, so any split must
  respect batches. The headline leave-molecule-out split mostly handles this, but the
  secondary random-split number must be batch-aware or it will be inflated.
- Pyrfume is CC BY-NC-ND 4.0. Fine for a portfolio project, awkward if a derived
  dataset is redistributed. The OpenPOM curated GS-LF CSV is the cleaner licence path.

## 8. Novelty check

A search for prior work on GNNs predicting MOX sensor array responses from molecular
structure returned nothing direct. The gas-sensor ML literature treats the analyte as
a categorical label, as the spec assumed. The molecular-graph literature predicts
human odor descriptors (OpenPOM, POM-Mix) but not sensor responses. The gap in the
spec is real. Option C sits directly in it.

## 9. Decision needed

Which option. Recommendation is C. It is the only one where the Phase 4
explainability work and the Phase 6 visualization can actually be executed as
specified, and it keeps the novel question rather than replacing it.

## 10. Sources

- UCI 270: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations
- UCI 251: http://archive.ics.uci.edu/ml/datasets/gas+sensor+arrays+in+open+sampling+settings
- UCI 361: https://archive.ics.uci.edu/ml/datasets/Twin+gas+sensor+arrays
- UCI 362: https://archive.ics.uci.edu/dataset/362/gas+sensors+for+home+activity+monitoring
- UCI 487: https://archive.ics.uci.edu/dataset/487/gas+sensor+array+temperature+modulation
- Wind tunnel data article: https://pmc.ncbi.nlm.nih.gov/articles/PMC4510097/
- Zenodo long-term drift: https://zenodo.org/records/15681119
- Drift limitations paper: https://arxiv.org/abs/2108.08793
- SmellNet: https://arxiv.org/html/2506.00239v1
- Pyrfume: https://pmc.ncbi.nlm.nih.gov/articles/PMC11557823/
- OpenPOM: https://github.com/ARY2260/openpom
