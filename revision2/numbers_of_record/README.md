# Numbers of record

`numbers_of_record.json` is the authoritative source for every quantity reported in the
manuscript. Where it disagrees with `../outputs/`, this file is correct: `../outputs/` is the
record of the revision-stage notebook run, and several quantities were recomputed afterwards
during copy-editing.

| file | what it is |
|---|---|
| `numbers_of_record.json` | every reported quantity at full precision |
| `M1_numbers.py` | the script that produces it |
| `M2_figures.py` | Figures 4 and 5 and Supplementary Figures S3–S6 |
| `icc_by_outcome.json` | radiograph-level ICC and design effect, by outcome |

## Conventions

**Intervals come from a cluster bootstrap over radiographs** — 2,000 replicates, seed 20260816,
with the IQR fences recomputed inside every replicate. 26 of the 122 test radiographs carry more
than one implant, so resampling implants independently would understate the intervals. The
non-clustered intervals are reported alongside as `ci_naive`; the difference is at most 14%.

Two exceptions use analytic intervals: the intra-observer ICCs in Section 3.3, which come from
`irr::icc` (see `../intra_rater/`), and the mAP intervals, which resample radiographs and
recompute average precision through Ultralytics `ap_per_class` (see the notebook, Section 5).

**OKS uses per-keypoint tolerance constants with k_i = 2·σ_i** and s² = 0.85 · w · h. The
constants are rounded to three decimals in `M1_numbers.py`, matching the manuscript text; the
unrounded values and their derivation are in `../sigma_derivation.py`. The difference is in the
fourth decimal of the mean OKS.

**The clustering ICC is the one-way random-effects estimator** (`icc1` in `M1_numbers.py`), with
the k₀ correction for unequal cluster sizes. On the absolute MBL error it gives 0.352, design
effect 1.086. The notebook reports 0.202 for the same outcome because it uses the REML
variance-component ratio from `statsmodels` `mixedlm`, which shrinks the between-radiograph
component under 26 informative clusters. Neither is wrong; the manuscript avoids the choice by
stating the directly verifiable result — clustered intervals are at most 14% wider.

**The localisation error is filtered by a 1.5 × IQR rule applied within each keypoint**, removing
71 of 912 measurements (7.8%). The unfiltered analysis is the primary one; the filtered figures
are reported for comparability with the previous round.

## Reproducing it

`M1_numbers.py` and `M2_figures.py` read intermediate tables — `per_instance.csv`,
`per_keypoint_errors.csv`, `matched_keypoints.csv`, `mbl_proj.csv`, `cluster_bootstrap.json` —
produced by `01_reproduce_internal_eval.py`, which are not yet deposited here. Until they are,
the scripts document the definitions and `numbers_of_record.json` carries the results.

The dataset path in `M1_numbers.py` was an absolute path on the machine it was written on;
it now resolves relative to the repository, or to `DATASET_ROOT` if that is set. Nothing else in
the deposited scripts was changed.

The archived test split names its label directory `test_labels/`, which is what `M1_numbers.py`
reads. Ultralytics expects `labels/` beside `images/` and, given `test_labels/`, reports
mAP 0.0000 without raising an error while writing an empty `labels.cache`.
