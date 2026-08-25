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

`M1_numbers.py` and `M2_figures.py` read intermediate tables produced upstream by
`01_reproduce_internal_eval.py`, which is not deposited here. `../rebuild_inputs.py` reconstructs
those tables into `rebuilt/` from `../outputs/per_implant_test.csv` and checks the result against
`numbers_of_record.json`: every quantity agrees to at least four significant figures, including
the fixed-sigma OKS, the localisation error, the MBL statistics and the clustering ICC.

One column does not survive the reconstruction. `M1_numbers.py` reads the `outlier` flag from
`per_keypoint_errors.csv` rather than computing it. Applying the 1.5 × IQR rule within each
keypoint to the reconstructed errors flags 69 measurements; the manuscript reports 71, and 71 is
the count that reproduces the published Table 2.

The gap is one implant, not the rule. `3-2-91` instance 0 is the only implant in the test set
where the two ways of pairing a prediction with an annotation disagree: its image carries two
predictions, box overlap selects the first and keypoint proximity the second, and the notebook
uses box overlap. Under that choice its BL2 error is 16.74% of the bounding-box diagonal against
a fence of 17.64%, and its AOI error 3.62% against a fence of 3.80% — in both keypoints the
closest value below the fence. Under the other choice both cross, which is exactly the two extra
exclusions. Dropping its BL2 measurement returns the reported 5.17 and 3.89 exactly; dropping its
AOI measurement returns 1.39 and 0.78 against the reported 1.37 and 0.76, so at least one further
AOI measurement differs between the two runs. Sixty-eight of the 152 implants come from an image
carrying more than one prediction, so the configuration is not unusual — this is the only one
where the rules part company.

`cluster_bootstrap.json` is read only by `M2_figures.py`, for figure annotations; the intervals
themselves are in `numbers_of_record.json` under `ci_cluster`. `matched_keypoints.csv` is not
needed at all: `M1_numbers.py` uses its coordinates only to compute the per-keypoint separation,
which `rebuilt/keypoint_distances.csv` carries directly.

The dataset path in `M1_numbers.py` was an absolute path on the machine it was written on;
it now resolves relative to the repository, or to `DATASET_ROOT` if that is set. Nothing else in
the deposited scripts was changed.

The archived test split names its label directory `test_labels/`, which is what `M1_numbers.py`
reads. Ultralytics expects `labels/` beside `images/` and, given `test_labels/`, reports
mAP 0.0000 without raising an error while writing an empty `labels.cache`.
