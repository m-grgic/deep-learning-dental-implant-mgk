# Revision 2 — canonical analysis

`IDJ_revision2_analysis.ipynb` is a single reproducible run that produces every quantitative
result reported in the manuscript, together with the additional analyses requested during peer
review. It supersedes the earlier notebook in the repository root for reporting purposes: that
notebook remains available as the record of the original analysis, but the values in the
manuscript are those produced here.

## Running it

Google Colab, top to bottom. The configuration cell expects the dataset, the model weights and
the archived intra-rater annotations under `MyDrive/Colab Notebooks`; every analysis decision is
set there and nothing else needs changing. Results are written to `revision2_outputs/` on Drive
and a copy of that run is committed here under `outputs/`.

Environment for the committed run: ultralytics 8.3.23, torch 2.11.0, Python 3.12, seed 42,
`model_02062025.pt`, `val()` at confidence 0.4 and IoU 0.7.

## What is in `outputs/`

| file | content |
|---|---|
| `results_manifest.json` | every reported quantity, with the configuration that produced it |
| `manuscript_crosscheck.csv` | each manuscript value against its recomputed counterpart |
| `per_implant_test.csv` | one row per ground-truth implant — the source of all statistics |
| `val_settings_matrix.csv` | mAP across model files and confidence thresholds |
| `map_bootstrap_ci.csv` | mAP point estimates with image-level bootstrap intervals |
| `table2_keypoint_error.csv` | per-keypoint localisation error, before and after outlier exclusion |
| `localisation_by_analysis_level.csv` | the same test at four units of analysis |
| `mbl_metrics.csv` | MBL accuracy under both geometric definitions and both subsets |
| `clustered_comparison.csv` | naive against cluster-aware confidence intervals |
| `sigma_intraobserver.csv` | empirically derived per-keypoint OKS tolerance constants |
| `intraobserver_agreement.csv`, `intraobserver_pairs.csv` | reliability of the reference standard |
| `figures/` | Figures 4 and 5 and Supplementary Figures S5 and S6, 300 dpi |

## Sections

1. Environment, configuration, seeds, and staging of the dataset to local disk
2. Dataset inventory and the distribution of implants per radiograph
3. Predictions and the per-implant table that every later section reads from
4. Detection: IoU, and mAP across model files and confidence thresholds
5. Confidence intervals for mAP by resampling radiographs and recomputing average precision
   with the same precision–recall integration as the primary evaluation
6. Keypoint localisation error, reported before and after outlier exclusion
7. OKS under a fixed tolerance constant and under empirically derived per-keypoint constants
8. Marginal bone loss under both geometric definitions
9. Clustered analysis treating implants as nested within radiographs
10. Intra-observer reliability, recomputed in Python from the archived annotations
11. Figures on a single percentage scale
12. Results manifest and the manuscript cross-check

## Notes on the analysis

**Marginal bone loss is computed by projection.** The shoulder-to-bone distance is projected
onto the longitudinal axis of the implant, as described in the Methods. An implant whose bone
level lies at or coronal to the shoulder is therefore assigned 0% loss, which applies to 31 of
the 152 implants in the internal test set. The earlier notebook measured the direct distance
between the two landmarks instead, which assigns a non-zero loss to every implant; both are
computed here so the difference is explicit.

**Confidence intervals for mAP describe the reported statistic.** Per-image match statistics are
captured from the Ultralytics validator and average precision is recomputed from them with
`ap_per_class`, the same function that produces the point estimate. The notebook asserts that
this recomputation reproduces `val()` before any interval is reported.

**Tolerance constants are derived from repeat annotations.** Following the COCO definition,
σ_k = √(E[d²/s²]) with s² the object area and the OKS exponent using k = 2σ, computed from the
two annotation rounds of the internal test set. These annotations were used only for evaluation
and played no part in model development or hyperparameter tuning. Results are reported under
both the derived constants and the fixed value used previously.

**Both filtered and unfiltered localisation results are reported**, at each unit of analysis,
with the unfiltered figures treated as the conservative estimate. One measurement in the test
set exceeds 50% of the bounding-box diagonal and is retained in the unfiltered analysis.

**IS1 and BL1 are the image-left keypoints**, IS2 and BL2 the image-right ones, matching the
keypoint order declared in the annotation export. They carry no anatomical side information.

## Reproducing the reliability analysis

`intra_rater/` holds the archived second annotation round, the R script that produced the
published reliability statistics and its console output. See the README there for why that
particular export is the reference copy.

## Files

- `IDJ_revision2_analysis.ipynb` — the notebook
- `IDJ_revision2_analysis.executed.ipynb` — the same notebook with the outputs of the committed run
- `build_nb.py` — generates the notebook; edit this rather than the `.ipynb` and regenerate
- `test_nb_math.py` — checks the geometry, OKS, sigma estimation and ICC implementations
  against real label files and, for the ICC, against the worked example in Shrout & Fleiss (1979)
