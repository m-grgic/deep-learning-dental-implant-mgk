# Intra-rater reliability — source files

Everything needed to reproduce the intra-observer reliability statistics reported in the
manuscript (Section 3.3), independently of the analysis notebook.

| file | what it is |
|---|---|
| `_annotations.coco.json` | second annotation round, COCO format, 122 radiographs / 154 implants |
| `po_implantatu_MBL.csv` | per-implant MBL for both rounds, the input to the R script |
| `bbox_iou.csv` | bounding-box IoU between rounds, per radiograph |
| `intra_rater_icc.R` | the analysis, `irr::icc(model = "twoway", type = "agreement")` |
| `intra_rater_icc_console.txt` | its console output, the source of the reported values |
| `README.roboflow.txt`, `README.dataset.txt` | original export metadata |

## Why this export specifically

The reported statistics were computed from the state of the intra-rater annotation project as
exported on **26 May 2026**. The project was edited afterwards: a later export shares only 58 of
its 154 implants with this one and does not reproduce the published values. Re-exporting the
project today therefore yields different annotations, so this archived copy is the reference.

The radiographs themselves are not redistributed here; only the annotations are, which is all
the reliability analysis needs.

## Reported values

Two-way, absolute agreement, single and average measures, n = 152 implants across 122
radiographs:

| measure | ICC(A,1) | 95% CI | ICC(A,2) | 95% CI |
|---|---|---|---|---|
| max MBL | 0.8336 | 0.7684–0.8802 | 0.9092 | 0.8684–0.9365 |
| MBL1 | 0.7394 | 0.6544–0.8054 | 0.8502 | 0.7905–0.8924 |
| MBL2 | 0.7838 | 0.7112–0.8394 | 0.8788 | 0.8309–0.9128 |

For max MBL: MAE 3.825 pp, RMSE 6.423 pp, Pearson r 0.8431, bias +1.743 pp (the first pass
reads higher), limits of agreement −10.415 to +13.900 pp. Mean bounding-box IoU between rounds
0.826 (SD 0.098).

MBL1 is the image-left side of the implant (keypoints IS1 and BL1) and MBL2 the image-right
side; the COCO category defines the keypoint order as AOI, BL2, IS2, IAC, IS1, BL1. The
keypoints carry no anatomical side information.

`IDJ_revision2_analysis.ipynb` in the parent directory reproduces every value in the table
above from `_annotations.coco.json` and the internal test-set labels.
