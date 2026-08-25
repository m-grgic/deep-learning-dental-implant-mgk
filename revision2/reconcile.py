#!/usr/bin/env python3
"""Compare the notebook run in outputs/ against the manuscript's numbers of record.

Writes outputs/manuscript_crosscheck.csv, one row per reported quantity, and prints a summary.
The manuscript column is read from numbers_of_record/, never typed in here.

Run:  python3 reconcile.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOR = HERE / "numbers_of_record"
OUT = HERE / "outputs"

record = json.load(open(NOR / "numbers_of_record.json"))
supp = json.load(open(NOR / "supplement.json"))
manifest = json.load(open(OUT / "results_manifest.json"))["results"]

PCT = 100.0  # numbers_of_record stores localisation and MBL as fractions


def nested(path):
    node = record
    for key in path:
        node = node[key]
    return node


# (manifest key, manuscript value, source tag)
ROWS = [
    ("n_images_test", record["n_radiographs"], "record"),
    ("n_implants_test", record["n_implants"], "record"),

    ("iou_sd", supp["iou_sd"]["value"], "supplement"),
    ("iou_ci", supp["iou_ci"]["value"], "supplement"),
    ("box_mAP50_95", supp["box_mAP50_95"]["value"], "supplement"),
    ("box_mAP50_95_ci", supp["box_mAP50_95_ci"]["value"], "supplement"),
    ("pose_mAP50_95", supp["pose_mAP50_95"]["value"], "supplement"),
    ("pose_mAP50_95_ci", supp["pose_mAP50_95_ci"]["value"], "supplement"),

    ("oks_fixed_mean", round(nested(["oks_old", "mean"]), 4), "record"),
    ("oks_fixed_median", round(nested(["oks_old", "median"]), 4), "record"),
    ("oks_wilcoxon_stat", nested(["oks_old", "wilcoxon"])[0], "record"),
    ("oks_empirical_mean", round(nested(["oks_new", "mean"]), 4), "record"),
    ("oks_empirical_ci", [round(nested(["ci_cluster", "oks_new_mean", b]), 4)
                          for b in ("lo", "hi")], "record"),

    ("kpt_err_unfiltered_n", nested(["kp_error", "n_measurements"]), "record"),
    ("kpt_err_n_outliers", nested(["kp_error", "n_outliers"]), "record"),
    ("kpt_err_unfiltered_mean",
     round(PCT * nested(["kp_error", "measurement_unfiltered", "mean"]), 2), "record"),
    ("kpt_err_unfiltered_sd",
     round(PCT * nested(["kp_error", "measurement_unfiltered", "sd"]), 2), "record"),
    ("kpt_err_filtered_mean",
     round(PCT * nested(["kp_error", "measurement_filtered", "mean"]), 2), "record"),
    ("kpt_err_filtered_sd",
     round(PCT * nested(["kp_error", "measurement_filtered", "sd"]), 2), "record"),
    ("kpt_err_unfiltered_ci", [round(PCT * nested(["ci_cluster", "err_unf", b]), 2)
                               for b in ("lo", "hi")], "record"),
    ("kpt_err_filtered_ci", [round(PCT * nested(["ci_cluster", "err_fil", b]), 2)
                             for b in ("lo", "hi")], "record"),

    ("mbl_MAE", round(PCT * nested(["mbl", "mae"]), 2), "record"),
    ("mbl_MAE_ci", [round(PCT * nested(["ci_cluster", "mbl_mae", b]), 2)
                    for b in ("lo", "hi")], "record"),
    ("mbl_RMSE", round(PCT * nested(["mbl", "rmse"]), 2), "record"),
    ("mbl_RMSE_ci", [round(PCT * nested(["ci_cluster", "mbl_rmse", b]), 2)
                     for b in ("lo", "hi")], "record"),
    ("mbl_r", round(nested(["mbl", "pearson_r"]), 4), "record"),
    ("mbl_r_ci", [round(nested(["ci_cluster", "mbl_r", b]), 4)
                  for b in ("lo", "hi")], "record"),
    ("mbl_wilcoxon_stat", nested(["mbl", "wilcoxon_mae5"])[0], "record"),
    ("mbl_ba_bias", round(PCT * nested(["mbl", "bias"]), 2), "record"),
    ("mbl_ba_loa_low", round(PCT * nested(["mbl", "loa_lo"]), 2), "record"),
    ("mbl_ba_loa_high", round(PCT * nested(["mbl", "loa_hi"]), 2), "record"),
    ("mbl_zero_reference_proj", nested(["mbl", "n_expert_zero"]), "record"),
    ("regression_slope", round(nested(["mbl", "slope"]), 2), "record"),
    ("regression_intercept_pct", round(nested(["mbl", "intercept"]), 2), "record"),
    ("regression_r2", round(nested(["mbl", "r2"]), 2), "record"),
]

TOLERANCE = {  # rounding-level differences that are not disagreements
    "oks_fixed_mean": 0.0001, "oks_fixed_median": 0.0001,
    "mbl_MAE": 0.01, "mbl_RMSE": 0.01, "mbl_r": 0.0001,
    "mbl_ba_bias": 0.01, "mbl_ba_loa_low": 0.01, "mbl_ba_loa_high": 0.01,
    "regression_slope": 0.01, "regression_intercept_pct": 0.01, "regression_r2": 0.01,
    "kpt_err_unfiltered_mean": 0.01, "kpt_err_unfiltered_sd": 0.01,
}


def classify(key, manuscript, repo):
    if repo is None:
        return "absent from outputs/"
    if isinstance(manuscript, list) or isinstance(repo, list):
        return "match" if manuscript == repo else "differs"
    tol = TOLERANCE.get(key, 0.0)
    return "match" if abs(float(manuscript) - float(repo)) <= tol + 1e-12 else "differs"


def fmt(v):
    return "" if v is None else (f"{v[0]}-{v[1]}" if isinstance(v, list) else str(v))


rows = []
for key, manuscript, source in ROWS:
    repo = manifest.get(key, {}).get("value")
    rows.append({"quantity": key, "manuscript": fmt(manuscript), "outputs": fmt(repo),
                 "status": classify(key, manuscript, repo), "manuscript_source": source})

with open(OUT / "manuscript_crosscheck.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["quantity", "manuscript", "outputs", "status",
                                       "manuscript_source"])
    w.writeheader()
    w.writerows(rows)

n_diff = sum(r["status"] != "match" for r in rows)
print(f"{len(rows)} quantities, {len(rows) - n_diff} match, {n_diff} differ\n")
for r in rows:
    if r["status"] != "match":
        print(f"  {r['quantity']:26s} manuscript {r['manuscript']:>18s}   "
              f"outputs {r['outputs']:>18s}")
print(f"\nwritten: {OUT / 'manuscript_crosscheck.csv'}")
