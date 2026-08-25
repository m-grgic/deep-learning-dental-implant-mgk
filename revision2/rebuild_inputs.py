#!/usr/bin/env python3
"""Rebuild the intermediate tables that numbers_of_record/M1_numbers.py reads.

M1_numbers.py was written against four CSVs and a JSON produced upstream by
01_reproduce_internal_eval.py, none of which are deposited here. All but one column of them are
recoverable from outputs/per_implant_test.csv, which this script demonstrates: it writes the
tables to numbers_of_record/rebuilt/ and recomputes M1's headline quantities from them, printing
each against numbers_of_record.json.

What is recoverable

    per_instance.csv         oks_fixed, iou, conf, detected
    mbl_proj.csv             true_max_proj / pred_max_proj, as fractions
    keypoint_distances.csv   per-keypoint separation in pixels, from sq_* x area; this stands in
                             for matched_keypoints.csv, whose coordinates M1 only ever uses to
                             compute that separation
    per_keypoint_errors.csv  error, as a fraction of the bounding-box diagonal

What is not

    per_keypoint_errors.csv:outlier. M1 reads this column, it does not compute it. Applying the
    1.5 x IQR rule within each keypoint to the errors above flags 69 measurements; the manuscript
    reports 71, and that is the count that reproduces the published Table 2.

    The gap is not the rule but one implant. 3-2-91 instance 0 is the only implant in the test set
    where the two ways of pairing a prediction with an annotation disagree: its image carries two
    predictions, box overlap selects the first and keypoint proximity the second, and this
    notebook uses box overlap. Under that choice its BL2 error is 16.74% of the diagonal against a
    fence of 17.64%, and its AOI error 3.62% against a fence of 3.80% - in both keypoints the
    closest value below the fence. Under the other choice both cross, which accounts for exactly
    the two extra exclusions. Removing that implant's BL2 measurement returns the reported 5.17
    and 3.89 exactly; removing its AOI measurement returns 1.39 and 0.78 against the reported 1.37
    and 0.76, so at least one further AOI measurement differs between the two runs.

    Which prediction 01_reproduce_internal_eval.py assigned to that implant is the open question.

    cluster_bootstrap.json. Only M2_figures.py reads it, for figure annotations; the intervals
    themselves are in numbers_of_record.json under ci_cluster.

Run:  python3 rebuild_inputs.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "outputs" / "per_implant_test.csv"
NOR = HERE / "numbers_of_record"
DST = NOR / "rebuilt"

KPTS = ["apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"]
LABEL = {"apex": "AOI", "desni_kost": "BL2", "desni_vrh": "IS2",
         "vrh": "IAC", "lijevi_vrh": "IS1", "lijevi_kost": "BL1"}
ORDER = ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"]
SIGMA = {"AOI": 0.029, "IAC": 0.073, "IS1": 0.221,
         "IS2": 0.216, "BL1": 0.237, "BL2": 0.243}   # as rounded in M1_numbers.py


def mean(x):
    return sum(x) / len(x)


def sd(x, ddof=1):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def median(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2


def quantile(x, p):
    x = sorted(x)
    i = (len(x) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return x[lo] + (x[hi] - x[lo]) * (i - lo)


def write(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def icc_one_way(values, groups):
    """One-way random-effects ICC with the k0 correction, as in M1_numbers.py."""
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    N, n, gm = len(values), len(by), mean(values)
    msb = sum(len(v) * (mean(v) - gm) ** 2 for v in by.values()) / (n - 1)
    msw = sum(sum((x - mean(v)) ** 2 for x in v) for v in by.values()) / (N - n)
    k0 = (N - sum(len(v) ** 2 for v in by.values()) / N) / (n - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), N / n


rows = list(csv.DictReader(open(SRC)))
num = lambda r, c: float(r[c])
DST.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ rebuilt tables
per_instance, mbl_proj, distances, per_kpt = [], [], [], []
for r in rows:
    stem, idx, area = r["image"].rsplit(".", 1)[0], int(r["instance_idx"]), num(r, "area")
    per_instance.append({"filename": stem, "instance_idx": idx,
                         "oks": r["oks_fixed"], "iou": r["iou"], "conf": r["conf"],
                         "detected": r["detected"]})
    mbl_proj.append({"filename": stem, "instance_idx": idx,
                     "tmax": num(r, "true_max_proj") / 100,
                     "pmax": num(r, "pred_max_proj") / 100})
    for n in KPTS:
        d_px = math.sqrt(num(r, "sq_" + n) * area)
        distances.append({"filename": stem, "instance_idx": idx, "keypoint": LABEL[n],
                          "d_px": f"{d_px:.10g}", "area_px": f"{area:.10g}"})
        per_kpt.append({"filename": stem, "instance_idx": idx,
                        "uid": f"{stem}#{idx}", "keypoint": LABEL[n],
                        "error": num(r, "d_" + n), "outlier": None})

# outlier flags, recomputed here; see the module docstring
by_kpt = defaultdict(list)
for row in per_kpt:
    by_kpt[row["keypoint"]].append(row)
for label, group in by_kpt.items():
    err = [g["error"] for g in group]
    q1, q3 = quantile(err, 25), quantile(err, 75)
    fence = q3 + 1.5 * (q3 - q1)
    for g in group:
        g["outlier"] = g["error"] > fence

write(DST / "per_instance.csv", ["filename", "instance_idx", "oks", "iou", "conf", "detected"],
      per_instance)
write(DST / "mbl_proj.csv", ["filename", "instance_idx", "tmax", "pmax"], mbl_proj)
write(DST / "keypoint_distances.csv", ["filename", "instance_idx", "keypoint", "d_px", "area_px"],
      distances)
write(DST / "per_keypoint_errors.csv",
      ["filename", "instance_idx", "uid", "keypoint", "error", "outlier"], per_kpt)

# ------------------------------------------------------------------ verification
record = json.load(open(NOR / "numbers_of_record.json"))
checks = []


def check(name, got, want, tol):
    checks.append((name, got, want, abs(got - want) <= tol))


oks_new, oks_old = [], []
for r in rows:
    area = num(r, "area")
    new, old = [], []
    for n in KPTS:
        d2 = num(r, "sq_" + n) * area
        new.append(math.exp(-d2 / (2 * area * (2 * SIGMA[LABEL[n]]) ** 2)))
        old.append(math.exp(-d2 / (2 * area * (2 * 0.075) ** 2)))
    oks_new.append(mean(new))
    oks_old.append(mean(old))

check("OKS, fixed sigma", mean(oks_old), record["oks_old"]["mean"], 1e-4)
check("OKS, per-keypoint sigma", mean(oks_new), record["oks_new"]["mean"], 1e-3)
check("OKS median, per-keypoint", median(oks_new), record["oks_new"]["median"], 1e-4)

err_all = [g["error"] for g in per_kpt]
ku = record["kp_error"]["measurement_unfiltered"]
check("localisation error, mean", mean(err_all), ku["mean"], 1e-4)
check("localisation error, SD", sd(err_all), ku["sd"], 1e-4)
check("localisation error, median", median(err_all), ku["median"], 1e-4)

diff = [m["pmax"] - m["tmax"] for m in mbl_proj]
check("MBL MAE", mean([abs(d) for d in diff]), record["mbl"]["mae"], 1e-4)
check("MBL RMSE", math.sqrt(mean([d * d for d in diff])), record["mbl"]["rmse"], 1e-4)
check("MBL bias", mean(diff), record["mbl"]["bias"], 1e-4)
check("MBL zero-reference implants", sum(1 for m in mbl_proj if m["tmax"] == 0),
      record["mbl"]["n_expert_zero"], 0)

icc, mbar = icc_one_way([abs(d) for d in diff], [m["filename"] for m in mbl_proj])
check("clustering ICC", icc, record["clustering"]["icc_abs_error"], 1e-3)
check("design effect", 1 + (mbar - 1) * icc, record["clustering"]["design_effect"], 1e-3)

n_out = sum(1 for g in per_kpt if g["outlier"])
check("outliers excluded", n_out, record["kp_error"]["n_outliers"], 0)

print(f"rebuilt from {SRC.name}: {len(rows)} implants, {len(per_kpt)} measurements -> {DST}\n")
print(f"{'quantity':<30}{'rebuilt':>14}{'record':>14}   status")
for name, got, want, ok in checks:
    print(f"{name:<30}{got:>14.6g}{want:>14.6g}   {'ok' if ok else 'DIFFERS'}")

print("\nper-keypoint exclusions")
print(f"{'kpt':<6}{'rebuilt':>9}{'record':>8}   {'filtered mean %':>16}{'record':>9}")
for label in ORDER:
    group = by_kpt[label]
    kept = [g["error"] for g in group if not g["outlier"]]
    ref = record["table2"][label]
    print(f"{label:<6}{sum(1 for g in group if g['outlier']):>9}{ref['n_excluded']:>8}"
          f"{100 * mean(kept):>16.2f}{ref['fil_mean']:>9.2f}")

ambiguous = [r for r in rows if r["match_agrees"] != "True"]
multi = sum(1 for r in rows if int(r["n_pred"]) > 1)
print(f"\n{multi} of {len(rows)} implants come from an image carrying more than one prediction; "
      f"in {len(ambiguous)} the two matching rules disagree")
for r in ambiguous:
    stem = r["image"].rsplit(".", 1)[0]
    print(f"  {stem[:24]} instance {r['instance_idx']}: {r['n_pred']} predictions, "
          f"box overlap picks {r['match_iou_idx']}, keypoint proximity picks {r['match_kpt_idx']}")
    for label in ORDER:
        name = next(k for k, v in LABEL.items() if v == label)
        err = [g["error"] for g in by_kpt[label]]
        q1, q3 = quantile(err, 25), quantile(err, 75)
        value = float(r["d_" + name])
        print(f"    {label:<4} {100 * value:7.2f}%   fence {100 * (q3 + 1.5 * (q3 - q1)):7.2f}%"
              f"   {'excluded' if value > q3 + 1.5 * (q3 - q1) else 'kept'}")

if not all(ok for _, _, _, ok in checks):
    print("\nThe only expected difference is the outlier count; see the module docstring.")
