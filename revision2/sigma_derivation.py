#!/usr/bin/env python3
"""Standalone derivation of the per-keypoint OKS tolerance constants reported in Section 2.5.2.

Reproduces the six sigma values from repository files only, without the notebook:

    AOI 0.0291   IAC 0.0733   IS1 0.2210   IS2 0.2159   BL1 0.2366   BL2 0.2434

Inputs
    intra_rater/_annotations.coco.json                          second annotation round
    ../Keypoint_detection.v10-512px-adaptive.yolov8/test/       first annotation round (YOLO)

Definition
    sigma_k = sqrt( mean_over_pairs( d_k^2 / s^2 ) )

    d_k   Euclidean distance in pixels between the two rounds for keypoint k
    s^2   0.85 * w_box * h_box in pixels, averaged over the two rounds' boxes
    mean  arithmetic mean over all 152 matched implants, no outlier exclusion
    OKS   uses k_k = 2 * sigma_k

Run:  python3 sigma_derivation.py
"""

import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Keypoint_detection.v10-512px-adaptive.yolov8"
COCO = HERE / "intra_rater" / "_annotations.coco.json"

IMG_SIZE = 512
AREA_FACTOR = 0.85
IOU_MIN = 0.10

# Slot order in both the YOLO labels and the COCO category.
KPTS = ["apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"]
LABEL = {"apex": "AOI", "vrh": "IAC", "lijevi_vrh": "IS1",
         "desni_vrh": "IS2", "lijevi_kost": "BL1", "desni_kost": "BL2"}
REPORTED = {"AOI": 0.0291, "IAC": 0.0733, "IS1": 0.2210,
            "IS2": 0.2159, "BL1": 0.2366, "BL2": 0.2434}


def coco_to_yolo_rows(path):
    """COCO annotations -> {image stem: [row, ...]} in YOLO keypoint format."""
    coco = json.load(open(path))
    names = {i["id"]: i.get("extra", {}).get("name", i["file_name"]).replace(".jpg", "")
             for i in coco["images"]}
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    per_image = {}
    for a in coco["annotations"]:
        w, h = sizes[a["image_id"]]
        bx, by, bw, bh = a["bbox"]
        kp = np.array(a["keypoints"], dtype=float).reshape(-1, 3)
        row = [0, (bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h]
        for x, y, v in kp:
            row += [x / w, y / h, int(v)]
        per_image.setdefault(names[a["image_id"]], []).append(np.array(row, dtype=float))
    return per_image


def load_label(path):
    data = np.loadtxt(path)
    return np.atleast_2d(data)


def bbox_iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def match_annotation_pair(rows_a, rows_b, w, h, iou_min=IOU_MIN):
    """Greedy bounding-box-overlap matching of implants between two annotation rounds."""
    if len(rows_a) == 0 or len(rows_b) == 0:
        return []
    scale = np.array([w, h, w, h])
    M = np.array([[bbox_iou_xywh(ra[1:5] * scale, rb[1:5] * scale) for rb in rows_b]
                  for ra in rows_a])
    used, pairs = set(), []
    for i in np.argsort(-M.max(axis=1)):
        avail = [M[i, j] if j not in used else -1.0 for j in range(len(rows_b))]
        j = int(np.argmax(avail))
        if avail[j] <= iou_min:
            continue
        used.add(j)
        pairs.append((rows_a[i], rows_b[j]))
    return pairs


def kps_from_row(row, w, h):
    kp = np.asarray(row[5:], dtype=float).reshape(-1, 3)
    return {n: (kp[i, 0] * w, kp[i, 1] * h) for i, n in enumerate(KPTS)}


def sigma_from_pairs(pairs, w=IMG_SIZE, h=IMG_SIZE, area_factor=AREA_FACTOR,
                     area_rule="mean", estimator="rms"):
    sq = {n: [] for n in KPTS}
    for ra, rb in pairs:
        ka, kb = kps_from_row(ra, w, h), kps_from_row(rb, w, h)
        aa = area_factor * ra[3] * w * ra[4] * h
        ab = area_factor * rb[3] * w * rb[4] * h
        area = {"mean": 0.5 * (aa + ab), "first": aa, "second": ab}[area_rule]
        for n in KPTS:
            d = float(np.hypot(ka[n][0] - kb[n][0], ka[n][1] - kb[n][1]))
            sq[n].append(d ** 2 / area)
    out = {}
    for n in KPTS:
        v = np.array(sq[n])
        out[LABEL[n]] = float(np.sqrt(v.mean())) if estimator == "rms" else float(
            np.sqrt(v).std(ddof=1))
    return out


def stem_key(name):
    """Strip the Roboflow export suffix: "1-1-51_jpg.rf.<hash>.txt" -> "1-1-51"."""
    return str(name).split("_jpg")[0].split(".rf.")[0].rsplit(".", 1)[0]


def build_pairs():
    reannot = {stem_key(k): v for k, v in coco_to_yolo_rows(COCO).items()}
    pairs, matched_images = [], 0
    for lbl in sorted(glob.glob(str(BASE / "test" / "labels" / "*.txt"))):
        stem = stem_key(Path(lbl).name)
        if stem not in reannot:
            continue
        rows_a = load_label(lbl)
        rows_b = np.array(reannot[stem])
        p = match_annotation_pair(list(rows_a), list(rows_b), IMG_SIZE, IMG_SIZE)
        if p:
            matched_images += 1
        pairs += p
    return pairs, matched_images


if __name__ == "__main__":
    if not COCO.is_file():
        raise SystemExit(f"missing {COCO}")
    if not (BASE / "test" / "labels").is_dir():
        raise SystemExit(
            f"missing {BASE / 'test' / 'labels'} — the archived export ships the first round as "
            f"test_labels/; rename it to labels/ next to test/images/")

    pairs, n_img = build_pairs()
    print(f"matched {len(pairs)} implants across {n_img} radiographs\n")

    sig = sigma_from_pairs(pairs)
    print("as reported (area = 0.85 * w * h, averaged over the two rounds, RMS estimator)")
    print(f"{'kpt':<5}{'this run':>10}{'paper':>10}{'diff':>9}")
    for k in ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"]:
        print(f"{k:<5}{sig[k]:>10.4f}{REPORTED[k]:>10.4f}{sig[k] - REPORTED[k]:>9.4f}")

    print("\nsensitivity to the three conventions that are easy to get wrong")
    variants = {
        "no 0.85 area factor": dict(area_factor=1.0),
        "area from first round only": dict(area_rule="first"),
        "SD of d/s instead of RMS": dict(estimator="sd"),
    }
    hdr = f"{'variant':<28}" + "".join(f"{k:>8}" for k in
                                       ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"])
    print(hdr)
    for name, kw in variants.items():
        s = sigma_from_pairs(pairs, **kw)
        print(f"{name:<28}" + "".join(f"{s[k]:>8.4f}" for k in
                                      ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"]))
