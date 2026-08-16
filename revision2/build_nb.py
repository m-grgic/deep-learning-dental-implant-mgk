#!/usr/bin/env python3
"""Build the canonical revision-2 analysis notebook for the IDJ manuscript.

Regenerate with:  python3 build_nb.py
Output:           <repo>/revision2/IDJ_revision2_analysis.ipynb
"""
import json
import os

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").split("\n")})


def code(src):
    lines = src.strip("\n").split("\n")
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    })


# =====================================================================
md(r"""
# IDJ revision 2 — canonical analysis notebook

**Purpose.** One reproducible run that produces *every* number reported in the manuscript,
plus the new analyses requested by the Editor and Reviewer 2 in the second review round.
Earlier notebooks (`DoraGalic_diplomski.ipynb`, `DoraGalic-diplomski-bu.ipynb.bak`) disagree
with each other and with the manuscript; this notebook replaces both as the single source of truth.

**How to run.** Google Colab (GPU runtime recommended, CPU works). Run all cells top to bottom.
Every reported quantity is collected into `RESULTS` and written to `outputs/results_manifest.json`;
the final section prints a manuscript-vs-recomputed comparison table.

| § | Content | Addresses |
|---|---------|-----------|
| 1 | Environment, config, seeds | reproducibility |
| 2 | Dataset inventory + cluster structure | Editor #6 |
| 3 | Predictions → per-implant table (single source for all downstream stats) | — |
| 4 | Detection: IoU, Ultralytics `val()` settings matrix | traces the mAP@50-95 = 0.7860 discrepancy |
| 5 | mAP confidence intervals by image-level bootstrap with PR integration | Editor #8, Reviewer 2 #2 |
| 6 | Keypoint localisation error — unfiltered **and** filtered, equal prominence | Editor #5 |
| 7 | OKS + empirical per-keypoint sigma derivation | Editor #10 |
| 8 | MBL quantification (both geometric definitions) | Editor #5, item 5 |
| 9 | Clustered / robust analysis, implants nested in radiographs | Editor #6 |
| 10 | Intra-observer reliability (ICC) recomputed in Python | Editor #4 |
| 11 | Figures 4, 5, S5, S6 on a single percentage scale | Reviewer 2 #3 |
| 12 | Results manifest + manuscript cross-check | item 5 |

**Inputs that must be supplied before §7 and §10 can run:** the second (re-)annotation of the
internal test set — 152 implants / 122 radiographs — in YOLO label format. Set `REANNOT_LABELS_DIR`
in the config cell. Every other section runs from data already in the repository.
""")

# ---------------------------------------------------------------------
md("## 1. Environment and configuration")

code(r"""
# Pinned to the version used for the results reported in the manuscript.
# Change ULTRALYTICS_VERSION only deliberately: mAP values differ measurably between versions.
ULTRALYTICS_VERSION = "8.3.23"

import importlib, subprocess, sys

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

try:
    import ultralytics
    assert ultralytics.__version__ == ULTRALYTICS_VERSION
except (ImportError, AssertionError):
    _pip(f"ultralytics=={ULTRALYTICS_VERSION}")
    importlib.invalidate_caches()

for _mod, _pkg in [("statsmodels", "statsmodels"), ("pingouin", "pingouin"), ("shapely", "shapely")]:
    try:
        importlib.import_module(_mod)
    except ImportError:
        _pip(_pkg)

IN_COLAB = "google.colab" in sys.modules or importlib.util.find_spec("google.colab") is not None
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
""")

code(r"""
import os, glob, json, math, random, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import ultralytics
from scipy import stats
from ultralytics import YOLO

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("ultralytics", ultralytics.__version__)
print("torch      ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy      ", np.__version__)
print("python     ", sys.version.split()[0])
""")

code(r"""
# ---------------------------------------------------------------- paths
DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks" if IN_COLAB else "."

BASE      = f"{DRIVE_ROOT}/Keypoint_detection.v10-512px-adaptive.yolov8"   # internal dataset
VANJSKA   = f"{DRIVE_ROOT}/vanjska"                                        # 3-annotator external set
OUT_DIR   = Path(f"{DRIVE_ROOT}/revision2_outputs")
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Second annotation of the internal test set.
#
# Use the COCO export of 26 May 2026, which is the state of the intra-rater project that the
# published reliability analysis was computed from. The project has been edited since: a YOLO
# export taken on 22 June 2026 shares only 58 of 154 implants with it and does not reproduce the
# published ICC. Re-exporting the project today therefore gives the wrong annotations, so the
# archived COCO file is the reference copy and travels with the paper.
#
# COCO keypoint order is AOI, BL2, IS2, IAC, IS1, BL1, which matches the slot order used here.
REANNOT_COCO_JSON = f"{DRIVE_ROOT}/intra_rater_coco/train/_annotations.coco.json"
REANNOT_LABELS_DIR = None   # filled in by the conversion cell below, or set directly to YOLO labels

MODEL_FILE  = "model_02062025.pt"                       # primary model
MODEL_ALT   = "model_31052025.pt"                       # checked in the §4 settings matrix
MODEL_PATH  = f"{BASE}/{MODEL_FILE}"
DATA_YAML   = f"{BASE}/data.yaml"

# ------------------------------------------------------- analysis knobs
IMG_SIZE     = 512
AREA_FACTOR  = 0.85          # COCO-style area = 0.85 * w * h, as in the original code
SIGMA_FIXED  = 0.075         # single tolerance constant used for all six keypoints so far
KPTS         = ["apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"]
# Manuscript Table 2 reproduces only under this assignment: IS1/BL1 are the image-LEFT points and
# IS2/BL2 the image-right ones (their IS1 = 2.01 matches lijevi_vrh, their BL1 = 4.68 matches
# lijevi_kost, to two decimals). Note this is the reverse of the label map in utils.py, which calls
# desni_* "Distal" and lijevi_* "Mesial"; the keypoints carry no anatomical side information.
KPT_LABEL    = {"apex": "AOI", "vrh": "IAC",
                "lijevi_vrh": "IS1", "desni_vrh": "IS2",
                "lijevi_kost": "BL1", "desni_kost": "BL2"}

# Inference settings for the per-implant table.
# NOTE: the original code passed conf=0.1 into a helper that discarded the argument, so the
# analysis effectively ran at the Ultralytics defaults below. Kept for comparability.
PRED_CONF = 0.25
PRED_IOU  = 0.70

# Ultralytics val() settings. conf=0.4 is what produced the values reported in the manuscript
# (0.7867 / 0.9332 vs the printed 0.7860 / 0.9328); conf=0.001 is the Ultralytics default and
# gives 0.7667 / 0.9212. Held at 0.4 so the reported point estimates and their confidence
# intervals describe the same statistic. The §4 matrix evaluates both either way.
VAL_CONF  = 0.4
VAL_IOU   = 0.70

APEX_OUTLIER_CUTOFF = 0.10   # legacy per-instance exclusion rule (scaled apex displacement)
N_BOOT      = 2000           # bootstrap iterations for CIs
N_BOOT_MAP  = 1000           # bootstrap iterations for mAP (each one re-integrates PR curves)

RESULTS = {}                 # every reported number lands here
def record(key, value, note=""):
    RESULTS[key] = {"value": value, "note": note}
    return value

assert os.path.isdir(BASE), f"dataset not found: {BASE}"
assert os.path.isfile(MODEL_PATH), f"model not found: {MODEL_PATH}"
print("BASE   ", BASE)
print("MODEL  ", MODEL_PATH)
print("OUTPUT ", OUT_DIR)
print("re-annotation labels:", REANNOT_LABELS_DIR or "NOT SUPPLIED — §7 sigma and §10 ICC will be skipped")
""")

md(r"""
### Staging the dataset on local disk

Google Drive is mounted over FUSE, so every file access is a network round trip. This notebook
touches the 1,214 label files once, all 122 test images once per `predict()` pass, and the whole
test split four more times in the `val()` matrix — thousands of round trips. Copying the dataset to
the Colab VM's local disk in a single sequential pass turns that from tens of minutes into seconds.

Outputs still go to Drive, so nothing is lost when the runtime is recycled. Set `STAGE_LOCAL = False`
to read directly from Drive.
""")

code(r"""
STAGE_LOCAL = True

if IN_COLAB and STAGE_LOCAL:
    import subprocess, time

    def _tar_copy(src, dest_parent):
        '''One tar stream instead of one FUSE round trip per file.'''
        parent, name = os.path.dirname(src.rstrip("/")), os.path.basename(src.rstrip("/"))
        cmd = (f'tar cf - -C "{parent}" --exclude="*.pt" --exclude="vizuali" "{name}" '
               f'| tar xf - -C "{dest_parent}"')
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    os.makedirs("/content/data", exist_ok=True)
    LOCAL_BASE = "/content/data/" + os.path.basename(BASE)
    marker = Path(LOCAL_BASE) / ".staged"

    if not marker.exists():
        t0 = time.time()
        _tar_copy(BASE, "/content/data")
        for mf in {MODEL_FILE, MODEL_ALT}:                 # weights are excluded from the tar
            src = f"{BASE}/{mf}"
            if os.path.isfile(src):
                subprocess.run(["cp", src, f"{LOCAL_BASE}/{mf}"], check=True)
        marker.touch()
        print(f"dataset staged in {time.time() - t0:.0f} s")
    else:
        print("dataset already staged")

    BASE = LOCAL_BASE
    DATA_YAML = f"{BASE}/data.yaml"
    MODEL_PATH = f"{BASE}/{MODEL_FILE}"

    if os.path.isdir(VANJSKA):
        local_v = "/content/data/" + os.path.basename(VANJSKA.rstrip("/"))
        if not os.path.isdir(local_v):
            _tar_copy(VANJSKA, "/content/data")
        VANJSKA = local_v

    if REANNOT_LABELS_DIR and os.path.isdir(REANNOT_LABELS_DIR):
        local_r = "/content/data/" + os.path.basename(REANNOT_LABELS_DIR.rstrip("/"))
        if not os.path.isdir(local_r):
            _tar_copy(REANNOT_LABELS_DIR, "/content/data")
        REANNOT_LABELS_DIR = local_r

print("BASE      ", BASE)
print("MODEL_PATH", MODEL_PATH)
print("VANJSKA   ", VANJSKA)
print("OUT_DIR   ", OUT_DIR, "(stays on Drive)")
assert os.path.isfile(MODEL_PATH), MODEL_PATH
""")

code(r"""
# Convert the archived COCO re-annotation to YOLO label files so the rest of the notebook can
# read it. COCO keypoint order (AOI, BL2, IS2, IAC, IS1, BL1) matches the slot order used here.
if REANNOT_COCO_JSON and os.path.isfile(REANNOT_COCO_JSON):
    dest = "/content/data/reannot_labels" if IN_COLAB else "./reannot_labels"
    os.makedirs(dest, exist_ok=True)
    coco = json.load(open(REANNOT_COCO_JSON))
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
        per_image.setdefault(names[a["image_id"]], []).append(row)
    for stem, rows in per_image.items():
        with open(f"{dest}/{stem}.txt", "w") as fh:
            fh.write("\n".join(" ".join(f"{c:.10g}" for c in r) for r in rows))
    REANNOT_LABELS_DIR = dest
    print(f"COCO re-annotation converted: {len(per_image)} images, "
          f"{sum(len(v) for v in per_image.values())} implants -> {dest}")
else:
    print("REANNOT_COCO_JSON not found; sections 7 and 10 will use REANNOT_LABELS_DIR if set")
""")

# ---------------------------------------------------------------------
md(r"""
## 2. Dataset inventory and cluster structure

Reproduces the split counts quoted in the manuscript and quantifies how many implants share a
radiograph — the input to the clustered analysis in §9.
""")

code(r"""
def split_inventory(split):
    lbl = sorted(glob.glob(f"{BASE}/{split}/labels/*.txt"))
    per_image = {}
    for f in lbl:
        rows = [l for l in open(f).read().strip().split("\n") if l.strip()]
        per_image[Path(f).stem] = len(rows)
    return per_image

inventory = {s: split_inventory(s) for s in ["train", "valid", "test"]}
rows = []
for s, d in inventory.items():
    rows.append({"split": s, "images": len(d), "implants": sum(d.values())})
inv_df = pd.DataFrame(rows)
inv_df.loc[len(inv_df)] = {"split": "TOTAL", "images": inv_df.images.sum(),
                           "implants": inv_df.implants.sum()}
print(inv_df.to_string(index=False))

for s in ["train", "valid", "test"]:
    record(f"n_images_{s}", len(inventory[s]))
    record(f"n_implants_{s}", sum(inventory[s].values()))
record("n_images_total", int(inv_df.images.iloc[-1]))
record("n_implants_total", int(inv_df.implants.iloc[-1]))
""")

code(r"""
test_counts = pd.Series(inventory["test"]).value_counts().sort_index()
print("Implants per radiograph (internal test set):")
for k, v in test_counts.items():
    print(f"  {k} implant(s): {v} images")

n_img = len(inventory["test"]); n_imp = sum(inventory["test"].values())
mean_cluster = n_imp / n_img
multi_imgs = int(sum(v for k, v in test_counts.items() if k > 1))
multi_imps = int(sum(k * v for k, v in test_counts.items() if k > 1))
print(f"\nmean cluster size            : {mean_cluster:.3f}")
print(f"images with >1 implant       : {multi_imgs} / {n_img}")
print(f"implants in such images      : {multi_imps} / {n_imp} ({100*multi_imps/n_imp:.1f}%)")
print(f"design effect at ICC=0.5     : {1 + (mean_cluster - 1) * 0.5:.3f}")

record("test_mean_cluster_size", round(mean_cluster, 4))
record("test_images_multi_implant", multi_imgs)
record("test_implants_in_multi_images", multi_imps)
""")

# ---------------------------------------------------------------------
md(r"""
## 3. Geometry, predictions, and the per-implant table

All downstream statistics read from `INST`, one row per ground-truth implant in the internal test
set. Formulas are copied verbatim from `utils.py` so results stay comparable with earlier runs.

Two MBL definitions are computed side by side, because the previous notebooks disagree:

* **`proj`** — bone-level and shoulder points are projected onto the implant's longitudinal axis
  before measuring (`getTwoSides` / `vectorProjection`). This is the method the manuscript's
  Methods section describes ("geometric projection ... onto the implant's longitudinal axis").
* **`eucl`** — plain Euclidean distance between shoulder and bone-level point, no projection.
  This is what the currently published notebook computes.
""")

code(r"""
# ----------------------------------------------------------- geometry (verbatim from utils.py)
def _pt(x, y):
    return {"x": float(x), "y": float(y)}

def getApexVrh(kp, w, h):
    apex = _pt(kp[0] * w, kp[1] * h)
    vrh  = _pt(kp[9] * w, kp[10] * h)
    return apex, vrh, {"i": apex["x"] - vrh["x"], "j": apex["y"] - vrh["y"]}

def getRightSide(kp, w, h):
    dd = _pt(kp[3] * w, kp[4] * h)
    dg = _pt(kp[6] * w, kp[7] * h)
    return dd, dg, {"i": dd["x"] - dg["x"], "j": dd["y"] - dg["y"]}

def getLeftSide(kp, w, h):
    lg = _pt(kp[12] * w, kp[13] * h)
    ld = _pt(kp[15] * w, kp[16] * h)
    return lg, ld, {"i": ld["x"] - lg["x"], "j": ld["y"] - lg["y"]}

def pointProjection(P, V, vec):
    npV, npP = np.array([V["x"], V["y"]]), np.array([P["x"], P["y"]])
    npv = np.array([vec["i"], vec["j"]])
    proj = npV + (np.dot(npP - npV, npv) / np.linalg.norm(npv) ** 2) * npv
    return _pt(proj[0], proj[1])

def vectorProjection(hvatiste, vrh, V, vec):
    ph, pv = pointProjection(hvatiste, V, vec), pointProjection(vrh, V, vec)
    HV = {"i": pv["x"] - ph["x"], "j": pv["y"] - ph["y"]}
    d = 0.0
    if HV["i"] * vec["i"] + HV["j"] * vec["j"] > 0:
        d = float(np.hypot(HV["i"], HV["j"]))
    return d

def getTwoSides(desni_donji, desni_gornji, lijevi_donji, lijevi_gornji, V, vec):
    return {
        "srednji": float(np.hypot(vec["i"], vec["j"])),
        "desni":  vectorProjection(desni_gornji, desni_donji, V, vec),
        "lijevi": vectorProjection(lijevi_gornji, lijevi_donji, V, vec),
    }

def _euclid(a, b):
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))

def mbl_sides(kps, mode):
    '''Return {'srednji','desni','lijevi'} in pixels for one keypoint set.'''
    vec = {"i": kps["apex"]["x"] - kps["vrh"]["x"], "j": kps["apex"]["y"] - kps["vrh"]["y"]}
    if mode == "proj":
        return getTwoSides(kps["desni_kost"], kps["desni_vrh"],
                           kps["lijevi_kost"], kps["lijevi_vrh"], kps["vrh"], vec)
    return {
        "srednji": _euclid(kps["vrh"], kps["apex"]),
        "desni":   _euclid(kps["desni_vrh"], kps["desni_kost"]),
        "lijevi":  _euclid(kps["lijevi_vrh"], kps["lijevi_kost"]),
    }

def bbox_iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def OKS(kp_true, kp_pred, sigma, area):
    k = 2 * np.asarray(sigma, dtype=float)
    d = np.array([_euclid(kp_true[n], kp_pred[n]) for n in KPTS])
    return float(np.mean(np.exp(-(d ** 2) / (2 * area * k ** 2))))
""")

code(r"""
def kps_from_row(row, w, h):
    apex, vrh, _ = getApexVrh(row[5:], w, h)
    dd, dg, _ = getRightSide(row[5:], w, h)
    lg, ld, _ = getLeftSide(row[5:], w, h)
    return {"apex": apex, "desni_kost": dd, "desni_vrh": dg,
            "vrh": vrh, "lijevi_vrh": lg, "lijevi_kost": ld}

def kps_from_pred(arr):
    return {n: _pt(arr[i][0], arr[i][1]) for i, n in enumerate(KPTS)}

def load_label(path):
    data = np.loadtxt(path)
    if data.ndim != 2:
        data = np.array([data])
    return data

def build_instance_table(model, split="test", base=BASE, label_dir=None):
    '''One row per ground-truth implant, with its matched prediction.'''
    img_paths = sorted(glob.glob(f"{base}/{split}/images/*.*"))
    lbl_dir = label_dir or f"{base}/{split}/labels"
    out = []
    for img_path in img_paths:
        stem = Path(img_path).stem
        lbl_path = f"{lbl_dir}/{stem}.txt"
        if not os.path.isfile(lbl_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        res = model.predict(img_path, conf=PRED_CONF, iou=PRED_IOU, verbose=False)[0]
        pred_boxes = res.boxes.xywh.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        pred_conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        pred_kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((0, 6, 3))

        for idx, row in enumerate(load_label(lbl_path)):
            gt_box = row[1:5] * np.array([w, h, w, h])
            box_w, box_h = float(row[3] * w), float(row[4] * h)
            area = AREA_FACTOR * box_w * box_h
            diag = float(np.hypot(box_w, box_h))
            gt_kps = kps_from_row(row, w, h)

            rec = {"image": Path(img_path).name, "instance_idx": idx,
                   "box_w": box_w, "box_h": box_h, "area": area, "diag": diag,
                   "n_pred": len(pred_boxes), "detected": len(pred_boxes) > 0}

            if len(pred_boxes) == 0:
                out.append(rec)
                continue

            ious = np.array([bbox_iou_xywh(gt_box, pb) for pb in pred_boxes])
            j_iou = int(np.argmax(ious))
            # legacy matching rule: prediction with the smallest mean scaled keypoint distance
            means = []
            for pk in pred_kpts:
                pk_d = kps_from_pred(pk)
                means.append(np.mean([_euclid(gt_kps[n], pk_d[n]) for n in KPTS]) / diag)
            j_kpt = int(np.argmin(means))

            rec.update({"iou": float(ious[j_iou]), "conf": float(pred_conf[j_iou]),
                        "match_iou_idx": j_iou, "match_kpt_idx": j_kpt,
                        "match_agrees": j_iou == j_kpt})

            pk = kps_from_pred(pred_kpts[j_kpt])
            for n in KPTS:
                d = _euclid(gt_kps[n], pk[n])
                rec[f"d_{n}"] = d / diag                 # scaled by bbox diagonal (Table 2)
                rec[f"sq_{n}"] = d ** 2 / area           # area-normalised, for sigma estimation
            rec["d_mean"] = float(np.mean([rec[f"d_{n}"] for n in KPTS]))
            rec["oks_fixed"] = OKS(gt_kps, pk, [SIGMA_FIXED] * 6, area)

            for mode in ("proj", "eucl"):
                t, p = mbl_sides(gt_kps, mode), mbl_sides(pk, mode)
                for side in ("desni", "lijevi"):
                    tv = t[side] / t["srednji"] * 100 if t["srednji"] else np.nan
                    pv = p[side] / p["srednji"] * 100 if p["srednji"] else np.nan
                    rec[f"true_{side}_{mode}"] = tv
                    rec[f"pred_{side}_{mode}"] = pv
                rec[f"true_max_{mode}"] = np.nanmax([rec[f"true_desni_{mode}"], rec[f"true_lijevi_{mode}"]])
                rec[f"pred_max_{mode}"] = np.nanmax([rec[f"pred_desni_{mode}"], rec[f"pred_lijevi_{mode}"]])
            out.append(rec)
    return pd.DataFrame(out)

pose_model = YOLO(MODEL_PATH)
INST = build_instance_table(pose_model, "test")
INST["apex_outlier"] = INST["d_apex"] > APEX_OUTLIER_CUTOFF
INST.to_csv(OUT_DIR / "per_implant_test.csv", index=False)

print(f"rows (ground-truth implants): {len(INST)}")
print(f"detected                    : {int(INST.detected.sum())}")
print(f"IoU-match == keypoint-match : {int(INST.match_agrees.sum())} / {int(INST.detected.sum())}")
print(f"apex displacement > {APEX_OUTLIER_CUTOFF}    : {int(INST.apex_outlier.sum())}")
INST.head(3)
""")

# ---------------------------------------------------------------------
md(r"""
## 4. Detection performance and the mAP settings matrix

The manuscript reports mAP@50-95 (BBox) = 0.7860 and (Pose) = 0.9328. Neither archived notebook
reproduces those values, so this cell runs `val()` across model files and confidence settings to
identify which configuration, if any, produced them.
""")

code(r"""
iou_vals = INST.loc[INST.detected, "iou"].to_numpy()

def boot_ci(x, fn=np.mean, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    b = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

lo, hi = boot_ci(iou_vals)
print(f"mean IoU   {iou_vals.mean():.4f}  (95% CI {lo:.4f}-{hi:.4f})")
print(f"median     {np.median(iou_vals):.4f}")
print(f"SD         {iou_vals.std(ddof=0):.4f}")
print(f"n          {len(iou_vals)}")

record("iou_mean", round(float(iou_vals.mean()), 4))
record("iou_ci", [round(lo, 4), round(hi, 4)])
record("iou_median", round(float(np.median(iou_vals)), 4))
record("iou_sd", round(float(iou_vals.std(ddof=0)), 4))

w = stats.wilcoxon(iou_vals - 0.8, alternative="greater")
print(f"\nWilcoxon signed-rank vs 0.8: statistic = {w.statistic:.1f}, p = {w.pvalue:.4g}")
print("  (tests the median / distribution of paired differences, NOT the arithmetic mean)")
record("iou_wilcoxon_stat", float(w.statistic))
record("iou_wilcoxon_p", float(w.pvalue))
""")

code(r"""
val_rows = []
for model_file in [MODEL_FILE, MODEL_ALT]:
    mp = f"{BASE}/{model_file}"
    if not os.path.isfile(mp):
        print(f"skip {model_file} (not found)")
        continue
    for conf in [0.001, 0.4]:
        m = YOLO(mp)
        r = m.val(data=DATA_YAML, split="test", conf=conf, iou=VAL_IOU,
                  plots=False, verbose=False)
        val_rows.append({
            "model": model_file, "conf": conf,
            "box_mAP50": round(float(r.box.map50), 4), "box_mAP50_95": round(float(r.box.map), 4),
            "pose_mAP50": round(float(r.pose.map50), 4), "pose_mAP50_95": round(float(r.pose.map), 4),
            "box_P": round(float(r.box.mp), 4), "box_R": round(float(r.box.mr), 4),
        })
VAL_MATRIX = pd.DataFrame(val_rows)
VAL_MATRIX.to_csv(OUT_DIR / "val_settings_matrix.csv", index=False)
print(VAL_MATRIX.to_string(index=False))
print("\nManuscript claims: box mAP@50=0.9933, mAP@50-95=0.7860 | pose mAP@50=0.9933, mAP@50-95=0.9328")
""")

code(r"""
# Primary configuration for everything reported downstream.
primary = VAL_MATRIX[(VAL_MATRIX.model == MODEL_FILE) & (VAL_MATRIX.conf == VAL_CONF)].iloc[0]
for k in ["box_mAP50", "box_mAP50_95", "pose_mAP50", "pose_mAP50_95"]:
    record(k, float(primary[k]), f"ultralytics {ultralytics.__version__}, conf={VAL_CONF}")
print(primary.to_string())
""")

# ---------------------------------------------------------------------
md(r"""
## 5. Confidence intervals for mAP

**Editor comment 8 / Reviewer 2 comment 2.** The previous bootstrap approximated AP as the
proportion of instances with IoU above each threshold, which ignores confidence ranking, false
positives, false negatives and precision-recall integration. That approach is removed.

The procedure here resamples **radiographs** (the sampling unit) with replacement and, for each
resample, recomputes average precision with Ultralytics' own `ap_per_class` — the same
precision-recall integration that produced the reported point estimates. Per-image match matrices
are captured directly from the validator, so no part of the metric is re-implemented.

The validation printed below is the point the Editor asked for: recomputing AP from the captured
per-image statistics over the full test set must reproduce `val()` exactly.
""")

code(r"""
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import ap_per_class

def capture_per_image_stats(model_path, data_yaml, split="test", conf=VAL_CONF, iou=VAL_IOU):
    '''Run val() with batch=1 and capture the per-image (tp, tp_p, conf, pred_cls, target_cls).'''
    captured = []
    original = PoseValidator.update_metrics

    def hooked(self, preds, batch):
        before = {k: len(v) for k, v in self.stats.items()}
        original(self, preds, batch)
        captured.append({k: [np.asarray(t.cpu() if hasattr(t, "cpu") else t)
                             for t in v[before[k]:]] for k, v in self.stats.items()})

    PoseValidator.update_metrics = hooked
    try:
        res = YOLO(model_path).val(data=data_yaml, split=split, conf=conf, iou=iou,
                                   batch=1, plots=False, verbose=False)
    finally:
        PoseValidator.update_metrics = original
    return res, captured

VAL_RES, PER_IMAGE = capture_per_image_stats(MODEL_PATH, DATA_YAML)
print(f"captured {len(PER_IMAGE)} images; keys: {sorted(PER_IMAGE[0].keys())}")
""")

code(r"""
def _cat(chunks, key):
    arrs = [a for c in chunks for a in c.get(key, []) if np.asarray(a).size or True]
    arrs = [np.atleast_1d(np.asarray(a)) for a in arrs]
    return np.concatenate(arrs) if arrs else np.zeros(0)

def _cat2d(chunks, key, ncol):
    arrs = [np.asarray(a).reshape(-1, ncol) for c in chunks for a in c.get(key, [])]
    return np.concatenate(arrs) if arrs else np.zeros((0, ncol))

NIOU = 10

def map_from_chunks(chunks, tp_key):
    tp = _cat2d(chunks, tp_key, NIOU)
    conf = _cat(chunks, "conf")
    pred_cls = _cat(chunks, "pred_cls")
    target_cls = _cat(chunks, "target_cls")
    if tp.shape[0] == 0 or target_cls.size == 0:
        return np.nan, np.nan
    order = np.argsort(-conf)
    res = ap_per_class(tp[order], conf[order], pred_cls[order], target_cls, plot=False)
    ap = next(a for a in res if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == NIOU)
    return float(ap[:, 0].mean()), float(ap.mean())

check = {
    "box":  map_from_chunks(PER_IMAGE, "tp"),
    "pose": map_from_chunks(PER_IMAGE, "tp_p"),
}
print("Validation — recomputed from captured per-image statistics vs Ultralytics val():")
print(f"  box  mAP@50    {check['box'][0]:.4f}   val(): {VAL_RES.box.map50:.4f}")
print(f"  box  mAP@50-95 {check['box'][1]:.4f}   val(): {VAL_RES.box.map:.4f}")
print(f"  pose mAP@50    {check['pose'][0]:.4f}   val(): {VAL_RES.pose.map50:.4f}")
print(f"  pose mAP@50-95 {check['pose'][1]:.4f}   val(): {VAL_RES.pose.map:.4f}")

max_dev = max(abs(check["box"][0] - VAL_RES.box.map50), abs(check["box"][1] - VAL_RES.box.map),
              abs(check["pose"][0] - VAL_RES.pose.map50), abs(check["pose"][1] - VAL_RES.pose.map))
print(f"\nmax absolute deviation: {max_dev:.8f}")
assert max_dev < 1e-4, (
    "The per-image statistics do not reproduce val(). The bootstrap below would not be a "
    "confidence interval for the reported statistic — fix this before reporting any CI.")
record("map_recomputation_max_deviation", float(max_dev),
       "bootstrap machinery reproduces Ultralytics val() exactly")
""")

code(r"""
rng = np.random.default_rng(SEED)
n_img = len(PER_IMAGE)
boot = {"box_mAP50": [], "box_mAP50_95": [], "pose_mAP50": [], "pose_mAP50_95": []}

for _ in range(N_BOOT_MAP):
    idx = rng.integers(0, n_img, n_img)
    chunks = [PER_IMAGE[i] for i in idx]
    b50, b5095 = map_from_chunks(chunks, "tp")
    p50, p5095 = map_from_chunks(chunks, "tp_p")
    boot["box_mAP50"].append(b50);   boot["box_mAP50_95"].append(b5095)
    boot["pose_mAP50"].append(p50);  boot["pose_mAP50_95"].append(p5095)

point = {"box_mAP50": check["box"][0], "box_mAP50_95": check["box"][1],
         "pose_mAP50": check["pose"][0], "pose_mAP50_95": check["pose"][1]}

rows = []
for k, v in boot.items():
    v = np.asarray(v, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    rows.append({"metric": k, "point_estimate": round(point[k], 4),
                 "bootstrap_mean": round(float(v.mean()), 4),
                 "bootstrap_median": round(float(np.median(v)), 4),
                 "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                 "abs_diff_point_vs_mean": round(abs(point[k] - float(v.mean())), 4)})
MAP_CI = pd.DataFrame(rows)
MAP_CI.to_csv(OUT_DIR / "map_bootstrap_ci.csv", index=False)
print(MAP_CI.to_string(index=False))
print(f"\n{N_BOOT_MAP} image-level resamples, seed {SEED}, PR integration via ultralytics.ap_per_class")

for r in rows:
    record(f"{r['metric']}_ci", [r["ci_low"], r["ci_high"]],
           f"image-level bootstrap, {N_BOOT_MAP} iterations")
""")

# ---------------------------------------------------------------------
md(r"""
## 6. Keypoint localisation error — unfiltered and filtered

**Editor comment 5.** Both analyses are reported with equal prominence; the unfiltered result is
the primary, more conservative estimate because it retains gross localisation failures.

The IQR rule is applied per keypoint over all `6 × n` measurements, matching the manuscript's
"n = 912 keypoint measurements, 71 outliers removed (7.8%)".
""")

code(r"""
det = INST[INST.detected].copy()
long = det.melt(id_vars=["image", "instance_idx"],
                value_vars=[f"d_{n}" for n in KPTS],
                var_name="kpt", value_name="err")
long["kpt"] = long["kpt"].str.replace("d_", "", regex=False)
long["label"] = long["kpt"].map(KPT_LABEL)
long["err_pct"] = long["err"] * 100

def iqr_cutoff(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 + 1.5 * (q3 - q1)

# Two IQR variants: pooled over all measurements, and applied within each keypoint. The manuscript's
# "71 outliers (7.8%)" corresponds to the per-keypoint variant, so both are reported.
long["outlier_pooled"] = long["err_pct"] > iqr_cutoff(long["err_pct"])
long["outlier_perkp"] = long.groupby("label")["err_pct"].transform(
    lambda s: s > iqr_cutoff(s))

kept = long[~long.outlier_perkp]          # filtered set used downstream
kept_pooled = long[~long.outlier_pooled]

print(f"unfiltered            : n = {len(long)}, mean = {long.err_pct.mean():.2f}%, "
      f"SD = {long.err_pct.std(ddof=0):.2f}%, median = {long.err_pct.median():.2f}%")
print(f"IQR per keypoint      : {int(long.outlier_perkp.sum())} removed "
      f"({100*long.outlier_perkp.mean():.1f}%) -> mean {kept.err_pct.mean():.2f}%, "
      f"SD {kept.err_pct.std(ddof=0):.2f}%")
print(f"IQR pooled            : {int(long.outlier_pooled.sum())} removed "
      f"({100*long.outlier_pooled.mean():.1f}%) -> mean {kept_pooled.err_pct.mean():.2f}%, "
      f"SD {kept_pooled.err_pct.std(ddof=0):.2f}%")
print("manuscript            : unfiltered 3.64% / SD 4.24% (n = 912); "
      "71 removed (7.8%); filtered 2.98% / SD 1.47%")

record("kpt_err_unfiltered_n", int(len(long)))
record("kpt_err_unfiltered_mean", round(float(long.err_pct.mean()), 2))
record("kpt_err_unfiltered_sd", round(float(long.err_pct.std(ddof=0)), 2))
record("kpt_err_n_outliers", int(long.outlier_perkp.sum()), "IQR applied within each keypoint")
record("kpt_err_filtered_mean", round(float(kept.err_pct.mean()), 2))
record("kpt_err_filtered_sd", round(float(kept.err_pct.std(ddof=0)), 2))
lo, hi = boot_ci(kept.err_pct.to_numpy())
record("kpt_err_filtered_ci", [round(lo, 2), round(hi, 2)])
lo_u, hi_u = boot_ci(long.err_pct.to_numpy())
record("kpt_err_unfiltered_ci", [round(lo_u, 2), round(hi_u, 2)])
print(f"\n95% CI (bootstrap): unfiltered {lo_u:.2f}-{hi_u:.2f}%   filtered {lo:.2f}-{hi:.2f}%")
print("  A normal-theory CI is not appropriate here: the unfiltered distribution is dominated by "
      "its tail.")
""")

code(r"""
# Gross localisation failures — what the unfiltered analysis is meant to expose.
gross = long[long.err_pct > 50].sort_values("err_pct", ascending=False)
print(f"measurements with error > 50% of the bounding-box diagonal: {len(gross)}")
for _, g in gross.iterrows():
    print(f"  {g['image'][:38]:38s} inst {g['instance_idx']}  {g['label']}  {g.err_pct:.1f}%")

if len(gross):
    without = long.drop(gross.index)
    print(f"\nexcluding those {len(gross)} measurement(s): mean {without.err_pct.mean():.2f}%, "
          f"SD {without.err_pct.std(ddof=0):.2f}%  (n = {len(without)})")
    print("  The manuscript reports 3.64% / SD 4.24% at n = 912, i.e. the published 'unfiltered'")
    print("  figure is reproduced only once this failure is removed while n is still quoted as 912.")
record("kpt_err_n_gross_failures", int(len(gross)))
""")

code(r"""
ORDER = ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"]
TABLE2 = pd.DataFrame({
    "n": long.groupby("label")["err_pct"].count(),
    "mean_unfiltered": long.groupby("label")["err_pct"].mean(),
    "sd_unfiltered": long.groupby("label")["err_pct"].std(),
    "median_unfiltered": long.groupby("label")["err_pct"].median(),
    "n_removed": long.groupby("label")["outlier_perkp"].sum(),
    "mean_filtered": kept.groupby("label")["err_pct"].mean(),
    "sd_filtered": kept.groupby("label")["err_pct"].std(),
}).reindex(ORDER).round(2)
TABLE2.to_csv(OUT_DIR / "table2_keypoint_error.csv")
print(TABLE2.to_string())
print("\nmanuscript Table 2 (filtered only): AOI 1.37, IAC 1.76, IS1 2.01, IS2 2.21, "
      "BL1 4.68, BL2 5.17")
record("table2", TABLE2.reset_index().to_dict("records"))
""")

code(r"""
# Analysis level matters and the manuscript mixes two. Table 2 is per measurement (6 per implant),
# while the published test statistic of 4960.0 with p = 0.0581 is only attainable at n = 152:
# E[W] = n(n+1)/4 = 5814 and SD = 544, so p = 0.058 implies W = 4960 exactly at n = 152, and 4305
# at n = 142. The construction that reproduces it is the per-implant mean of the measurements
# surviving the per-keypoint IQR rule. All levels are reported, each labelled with its unit.
per_implant = det.groupby(["image", "instance_idx"])["d_mean"].first() * 100
per_implant_kept = per_implant[per_implant <= iqr_cutoff(per_implant)]
per_implant_surviving = kept.groupby(["image", "instance_idx"])["err_pct"].mean()

levels = {
    "measurement, unfiltered": long.err_pct,
    "measurement, filtered": kept.err_pct,
    "per implant, unfiltered": per_implant,
    "per implant, filtered": per_implant_kept,
    "per implant, mean of surviving (manuscript)": per_implant_surviving,
}
rows = []
for tag, s in levels.items():
    w = stats.wilcoxon(s - 3.0, alternative="less")
    rows.append({"level": tag, "n": len(s), "mean": round(float(s.mean()), 2),
                 "median": round(float(s.median()), 2), "sd": round(float(s.std(ddof=0)), 2),
                 "wilcoxon_stat": round(float(w.statistic), 1),
                 "p_one_sided": float(f"{w.pvalue:.4g}")})
LEVELS = pd.DataFrame(rows)
LEVELS.to_csv(OUT_DIR / "localisation_by_analysis_level.csv", index=False)
print(LEVELS.to_string(index=False))
print("\nmanuscript: statistic = 4960.0, p = 0.0581")
print("\nThe test evaluates the median of the differences from 3%, not the arithmetic mean:")
print(f"  unfiltered mean is {long.err_pct.mean():.2f}% (above the 3% threshold) while the median "
      f"is {long.err_pct.median():.2f}%,")
print("  so a significant result here must not be worded as evidence that the mean lies below 3%.")

for tag, s in levels.items():
    key = tag.replace(", ", "_").replace(" ", "_")
    w = stats.wilcoxon(s - 3.0, alternative="less")
    record(f"kpt_wilcoxon_{key}_stat", float(w.statistic))
    record(f"kpt_wilcoxon_{key}_p", float(w.pvalue))
record("kpt_wilcoxon_unfiltered_stat", float(stats.wilcoxon(long.err_pct - 3.0, alternative="less").statistic))
record("kpt_wilcoxon_unfiltered_p", float(stats.wilcoxon(long.err_pct - 3.0, alternative="less").pvalue))
""")

# ---------------------------------------------------------------------
md(r"""
## 7. OKS and the per-keypoint tolerance constants

**Editor comment 10.** The published code applies a single fixed `sigma = 0.075` to all six
keypoints; it is not keypoint-specific and was not derived from annotation variability in this
dataset. This section (a) reports OKS under that fixed constant, and (b) derives keypoint-specific
constants empirically, following the COCO definition.

COCO derives the tolerance from redundant annotations of the same instances:

  sigma_k = sqrt( mean( d_k^2 / s^2 ) ),  s^2 = object area,  and OKS uses k_k = 2 * sigma_k

which is exactly the quantity the unused `PointDistanceForSigma` helper computes. Two sources are
supported: the intra-observer re-annotation of the internal test set (primary; requires
`REANNOT_LABELS_DIR`), and the three-annotator external set (secondary, sensitivity only).
""")

code(r"""
oks_fixed = det["oks_fixed"].to_numpy()
lo, hi = boot_ci(oks_fixed)
print(f"mean OKS (sigma = {SIGMA_FIXED}) : {oks_fixed.mean():.4f}  (95% CI {lo:.4f}-{hi:.4f})")
print(f"median                    : {np.median(oks_fixed):.4f}")
print(f"SD                        : {oks_fixed.std(ddof=0):.4f}")
print(f"n                         : {len(oks_fixed)}")

record("oks_fixed_mean", round(float(oks_fixed.mean()), 4))
record("oks_fixed_ci", [round(lo, 4), round(hi, 4)])
record("oks_fixed_median", round(float(np.median(oks_fixed)), 4))
record("oks_fixed_sd", round(float(oks_fixed.std(ddof=0)), 4))

w = stats.wilcoxon(oks_fixed - 0.8, alternative="greater")
print(f"\nWilcoxon vs 0.8: statistic = {w.statistic:.1f}, p = {w.pvalue:.4g}")
record("oks_wilcoxon_stat", float(w.statistic))
record("oks_wilcoxon_p", float(w.pvalue))
""")

code(r"""
def match_annotation_pair(rows_a, rows_b, w, h, iou_min=0.10):
    '''Match implants between two annotations of the same image by bounding-box overlap.

    Greedy assignment in descending order of each implant's best available overlap. Overlap is
    used rather than centre distance because two annotators can place near-concentric boxes of
    quite different extent on adjacent implants; on the internal test set this pairs 152/152
    implants, against 144 for centre-distance matching.
    '''
    if len(rows_a) == 0 or len(rows_b) == 0:
        return []
    M = np.array([[bbox_iou_xywh(ra[1:5] * np.array([w, h, w, h]),
                                 rb[1:5] * np.array([w, h, w, h]))
                   for rb in rows_b] for ra in rows_a])
    used, pairs = set(), []
    for i in np.argsort(-M.max(axis=1)):
        avail = [M[i, j] if j not in used else -1.0 for j in range(len(rows_b))]
        j = int(np.argmax(avail))
        if avail[j] <= iou_min:
            continue
        used.add(j)
        pairs.append((rows_a[i], rows_b[j]))
    return pairs

def sigma_from_pairs(pairs, w=IMG_SIZE, h=IMG_SIZE):
    sq = {n: [] for n in KPTS}
    for ra, rb in pairs:
        ka, kb = kps_from_row(ra, w, h), kps_from_row(rb, w, h)
        area = 0.5 * (AREA_FACTOR * ra[3] * w * ra[4] * h + AREA_FACTOR * rb[3] * w * rb[4] * h)
        for n in KPTS:
            sq[n].append(_euclid(ka[n], kb[n]) ** 2 / area)
    rows = []
    for n in KPTS:
        v = np.array(sq[n])
        trimmed = np.sort(v)[: int(len(v) * 0.95)]
        rows.append({"keypoint": KPT_LABEL[n], "internal": n, "n": len(v),
                     "sigma": round(float(np.sqrt(v.mean())), 4),
                     "sigma_trim95": round(float(np.sqrt(trimmed.mean())), 4),
                     "sigma_per_annotator": round(float(np.sqrt(v.mean() / 2)), 4)})
    return pd.DataFrame(rows)
""")

code(r"""
import hashlib

def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

def _pixel_md5(path):
    im = cv2.imread(path)
    return None if im is None else hashlib.md5(np.ascontiguousarray(im)).hexdigest()

def _stem_key(path):
    '''Strip the Roboflow export suffix: "001_jpg.rf.<hash>.txt" -> "001".'''
    return Path(path).name.split("_jpg")[0].split(".rf.")[0].rsplit(".", 1)[0]

def reannotation_file_pairs(reannot_dir):
    '''Pair each internal test label file with its re-annotation counterpart.

    A Roboflow re-export renames files, so basenames cannot be assumed to match. Strategies are
    tried per image in order: identical basename, identical image bytes, identical decoded
    pixels, then the leading stem before the Roboflow hash.
    '''
    def label_beside(img):
        for c in (Path(img).parent.parent / "labels" / (Path(img).stem + ".txt"),
                  Path(img).with_suffix(".txt")):
            if os.path.isfile(c):
                return str(c)
        return None

    r_lbls = {Path(p).name: p for p in glob.glob(f"{reannot_dir}/**/*.txt", recursive=True)}
    r_imgs = [p for e in ("jpg", "jpeg", "png")
              for p in glob.glob(f"{reannot_dir}/**/*.{e}", recursive=True)]
    by_md5, by_pix, by_stem = {}, {}, {}
    for p in r_imgs:
        lbl = label_beside(p)
        if lbl:
            by_md5.setdefault(_md5(p), lbl)
            ph = _pixel_md5(p)
            if ph:
                by_pix.setdefault(ph, lbl)
    for p in r_lbls.values():
        by_stem.setdefault(_stem_key(p), p)

    pairs, how, unmatched = [], {}, []
    for img in sorted(glob.glob(f"{BASE}/test/images/*.*")):
        stem = Path(img).stem
        t_lbl = f"{BASE}/test/labels/{stem}.txt"
        if not os.path.isfile(t_lbl):
            continue
        hit, via = None, None
        if stem + ".txt" in r_lbls:
            hit, via = r_lbls[stem + ".txt"], "basename"
        if hit is None and by_md5:
            hit, via = by_md5.get(_md5(img)), "image bytes"
        if hit is None and by_pix:
            hit, via = by_pix.get(_pixel_md5(img)), "decoded pixels"
        if hit is None:
            hit, via = by_stem.get(_stem_key(img)), "stem"
        if hit is None:
            unmatched.append(Path(img).name)
        else:
            pairs.append((t_lbl, hit))
            how[via] = how.get(via, 0) + 1
    print(f"re-annotation files paired: {len(pairs)} / 122   via {how}")
    if unmatched:
        print(f"  UNPAIRED ({len(unmatched)}): {unmatched[:5]}{' ...' if len(unmatched) > 5 else ''}")
    return pairs

SIGMA_TABLE = None
if REANNOT_LABELS_DIR and os.path.isdir(REANNOT_LABELS_DIR):
    pairs = []
    for t_lbl, r_lbl in reannotation_file_pairs(REANNOT_LABELS_DIR):
        pairs += match_annotation_pair(load_label(t_lbl), load_label(r_lbl), IMG_SIZE, IMG_SIZE)
    print(f"matched implant pairs (intra-observer): {len(pairs)}")
    SIGMA_TABLE = sigma_from_pairs(pairs)
    SIGMA_TABLE["source"] = "intra-observer, internal test set"
    print(SIGMA_TABLE.to_string(index=False))
    SIGMA_TABLE.to_csv(OUT_DIR / "sigma_intraobserver.csv", index=False)
    record("sigma_source", "intra-observer re-annotation of the internal test set")
    record("sigma_n_pairs", len(pairs))
    record("sigma_per_keypoint", SIGMA_TABLE.set_index("keypoint")["sigma"].to_dict())
else:
    print("REANNOT_LABELS_DIR not set — intra-observer sigma derivation skipped.")
    print("Provide the second annotation of the internal test set to complete Editor comment 10.")
""")

code(r"""
# Sensitivity source: three independent annotators on the external 190-image set.
# Reported only as a secondary check; see the note in revision2/README.md before citing it.
SIGMA_TABLE_EXT = None
ext_dirs = sorted(glob.glob(f"{VANJSKA}/Vanjska_evaluacija_*"))
if len(ext_dirs) >= 2:
    import itertools
    def load_ann(d):
        out = {}
        for f in glob.glob(f"{d}/train/labels/*.txt"):
            out[Path(f).name.split("_jpg")[0]] = load_label(f)
        return out
    anns = {Path(d).name.split("_")[-1]: load_ann(d) for d in ext_dirs}
    pairs = []
    for a, b in itertools.combinations(anns, 2):
        for k in anns[a]:
            if k in anns[b]:
                pairs += match_annotation_pair(anns[a][k], anns[b][k], IMG_SIZE, IMG_SIZE)
    print(f"annotators: {list(anns)} | matched implant pairs: {len(pairs)}")
    SIGMA_TABLE_EXT = sigma_from_pairs(pairs)
    SIGMA_TABLE_EXT["source"] = "inter-observer, external set"
    print(SIGMA_TABLE_EXT.to_string(index=False))
    SIGMA_TABLE_EXT.to_csv(OUT_DIR / "sigma_interobserver_external.csv", index=False)
else:
    print("external annotator sets not found — skipped")
""")

code(r"""
# Recompute OKS-dependent metrics with empirically derived, keypoint-specific constants.
SIGMA_USED = SIGMA_TABLE if SIGMA_TABLE is not None else SIGMA_TABLE_EXT
if SIGMA_USED is not None:
    sig = [float(SIGMA_USED.set_index("internal").loc[n, "sigma"]) for n in KPTS]
    print("per-keypoint sigma:", {KPT_LABEL[n]: s for n, s in zip(KPTS, sig)})

    def oks_row(r):
        d = np.array([r[f"d_{n}"] * r["diag"] for n in KPTS])
        k = 2 * np.array(sig)
        return float(np.mean(np.exp(-(d ** 2) / (2 * r["area"] * k ** 2))))

    det = det.assign(oks_empirical=det.apply(oks_row, axis=1))
    v = det["oks_empirical"].to_numpy()
    lo, hi = boot_ci(v)
    wv = stats.wilcoxon(v - 0.8, alternative="greater")
    print(f"\n{'':26s} {'empirical':>10s} {'fixed 0.075':>12s}")
    print(f"{'mean OKS':26s} {v.mean():10.4f} {oks_fixed.mean():12.4f}")
    print(f"{'median':26s} {np.median(v):10.4f} {np.median(oks_fixed):12.4f}")
    print(f"{'SD':26s} {v.std(ddof=0):10.4f} {oks_fixed.std(ddof=0):12.4f}")
    print(f"{'95% CI':26s} {lo:.4f}-{hi:.4f}")
    print(f"{'Wilcoxon vs 0.8':26s} {wv.statistic:10.1f} {stats.wilcoxon(oks_fixed - 0.8, alternative='greater').statistic:12.1f}")
    print(f"{'p':26s} {wv.pvalue:10.3g}")
    record("oks_empirical_mean", round(float(v.mean()), 4))
    record("oks_empirical_median", round(float(np.median(v)), 4))
    record("oks_empirical_sd", round(float(v.std(ddof=0)), 4))
    record("oks_empirical_ci", [round(lo, 4), round(hi, 4)])
    record("oks_empirical_wilcoxon_stat", float(wv.statistic))
    record("oks_empirical_wilcoxon_p", float(wv.pvalue))
    det.to_csv(OUT_DIR / "per_implant_test.csv", index=False)   # now carries oks_empirical
    print("\nNOTE: mAP@50-95 (Pose) also depends on sigma. Ultralytics uses its own COCO sigma")
    print("array; to report a pose mAP consistent with these constants, set")
    print("model.model.kpt_sigmas / OKS_SIGMA in the validator before re-running §4.")
else:
    print("no empirical sigma available — section skipped")
""")

# ---------------------------------------------------------------------
md(r"""
## 8. Marginal bone loss quantification

Reported for both geometric definitions and for both the full set and the apex-outlier-filtered
subset, so that the provenance of every previously reported figure is unambiguous.

The two definitions are not interchangeable. Under `proj`, the shoulder-to-bone vector is projected
onto the implant axis and the distance is set to zero when the bone level sits at or coronal to the
shoulder, i.e. an implant with no radiographic bone loss scores 0%. Under `eucl`, the raw distance
between the two points is taken regardless of direction, so the same implant scores a non-zero loss.
`proj` is the method the Methods section describes; `eucl` is what the currently published notebook
computes. The diagnostic below quantifies how far apart the two definitions place the reference
values, which is the origin of the competing MAE figures in circulation.
""")

code(r"""
# How much do the two definitions differ on the expert reference values?
d = det.dropna(subset=["true_max_proj", "true_max_eucl"])
zero_proj = int((d.true_max_proj <= 1e-9).sum())
zero_eucl = int((d.true_max_eucl <= 1e-9).sum())
print(f"implants with reference MBL = 0%   proj: {zero_proj}/{len(d)}   eucl: {zero_eucl}/{len(d)}")
print(f"mean reference MBL                 proj: {d.true_max_proj.mean():.2f}%   "
      f"eucl: {d.true_max_eucl.mean():.2f}%")
print(f"mean absolute difference between the two definitions: "
      f"{(d.true_max_proj - d.true_max_eucl).abs().mean():.2f} percentage points")
record("mbl_zero_reference_proj", zero_proj)
record("mbl_zero_reference_eucl", zero_eucl)
""")

code(r"""
def mbl_metrics(df, mode):
    d = df.dropna(subset=[f"true_max_{mode}", f"pred_max_{mode}"])
    t = d[f"true_max_{mode}"].to_numpy()
    p = d[f"pred_max_{mode}"].to_numpy()
    err = np.abs(t - p)
    r, pv = stats.pearsonr(t, p)
    mae_lo, mae_hi = boot_ci(err)
    rmse_lo, rmse_hi = boot_ci(np.sqrt((t - p) ** 2), fn=lambda a: np.sqrt(np.mean(a ** 2)))
    z, se = np.arctanh(r), 1 / np.sqrt(len(t) - 3)
    w = stats.wilcoxon(err - 5.0, alternative="less")
    diff = p - t
    bias, sd = diff.mean(), diff.std(ddof=1)
    # Bland-Altman intervals: the bias uses the standard error of the mean, the limits use
    # Bland & Altman's variance for a limit of agreement, sd * sqrt(1/n + 1.96^2 / (2(n-1))).
    se_bias = sd / np.sqrt(len(t))
    se_loa = sd * np.sqrt(1 / len(t) + 1.96 ** 2 / (2 * (len(t) - 1)))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    return {
        "mode": mode, "n": len(t),
        "MAE": round(float(err.mean()), 2), "MAE_ci": [round(mae_lo, 2), round(mae_hi, 2)],
        "RMSE": round(float(np.sqrt(np.mean((t - p) ** 2))), 2),
        "RMSE_ci": [round(rmse_lo, 2), round(rmse_hi, 2)],
        "r": round(float(r), 4), "r_p": float(pv),
        "r_ci": [round(float(np.tanh(z - 1.96 * se)), 4), round(float(np.tanh(z + 1.96 * se)), 4)],
        "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "ba_bias": round(float(bias), 2),
        "ba_bias_ci": [round(float(bias - 1.96 * se_bias), 2), round(float(bias + 1.96 * se_bias), 2)],
        "ba_loa_low": round(float(loa_lo), 2),
        "ba_loa_low_ci": [round(float(loa_lo - 1.96 * se_loa), 2), round(float(loa_lo + 1.96 * se_loa), 2)],
        "ba_loa_high": round(float(loa_hi), 2),
        "ba_loa_high_ci": [round(float(loa_hi - 1.96 * se_loa), 2), round(float(loa_hi + 1.96 * se_loa), 2)],
    }

subsets = {
    "all (n=152)": det,
    "apex-filtered": det[~det.apex_outlier],
}
rows = []
for name, d in subsets.items():
    for mode in ("proj", "eucl"):
        rows.append({"subset": name, **mbl_metrics(d, mode)})
MBL = pd.DataFrame(rows)
MBL.to_csv(OUT_DIR / "mbl_metrics.csv", index=False)
print(MBL.to_string(index=False))
print("\nManuscript reports: n = 152, MAE = 6.85%, RMSE = 9.54%, r = 0.6625,")
print("Wilcoxon statistic = 6554.0, p = 0.9133, bias = -3.0%, LoA -21% to +14%")
""")

code(r"""
# Primary specification for the manuscript: unfiltered, projection-based.
PRIMARY_MODE = "proj"
prim = MBL[(MBL.subset == "all (n=152)") & (MBL["mode"] == PRIMARY_MODE)].iloc[0]
for k in ["n", "MAE", "MAE_ci", "RMSE", "RMSE_ci", "r", "r_ci", "wilcoxon_stat",
          "wilcoxon_p", "ba_bias", "ba_bias_ci", "ba_loa_low", "ba_loa_low_ci",
          "ba_loa_high", "ba_loa_high_ci"]:
    record(f"mbl_{k}", prim[k], f"unfiltered, {PRIMARY_MODE}")
print(prim.to_string())
""")

# ---------------------------------------------------------------------
md(r"""
## 9. Clustered analysis — implants nested within radiographs

**Editor comment 6.** Three complementary treatments of the non-independence, all using the
radiograph as the clustering unit:

1. GEE with an exchangeable working correlation (population-averaged mean and robust CI);
2. a linear mixed model with a random intercept per image, giving the ICC;
3. a cluster bootstrap (resampling radiographs, not implants) for MAE, Pearson r, mean IoU and OKS.

The comparison table contrasts naive and cluster-aware intervals for each headline statistic.
""")

code(r"""
import statsmodels.api as sm
import statsmodels.formula.api as smf

cl = det.dropna(subset=[f"true_max_{PRIMARY_MODE}", f"pred_max_{PRIMARY_MODE}"]).copy()
cl["abs_err"] = (cl[f"true_max_{PRIMARY_MODE}"] - cl[f"pred_max_{PRIMARY_MODE}"]).abs()
cl["img"] = cl["image"].astype("category").cat.codes

gee = smf.gee("abs_err ~ 1", groups="img", data=cl,
              cov_struct=sm.cov_struct.Exchangeable(),
              family=sm.families.Gaussian()).fit()
print(gee.summary().tables[1])
print("working correlation:", gee.cov_struct.summary())

ols_naive = smf.ols("abs_err ~ 1", data=cl).fit()
ols_rob = smf.ols("abs_err ~ 1", data=cl).fit(cov_type="cluster",
                                              cov_kwds={"groups": cl["img"]})
print(f"\nMAE naive SE          : {ols_naive.bse[0]:.4f}  CI {ols_naive.conf_int().iloc[0].round(3).tolist()}")
print(f"MAE cluster-robust SE : {ols_rob.bse[0]:.4f}  CI {ols_rob.conf_int().iloc[0].round(3).tolist()}")
print(f"MAE GEE               : {gee.params[0]:.4f}  CI {gee.conf_int().iloc[0].round(3).tolist()}")

record("mbl_mae_gee", round(float(gee.params.iloc[0]), 2))
record("mbl_mae_gee_ci", [round(float(x), 2) for x in gee.conf_int().iloc[0]])
record("mbl_mae_cluster_robust_ci", [round(float(x), 2) for x in ols_rob.conf_int().iloc[0]])
""")

code(r"""
mixed = smf.mixedlm("abs_err ~ 1", cl, groups=cl["img"]).fit(reml=True)
var_re = float(mixed.cov_re.iloc[0, 0]); var_e = float(mixed.scale)
icc = var_re / (var_re + var_e)
print(f"random-intercept variance : {var_re:.3f}")
print(f"residual variance         : {var_e:.3f}")
print(f"ICC (image level)         : {icc:.4f}")
print(f"design effect             : {1 + (mean_cluster - 1) * icc:.4f}")
record("mbl_icc_image", round(icc, 4))
record("mbl_design_effect", round(1 + (mean_cluster - 1) * icc, 4))

# one-sided Wald test of the MAE against the 5% threshold, cluster-robust
z = (gee.params.iloc[0] - 5.0) / gee.bse.iloc[0]
print(f"\nGEE Wald test MAE < 5%: z = {z:.3f}, one-sided p = {stats.norm.cdf(z):.4f}")
record("mbl_mae_gee_vs5_p", float(stats.norm.cdf(z)))
""")

code(r"""
def cluster_bootstrap(df, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("image")]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        s = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            out.append(fn(s))
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5])

stats_defs = {
    "MBL MAE":  (lambda d: (d[f"true_max_{PRIMARY_MODE}"] - d[f"pred_max_{PRIMARY_MODE}"]).abs().mean(),
                 det.dropna(subset=[f"true_max_{PRIMARY_MODE}"])),
    "MBL Pearson r": (lambda d: stats.pearsonr(d[f"true_max_{PRIMARY_MODE}"], d[f"pred_max_{PRIMARY_MODE}"])[0],
                      det.dropna(subset=[f"true_max_{PRIMARY_MODE}"])),
    "mean IoU": (lambda d: d["iou"].mean(), det),
    "mean OKS": (lambda d: d["oks_fixed"].mean(), det),
    "mean keypoint error (%)": (lambda d: np.mean([d[f"d_{n}"].mean() * 100 for n in KPTS]), det),
}
rows = []
for name, (fn, d) in stats_defs.items():
    point = fn(d)
    # naive (implant-level) bootstrap, ignoring the clustering
    rng = np.random.default_rng(SEED)
    naive = [fn(d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(500)]
    n_lo, n_hi = np.percentile(naive, [2.5, 97.5])
    c_lo, c_hi = cluster_bootstrap(d, fn, n=500)
    rows.append({"statistic": name, "point": round(float(point), 4),
                 "naive_ci": [round(float(n_lo), 4), round(float(n_hi), 4)],
                 "cluster_ci": [round(float(c_lo), 4), round(float(c_hi), 4)],
                 "ci_width_ratio": round(float((c_hi - c_lo) / (n_hi - n_lo)), 3)})
CLUSTER = pd.DataFrame(rows)
CLUSTER.to_csv(OUT_DIR / "clustered_comparison.csv", index=False)
print(CLUSTER.to_string(index=False))
record("clustered_comparison", CLUSTER.to_dict("records"))
""")

# ---------------------------------------------------------------------
md(r"""
## 10. Intra-observer reliability

**Editor comment 4.** Recomputes in Python the values currently obtained with the R `irr` package
(ICC(A,1) = 0.834, ICC(A,k=2) = 0.909, MAE = 3.83 pp, r = 0.843, bbox IoU = 0.826 ± 0.098), so that
one notebook reproduces every reported number. Requires `REANNOT_LABELS_DIR`.
""")

code(r"""
def icc_a1_ak(*cols):
    '''Two-way mixed effects, absolute agreement: ICC(A,1) and ICC(A,k).'''
    M = np.column_stack(cols).astype(float)
    n, k = M.shape
    grand = M.mean()
    MSR = k * ((M.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    MSC = n * ((M.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + grand
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))
    icc1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    icck = (MSR - MSE) / (MSR + (MSC - MSE) / n)
    return float(icc1), float(icck)

# Self-test against the worked example in Shrout & Fleiss (1979), Table 1,
# for which the published values are ICC(2,1) = 0.290 and ICC(2,k) = 0.620.
_SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
_i1, _ik = icc_a1_ak(*[_SF[:, j] for j in range(4)])
print(f"ICC implementation self-test: ICC(A,1) = {_i1:.4f} (published 0.290), "
      f"ICC(A,k) = {_ik:.4f} (published 0.620)")
assert abs(_i1 - 0.290) < 0.002 and abs(_ik - 0.620) < 0.002

if REANNOT_LABELS_DIR and os.path.isdir(REANNOT_LABELS_DIR):
    recs = []
    for f, g in reannotation_file_pairs(REANNOT_LABELS_DIR):
        for ra, rb in match_annotation_pair(load_label(f), load_label(g), IMG_SIZE, IMG_SIZE):
            ka, kb = kps_from_row(ra, IMG_SIZE, IMG_SIZE), kps_from_row(rb, IMG_SIZE, IMG_SIZE)
            row = {"image": Path(f).name,
                   "iou": bbox_iou_xywh(ra[1:5] * IMG_SIZE, rb[1:5] * IMG_SIZE)}
            for tag, kk in [("a", ka), ("b", kb)]:
                for mode in ("proj", "eucl"):
                    s = mbl_sides(kk, mode)
                    # MBL1 is the image-left side (IS1/BL1), MBL2 the image-right side,
                    # matching the keypoint labelling used throughout and in the COCO export.
                    v1 = s["lijevi"] / s["srednji"] * 100
                    v2 = s["desni"] / s["srednji"] * 100
                    row[f"{tag}_mbl1_{mode}"], row[f"{tag}_mbl2_{mode}"] = v1, v2
                    row[f"{tag}_max_{mode}"] = max(v1, v2)
            recs.append(row)
    RE = pd.DataFrame(recs)
    print(f"paired implants: {len(RE)} of {sum(len(load_label(f)) for f, _ in reannotation_file_pairs(REANNOT_LABELS_DIR))}")
    print(f"bbox IoU: {RE.iou.mean():.3f} +/- {RE.iou.std(ddof=0):.3f}"
          f"   (manuscript: 0.826 +/- 0.098)")

    rows = []
    for mode in ("proj", "eucl"):
        a, b = RE[f"a_max_{mode}"], RE[f"b_max_{mode}"]
        i1, ik = icc_a1_ak(a, b)
        rows.append({"mode": mode, "n": len(RE),
                     "ICC_A1": round(i1, 3), "ICC_Ak": round(ik, 3),
                     "MAE_pp": round(float((a - b).abs().mean()), 2),
                     "pearson_r": round(float(stats.pearsonr(a, b)[0]), 3),
                     "ICC_A1_MBL1": round(icc_a1_ak(RE[f"a_mbl1_{mode}"], RE[f"b_mbl1_{mode}"])[0], 3),
                     "ICC_A1_MBL2": round(icc_a1_ak(RE[f"a_mbl2_{mode}"], RE[f"b_mbl2_{mode}"])[0], 3)})
    INTRA = pd.DataFrame(rows)
    print()
    print(INTRA.to_string(index=False))
    print("\nmanuscript  ICC(A,1) = 0.834, ICC(A,k) = 0.909, MAE = 3.83 pp, r = 0.843,"
          "\n            ICC(A,1) MBL1 = 0.739, MBL2 = 0.784")

    prim = INTRA[INTRA["mode"] == PRIMARY_MODE].iloc[0]
    record("intra_icc_a1", float(prim.ICC_A1)); record("intra_icc_ak", float(prim.ICC_Ak))
    record("intra_mae", float(prim.MAE_pp)); record("intra_r", float(prim.pearson_r))
    record("intra_iou_mean", round(float(RE.iou.mean()), 3))
    RE.to_csv(OUT_DIR / "intraobserver_pairs.csv", index=False)
    INTRA.to_csv(OUT_DIR / "intraobserver_agreement.csv", index=False)

    # The intra-observer error is the floor the Editor asked the model error to be read against.
    print(f"\nreference-standard repeatability (MAE {prim.MAE_pp} pp) vs model MAE "
          f"{RESULTS['mbl_MAE']['value']}% on the same 152 implants")
else:
    print("REANNOT_LABELS_DIR not set — intra-observer reliability skipped.")
""")

# ---------------------------------------------------------------------
md(r"""
## 11. Figures

**Reviewer 2 comment 3.** Every axis, annotation and regression coefficient is on the percentage
scale (0-100). Side labels follow the image-based right/left convention used in the annotations,
not mesial/distal.
""")

code(r"""
plot_df = det.dropna(subset=[f"true_max_{PRIMARY_MODE}", f"pred_max_{PRIMARY_MODE}"])
x = plot_df[f"true_max_{PRIMARY_MODE}"].to_numpy()
y = plot_df[f"pred_max_{PRIMARY_MODE}"].to_numpy()

slope, intercept, rv, pv, _ = stats.linregress(x, y)
r_sc, p_sc = stats.pearsonr(x, y)
z, se = np.arctanh(r_sc), 1 / np.sqrt(len(x) - 3)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.scatter(x, y, alpha=0.55, s=22, color="steelblue")
lim = [0, max(x.max(), y.max()) * 1.05]
ax.plot(lim, lim, "--", color="gray", lw=1.2, label="Identity line (y = x)")
ax.plot(np.array(lim), intercept + slope * np.array(lim), color="red", lw=1.2,
        label=f"Regression (R² = {rv**2:.2f})")
ax.annotate(f"r = {r_sc:.3f} (95% CI {np.tanh(z-1.96*se):.3f}–{np.tanh(z+1.96*se):.3f})\n"
            f"p < 0.0001,  n = {len(x)}\ny = {intercept:.2f} + {slope:.2f}x",
            xy=(0.04, 0.86), xycoords="axes fraction", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8))
ax.set_xlabel("Expert-measured MBL (%)"); ax.set_ylabel("Model-predicted MBL (%)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title("MBL: model vs expert (internal test set)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "Figure4_scatter.png", dpi=300); plt.show()

print(f"regression on percentage scale: y = {intercept:.2f} + {slope:.2f}x  (R² = {rv**2:.2f})")
print("manuscript currently prints y = 0.03 + 0.56x, which is the proportion-scale fit")
record("regression_intercept_pct", round(float(intercept), 2))
record("regression_slope", round(float(slope), 2))
record("regression_r2", round(float(rv ** 2), 2))
""")

code(r"""
diff = y - x
mean_xy = (x + y) / 2
bias, sd = diff.mean(), diff.std(ddof=1)
loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mean_xy, diff, alpha=0.55, s=22, color="steelblue")
for val, lab, style in [(bias, f"Bias {bias:.2f}%", "-"),
                        (loa_lo, f"Lower LoA {loa_lo:.2f}%", "--"),
                        (loa_hi, f"Upper LoA {loa_hi:.2f}%", "--")]:
    ax.axhline(val, ls=style, color="red" if style == "-" else "gray", lw=1.2)
    ax.text(ax.get_xlim()[1], val, " " + lab, va="center", fontsize=9)
ax.set_xlabel("Mean of model and expert MBL (%)")
ax.set_ylabel("Model − expert MBL (%)")
ax.set_title("Bland–Altman analysis (internal test set)")
fig.tight_layout(); fig.savefig(FIG_DIR / "Figure5_bland_altman.png", dpi=300); plt.show()
print(f"bias {bias:.2f}%  LoA {loa_lo:.2f}% to {loa_hi:.2f}%  (n = {len(diff)})")
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=long, x="label", y="err_pct",
            order=["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"], ax=ax)
ax.set_xlabel("Keypoint"); ax.set_ylabel("Scaled Euclidean error (% of bounding-box diagonal)")
ax.set_title("Keypoint localisation error, all measurements (unfiltered)")
fig.tight_layout(); fig.savefig(FIG_DIR / "FigureS5_keypoint_boxplot.png", dpi=300); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
panels = [("desni", "Absolute difference — right side"),
          ("lijevi", "Absolute difference — left side"),
          ("max", "Absolute difference — maximum MBL")]
for axi, (side, title) in zip(axes, panels):
    vals = (plot_df[f"true_{side}_{PRIMARY_MODE}"] - plot_df[f"pred_{side}_{PRIMARY_MODE}"]).abs()
    sns.histplot(vals, bins=20, kde=True, ax=axi)
    axi.set_title(title); axi.set_xlabel("Absolute difference in MBL (%)")
fig.tight_layout(); fig.savefig(FIG_DIR / "FigureS6_mbl_histograms.png", dpi=300); plt.show()
print("axes span:", round(float(plot_df[f'true_max_{PRIMARY_MODE}'].max()), 1), "% (percentage scale)")
""")

# ---------------------------------------------------------------------
md(r"""
## 12. Results manifest and manuscript cross-check

`MANUSCRIPT_CLAIMS` holds the values printed in `IDJ_manuscript_FINAL_2026-06-25.docx`. The table
flags every quantity this notebook cannot reproduce, so the response letter and the manuscript can
be corrected from one authoritative list.
""")

code(r"""
MANUSCRIPT_CLAIMS = {
    "n_images_total": 1214, "n_implants_total": 1570,
    "n_images_train": 973, "n_implants_train": 1258,
    "n_images_valid": 119, "n_implants_valid": 160,
    "n_images_test": 122, "n_implants_test": 152,
    "iou_mean": 0.8434, "iou_median": 0.8702, "iou_sd": 0.0933,
    "iou_wilcoxon_stat": 8825.0,
    "box_mAP50": 0.9933, "box_mAP50_95": 0.7860,
    "pose_mAP50": 0.9933, "pose_mAP50_95": 0.9328,
    "oks_fixed_mean": 0.8808, "oks_fixed_median": 0.9297, "oks_fixed_sd": 0.1220,
    "oks_wilcoxon_stat": 9951.0,
    "kpt_err_unfiltered_n": 912, "kpt_err_unfiltered_mean": 3.64, "kpt_err_unfiltered_sd": 4.24,
    "kpt_err_n_outliers": 71, "kpt_err_filtered_mean": 2.98, "kpt_err_filtered_sd": 1.47,
    "mbl_n": 152, "mbl_MAE": 6.85, "mbl_RMSE": 9.54, "mbl_r": 0.6625,
    "mbl_wilcoxon_stat": 6554.0, "mbl_wilcoxon_p": 0.9133,
    "mbl_ba_bias": -3.0, "mbl_ba_loa_low": -21.0, "mbl_ba_loa_high": 14.0,
    "regression_intercept_pct": 0.03, "regression_slope": 0.56, "regression_r2": 0.44,
    "intra_icc_a1": 0.834, "intra_icc_ak": 0.909, "intra_mae": 3.83,
    "intra_r": 0.843, "intra_iou_mean": 0.826,
}

rows = []
for key, claimed in MANUSCRIPT_CLAIMS.items():
    got = RESULTS.get(key, {}).get("value", None)
    if got is None:
        status = "not computed"
        delta = None
    elif isinstance(got, (int, float)) and isinstance(claimed, (int, float)):
        delta = round(float(got) - float(claimed), 4)
        tol = 0.005 if abs(claimed) < 5 else 0.05
        status = "match" if abs(delta) <= tol else "MISMATCH"
    else:
        delta = None
        status = "match" if got == claimed else "MISMATCH"
    rows.append({"quantity": key, "manuscript": claimed, "recomputed": got,
                 "delta": delta, "status": status})
CHECK = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(CHECK.to_string(index=False))
print("\nMISMATCH / not computed:", int((CHECK.status != "match").sum()), "of", len(CHECK))
CHECK.to_csv(OUT_DIR / "manuscript_crosscheck.csv", index=False)
""")

code(r"""
manifest = {
    "generated_by": "IDJ_revision2_analysis.ipynb",
    "environment": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                    "numpy": np.__version__, "python": sys.version.split()[0]},
    "config": {"model": MODEL_FILE, "pred_conf": PRED_CONF, "pred_iou": PRED_IOU,
               "val_conf": VAL_CONF, "val_iou": VAL_IOU, "seed": SEED,
               "sigma_fixed": SIGMA_FIXED, "primary_mbl_mode": PRIMARY_MODE,
               "n_boot": N_BOOT, "n_boot_map": N_BOOT_MAP,
               "reannotation_supplied": bool(REANNOT_LABELS_DIR)},
    "results": RESULTS,
}
with open(OUT_DIR / "results_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=float)
print("written:", OUT_DIR / "results_manifest.json")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size/1024:.1f} KB)")
""")

# =====================================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

repo = "/home/matthew/Documents/Workspace/dora-diplomski"
out = os.path.join(repo, "revision2", "IDJ_revision2_analysis.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}: {len(CELLS)} cells")
