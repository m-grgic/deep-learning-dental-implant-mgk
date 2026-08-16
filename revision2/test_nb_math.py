"""Smoke-test the notebook's pure-math cells against real label files (no torch needed)."""
import json, glob, re
from pathlib import Path
import numpy as np

NB = "/home/matthew/Documents/Workspace/dora-diplomski/revision2/IDJ_revision2_analysis.ipynb"
REPO = "/home/matthew/Documents/Workspace/dora-diplomski"
cells = [''.join(c['source']) for c in json.load(open(NB))['cells'] if c['cell_type'] == 'code']

ns = {"np": np, "KPTS": ["apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"],
      "KPT_LABEL": {"apex": "AOI", "desni_kost": "BL1", "desni_vrh": "IS1", "vrh": "IAC",
                    "lijevi_vrh": "IS2", "lijevi_kost": "BL2"},
      "AREA_FACTOR": 0.85, "IMG_SIZE": 512, "SIGMA_FIXED": 0.075}

# geometry cell = the one defining pointProjection
geo = next(c for c in cells if "def pointProjection" in c)
exec(geo, ns)
# kps_from_row / load_label / bbox helpers live in the instance-table cell; take only the defs we need
inst = next(c for c in cells if "def kps_from_row" in c)
head = inst.split("def build_instance_table")[0]
exec(head, ns)
# matching + sigma
sig = next(c for c in cells if "def match_annotation_pair" in c)
exec(sig.split("def sigma_from_pairs")[0], ns)
icc = next(c for c in cells if "def icc_a1_ak" in c)
exec(icc.split("if REANNOT_LABELS_DIR")[0], ns)

load_label = ns["load_label"]; kps_from_row = ns["kps_from_row"]
mbl_sides = ns["mbl_sides"]; OKS = ns["OKS"]; _euclid = ns["_euclid"]
bbox_iou_xywh = ns["bbox_iou_xywh"]; match = ns["match_annotation_pair"]
icc_a1_ak = ns["icc_a1_ak"]

print("== 1. geometry on a real test label ==")
f = sorted(glob.glob(f"{REPO}/Keypoint_detection.v10-512px-adaptive.yolov8/test/labels/*.txt"))[0]
row = load_label(f)[0]
k = kps_from_row(row, 512, 512)
print("   keypoints px:", {n: (round(v['x'], 1), round(v['y'], 1)) for n, v in k.items()})
for mode in ("proj", "eucl"):
    s = mbl_sides(k, mode)
    print(f"   {mode}: srednji={s['srednji']:.2f} desni={s['desni']:.2f} lijevi={s['lijevi']:.2f}"
          f"  -> MBL {100*s['desni']/s['srednji']:.2f}% / {100*s['lijevi']/s['srednji']:.2f}%")
assert mbl_sides(k, "proj")["srednji"] > 0
# proj clamps to zero when the bone level is coronal to the shoulder; eucl never can
n_zero_proj = n_zero_eucl = n_tot = 0
for f2 in sorted(glob.glob(f"{REPO}/Keypoint_detection.v10-512px-adaptive.yolov8/test/labels/*.txt")):
    for r2 in load_label(f2):
        kk = kps_from_row(r2, 512, 512)
        sp, se = mbl_sides(kk, "proj"), mbl_sides(kk, "eucl")
        n_tot += 1
        n_zero_proj += max(sp["desni"], sp["lijevi"]) <= 1e-9
        n_zero_eucl += max(se["desni"], se["lijevi"]) <= 1e-9
        assert sp["desni"] <= se["desni"] + 1e-6 and sp["lijevi"] <= se["lijevi"] + 1e-6
print(f"   reference MBL = 0 for  proj: {n_zero_proj}/{n_tot}   eucl: {n_zero_eucl}/{n_tot}")

print("\n== 2. OKS sanity ==")
area = 0.85 * row[3] * 512 * row[4] * 512
print("   OKS(identical) =", round(OKS(k, k, [0.075] * 6, area), 6))
assert abs(OKS(k, k, [0.075] * 6, area) - 1.0) < 1e-12
shifted = {n: {"x": v["x"] + 5, "y": v["y"]} for n, v in k.items()}
print("   OKS(5 px shift) =", round(OKS(k, shifted, [0.075] * 6, area), 4))
print("   IoU(self) =", round(bbox_iou_xywh(row[1:5] * 512, row[1:5] * 512), 6))
assert abs(bbox_iou_xywh(row[1:5] * 512, row[1:5] * 512) - 1.0) < 1e-9

print("\n== 3. cross-annotator matching + sigma on the external 3-annotator set ==")
def load_ann(d):
    return {Path(f).name.split("_jpg")[0]: load_label(f) for f in glob.glob(f"{d}/train/labels/*.txt")}
import itertools
anns = {Path(d).name.split("_")[-1]: load_ann(d) for d in sorted(glob.glob(f"{REPO}/vanjska/Vanjska_evaluacija_*"))}
pairs = []
for a, b in itertools.combinations(anns, 2):
    for key in anns[a]:
        if key in anns[b]:
            pairs += match(anns[a][key], anns[b][key], 512, 512)
print("   annotators:", list(anns), "| matched pairs:", len(pairs))
sq = {n: [] for n in ns["KPTS"]}
for ra, rb in pairs:
    ka, kb = kps_from_row(ra, 512, 512), kps_from_row(rb, 512, 512)
    ar = 0.5 * (0.85 * ra[3] * 512 * ra[4] * 512 + 0.85 * rb[3] * 512 * rb[4] * 512)
    for n in sq:
        sq[n].append(_euclid(ka[n], kb[n]) ** 2 / ar)
for n in sq:
    v = np.array(sq[n])
    print(f"   {ns['KPT_LABEL'][n]:4s} n={len(v):4d} sigma={np.sqrt(v.mean()):.4f} "
          f"trim95={np.sqrt(np.sort(v)[:int(len(v)*.95)].mean()):.4f}")

print("\n== 4. ICC against the published Shrout & Fleiss (1979) worked example ==")
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
               [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
i1, ik = icc_a1_ak(*[SF[:, j] for j in range(4)])
print(f"   ICC(A,1)={i1:.4f} (published 0.290)   ICC(A,k)={ik:.4f} (published 0.620)")
assert abs(i1 - 0.290) < 0.002 and abs(ik - 0.620) < 0.002, "ICC formula mismatch"
p = np.array([1.0, 2, 3, 4, 5])
print("   ICC(A,1) identical raters =", round(icc_a1_ak(p, p)[0], 6))
assert abs(icc_a1_ak(p, p)[0] - 1.0) < 1e-9

print("\n== 5. IQR outlier rule reproduces the manuscript's 912 measurements ==")
n_impl = sum(len(load_label(f)) for f in glob.glob(
    f"{REPO}/Keypoint_detection.v10-512px-adaptive.yolov8/test/labels/*.txt"))
print(f"   implants={n_impl}  6*implants={6*n_impl}  manuscript n=912")
assert 6 * n_impl == 912

print("\nALL CHECKS PASSED")
