#!/usr/bin/env python
"""
M2_figures.py - manuscript figures for the merged round-2 package with corrected data & consistent % scale.

Fig 4  : scatter expert vs model maxMBL (%), regression stated on the SAME % scale (R2.3)
Fig 5  : Bland-Altman (%), bias/LoA + cluster-bootstrap CIs
Fig S3 : IoU distribution
Fig S4 : OKS distribution (per-keypoint sigma_i, k_i = 2*sigma_i)
Fig S5 : per-keypoint scaled error boxplots, UNFILTERED, full range + zoom (E6)
Fig S6 : per-side absolute MBL error histograms, % scale, sides defined as image left/right (R2.3)

All MBL panels use the projection method described in Methods Eq. 2a/2b.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + "/output"
FIG = os.path.abspath(HERE + "/../figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GRAY, DARK = "#4477AA", "#EE7733", "#8C8C8C", "#333333"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.dpi": 300,
})

SRC = os.path.abspath(HERE + "/../../revision2nd_response/analysis/output")
mbl = pd.read_csv(f"{SRC}/mbl_proj.csv")
kdf = pd.read_csv(f"{SRC}/per_keypoint_errors.csv")
idf = pd.read_csv(f"{OUT}/per_instance_oks.csv")
cb = json.load(open(f"{SRC}/cluster_bootstrap.json"))["ci"]
NR = json.load(open(f"{OUT}/numbers_of_record.json"))

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------------- Figure 4: scatter ----------------
x = mbl.tmax*100; y = mbl.pmax*100
from scipy import stats as st
sl, ic, rv, pv, se = st.linregress(x, y)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
lim = max(x.max(), y.max())*1.05
ax.plot([0, lim], [0, lim], ls="--", lw=1.2, c=GRAY, label="Identity (y = x)")
xs = np.linspace(0, lim, 50)
ax.plot(xs, ic + sl*xs, lw=1.8, c=ORANGE,
        label=f"Regression: y = {ic:.2f} + {sl:.2f}x  (R² = {rv**2:.2f})")
ax.scatter(x, y, s=22, c=BLUE, alpha=0.55, edgecolors="white", linewidths=0.4, zorder=3)
ax.set_xlabel("Expert-measured maximal MBL (% of implant length)")
ax.set_ylabel("Model-predicted maximal MBL (% of implant length)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
rci = cb["mbl_proj_r"]
ax.text(0.98, 0.02, f"r = {rv:.3f} (95% CI {rci['lo']:.3f}\u2013{rci['hi']:.3f})\nn = {len(x)} implants",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=DARK)
save(fig, "Figure4_scatter_MBL")

# ---------------- Figure 5: Bland-Altman ----------------
diff = (mbl.pmax - mbl.tmax)*100
avg = (mbl.pmax + mbl.tmax)/2*100
bias = diff.mean(); sd = diff.std(ddof=1)
lo, hi = bias-1.96*sd, bias+1.96*sd
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.scatter(avg, diff, s=22, c=BLUE, alpha=0.55, edgecolors="white", linewidths=0.4, zorder=3)
ax.axhline(0, lw=0.8, c="#BBBBBB")
ax.axhline(bias, lw=1.6, c=ORANGE)
ax.axhline(lo, lw=1.2, c=GRAY, ls="--"); ax.axhline(hi, lw=1.2, c=GRAY, ls="--")
def ci_str(key):
    if key in cb: return f" [95% CI {cb[key]['lo']*100:.1f} to {cb[key]['hi']*100:.1f}]"
    return ""
ax.annotate(f"Bias {bias:.1f}%{ci_str('mbl_proj_bias')}", xy=(1.01, bias), xycoords=("axes fraction", "data"),
            fontsize=8, color=DARK, va="center")
ax.annotate(f"+1.96 SD  {hi:.1f}%", xy=(1.01, hi), xycoords=("axes fraction", "data"), fontsize=8, color=DARK, va="center")
ax.annotate(f"−1.96 SD  {lo:.1f}%", xy=(1.01, lo), xycoords=("axes fraction", "data"), fontsize=8, color=DARK, va="center")
ax.set_xlabel("Mean of model and expert maximal MBL (%)")
ax.set_ylabel("Model − expert difference (percentage points)")
save(fig, "Figure5_bland_altman_MBL")

# ---------------- S3 / S4: IoU & OKS distributions ----------------
for col, name, xlabel in [("iou", "FigureS3_iou_distribution", "IoU"),
                          ("oks_new", "FigureS4_oks_distribution", "OKS")]:
    v = idf[col].dropna()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.hist(v, bins=20, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(0.8, c=ORANGE, lw=1.4, ls="--")
    ax.text(0.8, ax.get_ylim()[1]*0.97, " 0.8 threshold", color=ORANGE, fontsize=8, va="top")
    ax.axvline(v.mean(), c=DARK, lw=1.2)
    ax.text(v.mean(), ax.get_ylim()[1]*0.85, f" mean {v.mean():.3f}", color=DARK, fontsize=8, va="top")
    ax.set_xlabel(xlabel); ax.set_ylabel("Number of implants")
    save(fig, name)

# ---------------- S5: per-keypoint boxplots, unfiltered, full + zoom ----------------
order = ["AOI", "IAC", "IS1", "IS2", "BL1", "BL2"]
data = [kdf[kdf.keypoint == k].error.values*100 for k in order]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, ylim, title in [(axes[0], None, "Full range (all 912 measurements)"),
                        (axes[1], (0, 15), "Zoom ≤ 15%")]:
    bp = ax.boxplot(data, labels=order, widths=0.55, patch_artist=True,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=GRAY,
                                    markeredgecolor="none", alpha=0.7),
                    medianprops=dict(color=ORANGE, lw=1.5),
                    boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=DARK, lw=0.8),
                    whiskerprops=dict(color=DARK, lw=0.8), capprops=dict(color=DARK, lw=0.8))
    ax.set_ylabel("Scaled Euclidean error (% of box diagonal)")
    ax.set_title(title, fontsize=9)
    if ylim: ax.set_ylim(*ylim)
axes[0].annotate("gross localisation failure\n(AOI, 196%)", xy=(1, data[0].max()),
                 xytext=(2.1, data[0].max()*0.82), fontsize=7.5, color=DARK,
                 arrowprops=dict(arrowstyle="->", color=DARK, lw=0.7))
save(fig, "FigureS5_keypoint_errors_unfiltered")

# ---------------- S6: side-specific absolute errors ----------------
d1 = (mbl.p_lijevi - mbl.t_lijevi).abs()*100   # side 1 = image left  (IS1-BL1)
d2 = (mbl.p_desni - mbl.t_desni).abs()*100     # side 2 = image right (IS2-BL2)
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True, sharex=True)
bins = np.arange(0, max(d1.max(), d2.max())+2.5, 2.5)
for ax, d, lab in [(axes[0], d1, "Side 1 (image left; IS1–BL1)"),
                   (axes[1], d2, "Side 2 (image right; IS2–BL2)")]:
    ax.hist(d, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(d.mean(), c=ORANGE, lw=1.4)
    ax.text(d.mean()+0.6, ax.get_ylim()[1]*0.92, f"MAE {d.mean():.2f}%", color=ORANGE, fontsize=8)
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("Absolute difference (percentage points)")
axes[0].set_ylabel("Number of implants")
save(fig, "FigureS6_side_absolute_errors")

print("all figures done ->", FIG)
