#!/usr/bin/env python
"""
M1_numbers.py - every number that appears in the merged package, recomputed here.

Sources (all produced by revision2nd_response/analysis/01_reproduce_internal_eval.py,
which reproduces the published detection/localisation statistics exactly):
    per_instance.csv, per_keypoint_errors.csv, matched_keypoints.csv, mbl_proj.csv
plus the archived ground-truth label files (for bounding-box width/height).

Adds, relative to that run:
  - OKS recomputed with the empirically derived per-keypoint sigmas (k_i = 2*sigma_i)
  - cluster and non-clustered bootstrap CIs for the new OKS
  - radiograph-level ICC and design effect for the MBL absolute error
  - measurement-level descriptive statistics and Wilcoxon tests
Output: output/numbers_of_record.json
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats

SEED, NBOOT = 20260816, 2000
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(HERE + "/../../revision2nd_response/analysis/output")
BASE = os.environ.get("DATASET_ROOT",
                     os.path.abspath(HERE + "/../../Keypoint_detection.v10-512px-adaptive.yolov8"))
LBL  = os.path.join(BASE, "test", "test_labels")
OUT  = HERE + "/output"; os.makedirs(OUT, exist_ok=True)

KP   = ["apex","desni_kost","desni_vrh","vrh","lijevi_vrh","lijevi_kost"]
LABEL= {"apex":"AOI","desni_kost":"BL2","desni_vrh":"IS2","vrh":"IAC","lijevi_vrh":"IS1","lijevi_kost":"BL1"}
SIGMA= {"AOI":0.029,"IAC":0.073,"IS1":0.221,"IS2":0.216,"BL1":0.237,"BL2":0.243}   # per-keypoint sigma_i

mk  = pd.read_csv(f"{SRC}/matched_keypoints.csv")
idf = pd.read_csv(f"{SRC}/per_instance.csv")
kdf = pd.read_csv(f"{SRC}/per_keypoint_errors.csv")
mbl = pd.read_csv(f"{SRC}/mbl_proj.csv")

# ---------- OKS with per-keypoint sigmas -------------------------------------------
wh = {}
for f in glob.glob(LBL + "/*.txt"):
    stem = os.path.basename(f)[:-4]
    for i, line in enumerate(open(f).read().strip().splitlines()):
        v = [float(x) for x in line.split()]
        if len(v) >= 5: wh[(stem, i)] = (v[3]*512.0, v[4]*512.0)
area = np.array([wh[(r.filename, r.instance_idx)][0]*wh[(r.filename, r.instance_idx)][1]
                 for r in mk.itertuples()]) * 0.85
d = np.column_stack([np.hypot(mk[f"t_{n}_x"]-mk[f"p_{n}_x"], mk[f"t_{n}_y"]-mk[f"p_{n}_y"]) for n in KP])
k_new = np.array([2*SIGMA[LABEL[n]] for n in KP])
k_old = np.full(6, 2*0.075)
oks_new = np.mean(np.exp(-(d**2)/(2*area[:, None]*(k_new[None, :]**2))), axis=1)
oks_old = np.mean(np.exp(-(d**2)/(2*area[:, None]*(k_old[None, :]**2))), axis=1)
mk_key = mk[["filename","instance_idx"]].copy(); mk_key["oks_new"] = oks_new; mk_key["oks_old"] = oks_old
idf = idf.merge(mk_key, on=["filename","instance_idx"], how="left")
idf.to_csv(f"{OUT}/per_instance_oks.csv", index=False)

assert abs(oks_old.mean() - idf.oks.mean()) < 1e-5, "old-sigma OKS must match the reproduction run"  # CSV coordinate precision

def wil(x, thr, alt):
    W, p = stats.wilcoxon(np.asarray(x) - thr, alternative=alt); return float(W), float(p)

R = {"seed": SEED, "n_boot": NBOOT, "n_implants": int(len(idf)), "n_radiographs": int(idf.filename.nunique())}
R["oks_new"] = dict(mean=float(oks_new.mean()), sd=float(oks_new.std(ddof=1)),
                    median=float(np.median(oks_new)), wilcoxon=wil(oks_new, 0.8, "greater"),
                    sigma=SIGMA, note="k_i = 2*sigma_i")
R["oks_old"] = dict(mean=float(oks_old.mean()), sd=float(oks_old.std(ddof=1)),
                    median=float(np.median(oks_old)), wilcoxon=wil(oks_old, 0.8, "greater"),
                    sigma=0.075, note="uniform sigma, k = 2*sigma (as used for the published values)")

# ---------- measurement-level and implant-level localisation error ------------------
e = kdf.error.values; out = kdf.outlier.astype(str).isin(["True","1","true"]).values
ia_unf = kdf.groupby(["filename","instance_idx"]).error.mean().values
ia_fil = kdf[~out].groupby(["filename","instance_idx"]).error.mean().values
R["kp_error"] = {
 "n_measurements": int(len(e)), "n_outliers": int(out.sum()),
 "pct_outliers": round(100*out.sum()/len(e), 1),
 "measurement_unfiltered": dict(mean=float(e.mean()), sd=float(e.std(ddof=1)), median=float(np.median(e)),
                                n=int(len(e)), wilcoxon=wil(e, 0.03, "less")),
 "measurement_filtered":   dict(mean=float(e[~out].mean()), sd=float(e[~out].std(ddof=1)),
                                median=float(np.median(e[~out])), n=int((~out).sum()),
                                wilcoxon=wil(e[~out], 0.03, "less")),
 "implant_unfiltered": dict(mean=float(ia_unf.mean()), sd=float(ia_unf.std(ddof=1)), n=int(len(ia_unf)),
                            wilcoxon=wil(ia_unf, 0.03, "less")),
 "implant_filtered":   dict(mean=float(ia_fil.mean()), sd=float(ia_fil.std(ddof=1)), n=int(len(ia_fil)),
                            wilcoxon=wil(ia_fil, 0.03, "less")),
 "worst_measurement": float(e.max()),
 "worst_row": kdf.loc[kdf.error.idxmax(), ["filename","keypoint","error"]].to_dict(),
}
tab = {}
for kp, g in kdf.groupby("keypoint"):
    go = g.outlier.astype(str).isin(["True","1","true"])
    tab[kp] = dict(unf_mean=round(100*g.error.mean(), 2), unf_sd=round(100*g.error.std(ddof=1), 2),
                   unf_median=round(100*g.error.median(), 2), n_excluded=int(go.sum()),
                   fil_mean=round(100*g[~go].error.mean(), 2), fil_sd=round(100*g[~go].error.std(ddof=1), 2))
R["table2"] = tab

# ---------- MBL ---------------------------------------------------------------------
diff = (mbl.pmax - mbl.tmax).values; absd = np.abs(diff)
sl, ic, r_, p_, _ = stats.linregress(mbl.tmax*100, mbl.pmax*100)
R["mbl"] = dict(n=int(len(mbl)), mae=float(absd.mean()), rmse=float(np.sqrt((diff**2).mean())),
                pearson_r=float(np.corrcoef(mbl.tmax, mbl.pmax)[0,1]),
                pearson_p=float(stats.pearsonr(mbl.tmax, mbl.pmax)[1]),
                wilcoxon_mae5=wil(absd, 0.05, "less"),
                bias=float(diff.mean()), loa_lo=float(diff.mean()-1.96*diff.std(ddof=1)),
                loa_hi=float(diff.mean()+1.96*diff.std(ddof=1)),
                slope=float(sl), intercept=float(ic), r2=float(r_**2),
                n_expert_zero=int((mbl.tmax == 0).sum()))

# ---------- ICC by radiograph and design effect (one-way random effects) -------------
def icc1(values, groups):
    df = pd.DataFrame({"v": values, "g": groups})
    k = df.groupby("g").v.count(); n = len(k); N = len(df)
    gm = df.v.mean(); gmean = df.groupby("g").v.mean()
    msb = float((k*(gmean-gm)**2).sum()/(n-1))
    msw = float(((df.v - df.g.map(gmean))**2).sum()/(N-n))
    k0 = (N - (k**2).sum()/N)/(n-1)
    icc = (msb-msw)/(msb+(k0-1)*msw)
    return float(icc), float(k0), float(N/n)
icc_abs, k0, mbar = icc1(absd, mbl.filename.values)
R["clustering"] = dict(icc_abs_error=icc_abs, mean_cluster_size=mbar,
                       design_effect=1+(mbar-1)*icc_abs,
                       n_multi_implant_radiographs=int((mbl.groupby("filename").size() > 1).sum()),
                       n_implants_in_multi=int(mbl.groupby("filename").size()[lambda s: s > 1].sum()))

# ---------- bootstrap: cluster (by radiograph) and non-clustered (by implant) --------
files = sorted(idf.filename.unique())
inst_by_file = {f: g for f, g in idf.groupby("filename")}
mbl_by_file  = {f: g for f, g in mbl.groupby("filename")}
kp_by_file   = {f: g for f, g in kdf.groupby("filename")}

def stats_from(inst, mb, kp):
    o = {}
    o["oks_new_mean"] = inst.oks_new.mean(); o["oks_new_median"] = inst.oks_new.median()
    dd = (mb.pmax - mb.tmax); ad = dd.abs()
    o["mbl_mae"] = ad.mean(); o["mbl_rmse"] = float(np.sqrt((dd**2).mean()))
    o["mbl_r"] = float(np.corrcoef(mb.tmax, mb.pmax)[0,1]) if mb.tmax.std() > 0 else np.nan
    o["mbl_bias"] = dd.mean()
    kp = kp.reset_index(drop=True)
    keep = np.ones(len(kp), dtype=bool)
    for _, g in kp.groupby("keypoint"):
        q1, q3 = g.error.quantile(.25), g.error.quantile(.75)
        lo, hi = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        keep[g.index.values] = ((g.error >= lo) & (g.error <= hi)).values
    o["err_unf"] = kp.groupby("uid").error.mean().mean()
    o["err_fil"] = kp[keep].groupby("uid").error.mean().mean()
    return o

kdf["uid"] = kdf.filename + "#" + kdf.instance_idx.astype(str)
kp_by_file = {f: g for f, g in kdf.groupby("filename")}
uids = idf.filename + "#" + idf.instance_idx.astype(str)
idf["uid"] = uids; mbl["uid"] = mbl.filename + "#" + mbl.instance_idx.astype(str)
inst_by_uid = {u: g for u, g in idf.groupby("uid")}
mbl_by_uid  = {u: g for u, g in mbl.groupby("uid")}
kp_by_uid   = {u: g for u, g in kdf.groupby("uid")}

def run_boot(unit):
    rng = np.random.default_rng(SEED); acc = {}
    keys = files if unit == "cluster" else list(uids)
    for _ in range(NBOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        if unit == "cluster":
            sel = [keys[i] for i in pick]
            inst = pd.concat([inst_by_file[f] for f in sel]); mb = pd.concat([mbl_by_file[f] for f in sel])
            kp = pd.concat([kp_by_file[f].assign(uid=lambda t, j=j: t.uid + f"@{j}") for j, f in enumerate(sel)])
        else:
            sel = [keys[i] for i in pick]
            inst = pd.concat([inst_by_uid[u] for u in sel]); mb = pd.concat([mbl_by_uid[u] for u in sel])
            kp = pd.concat([kp_by_uid[u].assign(uid=f"{u}@{j}") for j, u in enumerate(sel)])
        for k, v in stats_from(inst, mb, kp).items(): acc.setdefault(k, []).append(v)
    return {k: dict(lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5))) for k, v in acc.items()}

R["ci_cluster"] = run_boot("cluster")
R["ci_naive"]   = run_boot("implant")
json.dump(R, open(f"{OUT}/numbers_of_record.json", "w"), indent=1)

f = lambda x: f"{100*x:.2f}"
print(f"OKS new sigma : mean {R['oks_new']['mean']:.4f}  sd {R['oks_new']['sd']:.4f}  median {R['oks_new']['median']:.4f}  W {R['oks_new']['wilcoxon'][0]:.0f}")
print(f"OKS old sigma : mean {R['oks_old']['mean']:.4f}  (published 0.8808)")
print(f"kp error impl : unf {f(R['kp_error']['implant_unfiltered']['mean'])}%  fil {f(R['kp_error']['implant_filtered']['mean'])}%  outliers {R['kp_error']['n_outliers']}")
print(f"kp error meas : unf {f(R['kp_error']['measurement_unfiltered']['mean'])}%  fil {f(R['kp_error']['measurement_filtered']['mean'])}%")
print(f"MBL           : MAE {f(R['mbl']['mae'])}%  RMSE {f(R['mbl']['rmse'])}%  r {R['mbl']['pearson_r']:.4f}  zeros {R['mbl']['n_expert_zero']}")
print(f"clustering    : ICC {R['clustering']['icc_abs_error']:.3f}  DEFF {R['clustering']['design_effect']:.3f}  multi-implant radiographs {R['clustering']['n_multi_implant_radiographs']}")
for k in ("oks_new_mean","mbl_mae","mbl_r","err_unf","err_fil"):
    c, n = R["ci_cluster"][k], R["ci_naive"][k]
    print(f"  {k:14s} cluster [{c['lo']:.4f}, {c['hi']:.4f}]   naive [{n['lo']:.4f}, {n['hi']:.4f}]")
