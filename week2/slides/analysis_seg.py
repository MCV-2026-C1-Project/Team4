"""
Numbers behind the Task 3 / Task 4 slides (background removal). Writes slides/data_seg.json.

Run from the week2 folder:  python slides/analysis_seg.py
Same wall model as run_w2.py (15-px border strip at full resolution, robust Gaussian in Lab); only the decision rule
and the post-processing steps change between the variants.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_w2 as R  # noqa: E402

q2 = R.find_dir(HERE.parent / "qsd2_w2")
PATHS = R.image_paths(q2)
FULL = [cv2.imread(str(p)) for p in PATHS]
GT = [cv2.imread(str(p.with_suffix(".png")), 0) > 0 for p in PATHS]


def wall_model(img, keep=0.8):
    P = R.border_pixels(img)
    d = np.linalg.norm(P - np.median(P, 0), axis=1)
    Pk = P[d <= np.quantile(d, keep)]
    return P, Pk.mean(0), np.cov(Pk.T) + np.eye(3)


def maha(x, mu, cov):
    d = x.reshape(-1, 3) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)).reshape(x.shape[:2])


def work_lab(img):
    return cv2.cvtColor(cv2.medianBlur(R.resize_max(img, R.WORK_SIDE), 5), cv2.COLOR_BGR2LAB).astype(np.float64)


def post(m, steps=("morph", "largest", "holes", "rowcol")):
    h, w = m.shape
    m = m.astype(np.uint8)
    s = min(h, w)
    se = lambda f: cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(f * s)) | 1,) * 2)
    if "morph" in steps:
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, se(0.03)), cv2.MORPH_OPEN, se(0.02))
    if "largest" in steps:
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            m = (lbl == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = m.astype(bool)
    if "holes" in steps:
        pad = cv2.copyMakeBorder(m.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
        m = pad[1:-1, 1:-1] != 2
    if "rowcol" in steps:
        rows, cols = np.zeros_like(m), np.zeros_like(m)
        for y in np.where(m.any(1))[0]:
            xs = np.where(m[y])[0]; rows[y, xs[0]:xs[-1] + 1] = True
        for x in np.where(m.any(0))[0]:
            ys = np.where(m[:, x])[0]; cols[ys[0]:ys[-1] + 1, x] = True
        m = rows & cols
    if not m.any():
        m[:] = True
    return m


# precompute per image: work Lab, wall model, distance maps
PRE = []
for img in FULL:
    lab = work_lab(img)
    P, mu, cov = wall_model(img)
    d = maha(lab, mu, cov)
    PB = maha(P.reshape(-1, 1, 3), mu, cov).ravel()          # distances of the border pixels themselves
    PRE.append(dict(lab=lab, P=P, mu=mu, cov=cov, d=d, dB=PB))


def to_full(m, img):
    H, W = img.shape[:2]
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def score(masks):
    r = np.array([R.prf(m, g) for m, g in zip(masks, GT)])
    tp = sum(np.sum(m & g) for m, g in zip(masks, GT)); fp = sum(np.sum(m & ~g) for m, g in zip(masks, GT))
    fn = sum(np.sum(~m & g) for m, g in zip(masks, GT))
    P, Rc = tp / (tp + fp), tp / (tp + fn)
    return dict(P=float(r[:, 0].mean()), R=float(r[:, 1].mean()), F1=float(r[:, 2].mean()), F1_min=float(r[:, 2].min()),
                n_below_09=int((r[:, 2] < 0.9).sum()), per_image=r[:, 2].round(4).tolist(),
                pooled=dict(P=float(P), R=float(Rc), F1=float(2 * P * Rc / (P + Rc))))


def run(name, fg_fn, steps=("morph", "largest", "holes", "rowcol")):
    masks, thr = [], []
    for img, pre in zip(FULL, PRE):
        fg, t = fg_fn(pre)
        thr.append(t)
        masks.append(to_full(post(fg, steps), img))
    s = score(masks)
    s["thr_range"] = [float(np.min(thr)), float(np.max(thr))]
    print(f"{name:44s} P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} min={s['F1_min']:.3f} #<0.9={s['n_below_09']:2d}"
          f"  pooledF1={s['pooled']['F1']:.3f}  thr {s['thr_range'][0]:.1f}-{s['thr_range'][1]:.1f}", flush=True)
    return s, masks


def otsu_on_log(pre):
    t = np.expm1(R.otsu_threshold(np.log1p(pre["d"])))
    return pre["d"] > t, t


D = {}
print("== thresholds (same wall model, same post-processing)")
thr = {}
thr["Otsu on log(1+d) (final)"], FINAL = run("Otsu", otsu_on_log)
thr["Otsu on d (no log)"], _ = run("Otsu on raw d", lambda p: (p["d"] > R.otsu_threshold(p["d"]), R.otsu_threshold(p["d"])))
for k in (3, 5, 8):
    thr[f"fixed d > {k}"], _ = run(f"fixed d > {k}", lambda p, k=k: (p["d"] > k, k))
for q in (95, 99):
    thr[f"border percentile {q}"], _ = run(f"border pct {q}", lambda p, q=q: (p["d"] > np.percentile(p["dB"], q), np.percentile(p["dB"], q)))


def m1(p, T):     # M1: Euclidean distance to the median wall colour, fixed T (Lab units)
    d = np.linalg.norm(p["lab"] - np.median(p["P"], 0), axis=2)
    return d > T, T


for T in (15, 25, 35):
    thr[f"M1 Euclidean, T = {T}"], _ = run(f"M1 euclid T={T}", lambda p, T=T: m1(p, T))


def m3(p):        # M3: 3-component GMM of the border, background if as likely as 99% of the border pixels
    from sklearn.mixture import GaussianMixture
    P = p["P"][np.random.default_rng(0).choice(len(p["P"]), min(len(p["P"]), 4000), replace=False)]
    g = GaussianMixture(3, covariance_type="full", reg_covar=1.0, random_state=0).fit(P)
    nll = -g.score_samples(p["lab"].reshape(-1, 3)).reshape(p["lab"].shape[:2])
    t = np.percentile(-g.score_samples(p["P"]), 99)
    return nll > t, t


thr["M3 GMM (3 Gaussians)"], _ = run("M3 GMM", m3)
D["thresholds"] = thr

# ----------------------------------------------------------------------------- wall statistics
sd = np.array([np.sqrt(np.diag(p["cov"] - np.eye(3))) for p in PRE])
D["wall_sigma"] = dict(L=float(np.median(sd[:, 0])), a=float(np.median(sd[:, 1])), b=float(np.median(sd[:, 2])))
D["otsu_sigma_range"] = thr["Otsu on log(1+d) (final)"]["thr_range"]
print("wall sigma (median over images) L/a/b:", D["wall_sigma"])

# ----------------------------------------------------------------------------- post-processing ablation
print("== post-processing")
pp = {}
for name, steps in [("raw threshold", ()), ("+ closing & opening", ("morph",)), ("+ largest component", ("morph", "largest")),
                    ("+ fill holes", ("morph", "largest", "holes")), ("+ row/column fill (final)", ("morph", "largest", "holes", "rowcol"))]:
    pp[name], _ = run(name, otsu_on_log, steps)
D["postproc"] = pp
h, w = PRE[0]["lab"].shape[:2]
D["se"] = dict(close=max(1, int(0.03 * min(h, w))) | 1, open=max(1, int(0.02 * min(h, w))) | 1, work=[h, w])

# ----------------------------------------------------------------------------- shadows
print("== shadows")
GT_SMALL = [cv2.resize(g.astype(np.uint8), p["d"].shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for g, p in zip(GT, PRE)]
shadow = {}
for q in (0, 14):
    p, g = PRE[q], GT_SMALL[q]
    fg, t = otsu_on_log(p)
    raw_fp = fg & ~g
    fin = post(fg)
    fp = fin & ~g
    dL = (p["lab"][fp] - p["mu"])[:, 0]
    best = max((R.prf(to_full(post(p["d"] > tt), FULL[q]), GT[q])[2], float(tt)) for tt in np.exp(np.linspace(0, np.log(p["d"].max()), 120)))
    shadow[q] = dict(F1=R.prf(to_full(fin, FULL[q]), GT[q])[2], fp_born_at_threshold=float((raw_fp & fp).sum() / max(1, fp.sum())),
                     dL_median=float(np.median(dL)) if len(dL) else 0.0, best_thr=best[1], best_F1=best[0], otsu_thr=float(t),
                     fp_area=float(fp.sum() / g.sum()))
    print(q, shadow[q])
D["shadow_cases"] = shadow


def otsu3_upper(v, nbins=128):
    hist, e = np.histogram(v, nbins); p = hist / hist.sum(); c = (e[:-1] + e[1:]) / 2
    Pc = np.concatenate([[0], np.cumsum(p)]); M = np.concatenate([[0], np.cumsum(p * c)]); mt = M[-1]
    best, bt = -1, None
    for i in range(1, nbins - 1):
        for j in range(i + 1, nbins):
            w0, w1, w2 = Pc[i], Pc[j] - Pc[i], 1 - Pc[j]
            if min(w0, w1, w2) <= 0:
                continue
            m0, m1_, m2 = M[i] / w0, (M[j] - M[i]) / w1, (mt - M[j]) / w2
            s = w0 * (m0 - mt) ** 2 + w1 * (m1_ - mt) ** 2 + w2 * (m2 - mt) ** 2
            if s > best:
                best, bt = s, e[j]
    return bt


def sigmaL(p, k):
    S = np.diag([k, 1, 1]); d = maha(p["lab"], p["mu"], S @ (p["cov"]) @ S)
    t = np.expm1(R.otsu_threshold(np.log1p(d)))
    return d > t, t


rules = {}
base = np.array(thr["Otsu on log(1+d) (final)"]["per_image"])
for name, fn in [("Otsu + 0.3 (stricter)", lambda p: (np.log1p(p["d"]) > R.otsu_threshold(np.log1p(p["d"])) + 0.3, 0)),
                 ("Otsu + 0.5 (stricter)", lambda p: (np.log1p(p["d"]) > R.otsu_threshold(np.log1p(p["d"])) + 0.5, 0)),
                 ("3-class Otsu, upper cut", lambda p: (np.log1p(p["d"]) > otsu3_upper(np.log1p(p["d"]).ravel()), 0)),
                 ("brightness weighs 3x less", lambda p: sigmaL(p, 3))]:
    s, _ = run(name, fn)
    pi = np.array(s["per_image"])
    s["fixes_q0"] = float(pi[0]); s["fixes_q14"] = float(pi[14]); s["n_worse"] = int((pi < base - 0.02).sum())
    rules[name] = s
D["shadow_rules"] = rules

# ----------------------------------------------------------------------------- final masks: per image and Task 4
D["final"] = thr["Otsu on log(1+d) (final)"]
json.dump(D, open(HERE / "data_seg.json", "w"), indent=1)
print("saved")
