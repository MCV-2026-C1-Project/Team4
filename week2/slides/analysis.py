"""
Numbers behind the week-2 slides. Writes slides/data.json.

Run from the week2 folder:  python slides/analysis.py
Uses the same functions as run_w2.py (descriptor, mask), so the slides show exactly what is submitted.
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_w2 as R  # noqa: E402

OUT = HERE / "data.json"
CONV = {"YCRCB": cv2.COLOR_BGR2YCrCb, "HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB, "RGB": cv2.COLOR_BGR2RGB}
UNIFORM = {"HSV": [(0, 180), (0, 256), (0, 256)]}
LUM = {"YCRCB": 0, "LAB": 0, "HSV": 2}          # brightness channel; the other two are the chroma channels
DEFAULT_SCHEME = {"YCRCB": "range", "LAB": "range", "HSV": "uniform", "RGB": "uniform"}

# ----------------------------------------------------------------------------- data
bb_paths = R.image_paths(HERE.parent.parent / "week1" / "BBDD", "bbdd_*.jpg")
BB_IDS = [R.image_id(p) for p in bb_paths]
BB = [R.resize_max(cv2.imread(str(p))) for p in bb_paths]
q1, q2 = R.find_dir(HERE.parent / "qsd1_w2"), R.find_dir(HERE.parent / "qsd2_w2")
G1 = [g[0] for g in pickle.load(open(q1 / "gt_corresps.pkl", "rb"))]
G2 = [g[0] for g in pickle.load(open(q2 / "gt_corresps.pkl", "rb"))]
QSD1 = [R.resize_max(cv2.imread(str(p))) for p in R.image_paths(q1)]
FULL2 = [cv2.imread(str(p)) for p in R.image_paths(q2)]
GTM2 = [cv2.imread(str(p.with_suffix(".png")), 0) > 0 for p in R.image_paths(q2)]
PRED2 = [R.compute_mask(im) for im in FULL2]
QSD2_GT = [R.crop_to_mask(im, m) for im, m in zip(FULL2, GTM2)]
QSD2_PRED = [R.crop_to_mask(im, m) for im, m in zip(FULL2, PRED2)]


def tight(img, gt, pad=15):
    x, y, w, h = cv2.boundingRect(gt.astype(np.uint8))
    y0, x0 = max(0, y - pad), max(0, x - pad)
    return img[y0:y + h + pad, x0:x + w + pad], gt[y0:y + h + pad, x0:x + w + pad]


TIGHT = [tight(im, g) for im, g in zip(FULL2, GTM2)]
QSD2_15 = [R.crop_to_mask(im, R.compute_mask(im)) for im, _ in TIGHT]
SETS = {"QSD1": (QSD1, G1), "QSD2-GT": (QSD2_GT, G2), "QSD2-pred": (QSD2_PRED, G2), "QSD2-15px": (QSD2_15, G2)}


# ----------------------------------------------------------------------------- generic descriptor
def cfg(**kw):
    c = dict(space="YCRCB", kind="3d", bins=(6, 12, 12), scheme=None, levels=(1, 2, 4, 8), weights=(1, 2, 4, 8),
             margin=0.05, norm=True)
    c.update(kw)
    if c["scheme"] is None:
        c["scheme"] = DEFAULT_SCHEME[c["space"]]
    return c


def prep(img, c):
    x = R.inner_crop(img, c["margin"])
    if c["norm"]:
        x = R.channel_mean_128(x)
    return cv2.cvtColor(x, CONV[c["space"]])


_edge_cache = {}


def channel_edges(c, k, n):
    key = (c["space"], c["scheme"], c["margin"], c["norm"], k, n)
    if key not in _edge_cache:
        if c["scheme"] == "uniform":
            lo, hi = UNIFORM.get(c["space"], [(0, 256)] * 3)[k]
            _edge_cache[key] = np.linspace(lo, hi, n + 1)
        else:
            P = np.concatenate([prep(x, c)[..., k].ravel() for x in BB])
            P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)].astype(np.float64)
            lo, hi = np.percentile(P, [0.5, 99.5])
            e = np.linspace(lo, hi, n + 1); e[0], e[-1] = -1, 256
            _edge_cache[key] = e
    return _edge_cache[key]


def lut(e):
    return np.clip(np.searchsorted(e[1:-1], np.arange(256), side="right"), 0, len(e) - 2)


def block_hists(idx, nbins, levels, weights):
    h, w = idx.shape
    wt = np.asarray(weights, float) / np.sum(weights)
    out = []
    for n, wl in zip(levels, wt):
        ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
        for i in range(n):
            for j in range(n):
                hb = np.bincount(idx[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel(), minlength=nbins).astype(np.float32)
                out.append(wl * hb / (hb.sum() + 1e-10) / (n * n))
    return out


def describe(img, c):
    x = prep(img, c)
    kind, b = c["kind"], c["bins"]
    if kind == "3d":
        q = [lut(channel_edges(c, k, b[k]))[x[..., k]] for k in range(3)]
        idx = q[0] * b[1] * b[2] + q[1] * b[2] + q[2]
        return np.concatenate(block_hists(idx, b[0] * b[1] * b[2], c["levels"], c["weights"]))
    parts = []
    if kind == "1d":
        for k in range(3):
            parts.append(np.concatenate(block_hists(lut(channel_edges(c, k, b[k]))[x[..., k]], b[k], c["levels"], c["weights"])) / 3)
    else:   # "2d" chroma only, or "2d1d" = 2D chroma + 1D brightness
        ch = [k for k in range(3) if k != LUM[c["space"]]]
        q = [lut(channel_edges(c, k, b[k]))[x[..., k]] for k in ch]
        parts.append(np.concatenate(block_hists(q[0] * b[ch[1]] + q[1], b[ch[0]] * b[ch[1]], c["levels"], c["weights"])))
        if kind == "2d1d":
            L = LUM[c["space"]]
            parts.append(np.concatenate(block_hists(lut(channel_edges(c, L, b[L]))[x[..., L]], b[L], c["levels"], c["weights"])))
            parts = [p / 2 for p in parts]
    return np.concatenate(parts)


def evaluate(c, sets=("QSD1", "QSD2-GT", "QSD2-pred", "QSD2-15px"), return_ranks=False):
    DB = np.stack([describe(x, c) for x in BB])
    res, ranks = {}, {}
    for s in sets:
        imgs, gt = SETS[s]
        rk = []
        for im, g in zip(imgs, gt):
            order = np.argsort(R.hellinger(describe(im, c), DB), kind="stable")
            rk.append([BB_IDS[i] for i in order].index(g) + 1)
        rk = np.array(rk)
        res[s] = dict(map1=float((rk == 1).mean()), map5=float(np.where(rk <= 5, 1 / rk, 0).mean()), mrr=float((1 / rk).mean()))
        ranks[s] = rk.tolist()
    dev = [r for s in ("QSD1", "QSD2-GT") if s in sets for r in ranks[s]]
    if dev:
        dev = np.array(dev)
        res["dev60"] = dict(map1=float((dev == 1).mean()), map5=float(np.where(dev <= 5, 1 / dev, 0).mean()), mrr=float((1 / dev).mean()))
    return (res, ranks) if return_ranks else res


def log(name, r):
    print(f"{name:48s} " + "  ".join(f"{k}={v['map1']:.3f}/{v['mrr']:.3f}" for k, v in r.items()), flush=True)


DATA = {}
DEV = ("QSD1", "QSD2-GT")
GLOBAL = dict(levels=(1,), weights=(1,))

# ----------------------------------------------------------------------------- T1.1 histogram type x colour space (global)
print("== T1.1 type x space (global, best bins)")
BINS = {"1d": [8, 16, 32, 64], "2d": [8, 16, 32], "2d1d": [8, 16, 32], "3d": [4, 6, 8, 12, 16]}
types = {}
for space in ("HSV", "LAB", "YCRCB", "RGB"):
    for kind in ("1d", "2d", "2d1d", "3d"):
        if space == "RGB" and kind in ("2d", "2d1d"):
            continue
        best = None
        for n in BINS[kind]:
            r = evaluate(cfg(space=space, kind=kind, bins=(n, n, n), **GLOBAL), DEV)
            if best is None or (r["dev60"]["map1"], r["dev60"]["mrr"]) > (best[0]["dev60"]["map1"], best[0]["dev60"]["mrr"]):
                best = (r, n)
        types[f"{space}/{kind}"] = dict(bins=best[1], **best[0]["dev60"])
        log(f"{space} {kind} best bins={best[1]}", {"dev60": best[0]["dev60"]})
DATA["types"] = types

# ----------------------------------------------------------------------------- T1.1 bins curve, uniform vs range (3D global)
print("== T1.1 bins curve")
curve = {}
for space, schemes in (("YCRCB", ("uniform", "range")), ("LAB", ("uniform", "range")), ("HSV", ("uniform",))):
    for sch in schemes:
        pts = []
        for n in (2, 3, 4, 6, 8, 12, 16, 24):
            r = evaluate(cfg(space=space, scheme=sch, bins=(n, n, n), **GLOBAL), DEV)["dev60"]
            pts.append(dict(n=n, dim=n ** 3, **r))
        curve[f"{space}/{sch}"] = pts
        print(space, sch, [(p["n"], round(p["map1"], 3), round(p["mrr"], 3)) for p in pts], flush=True)
DATA["bins_curve"] = curve

# bins actually used by the BBDD images (3D 8x8x8): uniform vs range
used = {}
for space in ("YCRCB", "LAB"):
    for sch in ("uniform", "range"):
        c = cfg(space=space, scheme=sch, bins=(8, 8, 8), **GLOBAL)
        used[f"{space}/{sch}"] = float(np.mean([(describe(x, c) > 0).sum() for x in BB]))
print("bins used per painting (of 512):", used)
DATA["bins_used"] = used
# chroma range of the BBDD (after normalisation)
c = cfg(space="YCRCB", scheme="range")
P = np.concatenate([prep(x, c).reshape(-1, 3) for x in BB])
P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)]
DATA["ycrcb_pct"] = {ch: [float(v) for v in np.percentile(P[:, k], [0.5, 99.5])] for k, ch in enumerate("Y Cr Cb".split())}
print("YCrCb 0.5-99.5 pct:", DATA["ycrcb_pct"])

# ----------------------------------------------------------------------------- T1.2 number of blocks, 1D vs 3D
print("== T1.2 blocks curve")
blocks = {}
for kind, bins in (("1d", (16, 16, 16)), ("3d", (6, 12, 12))):
    pts = []
    for n in (1, 2, 3, 4, 5, 6, 8):
        r = evaluate(cfg(kind=kind, bins=bins, levels=(n,), weights=(1,)), DEV)["dev60"]
        pts.append(dict(n=n, **r))
    r = evaluate(cfg(kind=kind, bins=bins), DEV)["dev60"]
    pts.append(dict(n="pyramid", **r))
    blocks[kind] = pts
    print(kind, [(p["n"], round(p["map1"], 3), round(p["mrr"], 3)) for p in pts], flush=True)
DATA["blocks_curve"] = blocks

# example for the blocks slide: queries that the global 3D histogram gets wrong and the pyramid gets right
print("== T1.2 examples")
cg = cfg(**GLOBAL); cf = cfg()
_, rg = evaluate(cg, DEV, return_ranks=True)
_, rf = evaluate(cf, DEV, return_ranks=True)
DBg = np.stack([describe(x, cg) for x in BB])
cands = []
for s in DEV:
    imgs, gt = SETS[s]
    for i, (a, b) in enumerate(zip(rg[s], rf[s])):
        if a > 1 and b == 1:
            top = BB_IDS[int(np.argmin(R.hellinger(describe(imgs[i], cg), DBg)))]
            cands.append(dict(set=s, q=i, gt=gt[i], rank_global=a, rank_final=b, confuser=top))
print(cands)
DATA["block_examples"] = cands

# ----------------------------------------------------------------------------- T1.3 leave-one-out ablation
print("== T1.3 ablation")
abl = {}
for name, over in [("final", {}), ("no light normalisation", dict(norm=False)), ("no blocks (global)", GLOBAL),
                   ("1D instead of 3D", dict(kind="1d", bins=(16, 16, 16))), ("uniform bins", dict(scheme="uniform")),
                   ("no pyramid (8x8 only)", dict(levels=(8,), weights=(1,))), ("no 5% margin", dict(margin=0.0)),
                   ("HSV instead of YCrCb", dict(space="HSV")), ("Lab instead of YCrCb", dict(space="LAB")),
                   ("RGB instead of YCrCb", dict(space="RGB"))]:
    abl[name] = evaluate(cfg(**over))
    log(name, abl[name])
DATA["ablation"] = abl

# W1 descriptor (HSV 1D 16 bins, chi2 in W1; here Hellinger for a common comparison) for the Task 2 / Task 5 context
DATA["w1_like"] = evaluate(cfg(space="HSV", kind="1d", bins=(16, 16, 16), margin=0.0, **GLOBAL))
log("W1-like HSV 1D 16", DATA["w1_like"])

# ----------------------------------------------------------------------------- T5 retrieval with our masks
print("== T5")
t5 = {}
for name, over in [("8x8, no margin", dict(levels=(8,), weights=(1,), margin=0.0)),
                   ("8x8 + 5% margin", dict(levels=(8,), weights=(1,))),
                   ("pyramid, no margin", dict(margin=0.0)),
                   ("pyramid + 5% margin (final)", {})]:
    r, rk = evaluate(cfg(**over), ("QSD2-GT", "QSD2-pred", "QSD2-15px"), return_ranks=True)
    t5[name] = dict(res=r, ranks=rk)
    log(name, r)
DATA["t5"] = t5

# mechanism: how much wall each predicted crop carries and how it moves the normalisation
mech = []
for i, (im, pm, gm) in enumerate(zip(FULL2, PRED2, GTM2)):
    x, y, w, h = cv2.boundingRect(pm.astype(np.uint8))
    wall = float((~gm[y:y + h, x:x + w]).mean())
    mp = im[y:y + h, x:x + w].reshape(-1, 3).mean(0)
    xg, yg, wg, hg = cv2.boundingRect(gm.astype(np.uint8))
    mg = im[yg:yg + hg, xg:xg + wg].reshape(-1, 3).mean(0)
    mech.append(dict(q=i, wall_frac=wall, mean_pred=mp.tolist(), mean_gt=mg.tolist(),
                     gain_ratio=float(np.mean(mg / mp))))
DATA["t5_mech"] = mech
print("wall fraction in predicted crops: mean %.3f max %.3f" % (np.mean([m["wall_frac"] for m in mech]), np.max([m["wall_frac"] for m in mech])))

json.dump(DATA, open(OUT, "w"), indent=1)
print("saved", OUT)
