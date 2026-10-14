"""
C1 - Week 2 - Task 6: predictions for the blind test sets QST1 and QST2.

Descriptor (both methods): illumination normalisation (each RGB channel scaled to mean 128) on the image without a 5%
inner margin -> 3D colour histogram per block -> concatenation of the blocks -> Hellinger kernel.
    method1: YCrCb, 6x12x12 bins adapted to the real range of each channel, spatial pyramid 1x1+2x2+4x4+8x8
             (level weights 1:2:4:8, finer levels weigh more)
    method2: HSV, 8x6x6 bins (uniform), 5x5 blocks

Background removal for QST2 (colour only): Gaussian model of the wall colour (robust mean / covariance in Lab of the
15-pixel strip along the border of the full-resolution image, which the organisers guarantee to be wall) -> Mahalanobis
distance -> Otsu threshold -> morphology, largest component, hole filling, row/column filling. The painting is then
cropped with the bounding box of the mask.

The week-2 instructions ask for a single method, so only method1 is submitted by default (method2 is the alternative we
compared it with; use --all-methods to also generate it).

Outputs (list of lists of int, one list of K=10 BBDD ids per query, in file order):
    results/QST1/method1/result.pkl
    results/QST2/method1/result.pkl
    results/QST2/method1/*.png        (masks, 0/255, same name as the query)

Usage (from the week2 folder; the BBDD is read from ../week1/BBDD):
    python run_w2.py --check                       # evaluate on QSD1 and QSD2 first (mAP and mask P/R/F1)
    python run_w2.py                               # generate the QST1 / QST2 predictions
    python run_w2.py --qst1 qst1_w2 --qst2 qst2_w2 # if the test folders have other names
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

K = 10
MAX_SIDE = 512
WORK_SIDE = 400
BORDER_PX = 15   # the organisers guarantee at least 15 px of wall on every side of the painting
EPS = 1e-10

SPACES = {"HSV": (cv2.COLOR_BGR2HSV, [(0, 180), (0, 256), (0, 256)]), "YCRCB": (cv2.COLOR_BGR2YCrCb, [(0, 256)] * 3)}
METHODS = {
    "method1": dict(space="YCRCB", bins=(6, 12, 12), scheme="range", levels=(1, 2, 4, 8), weights=(1, 2, 4, 8), margin=0.05),
    "method2": dict(space="HSV", bins=(8, 6, 6), scheme="uniform", levels=(5,), weights=(1,), margin=0.05),
}


# ----------------------------------------------------------------------------- data
def image_id(path):
    return int(Path(path).stem.split("_")[-1])


def resize_max(img, max_side=MAX_SIDE):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def image_paths(folder, pattern="*.jpg"):
    paths = sorted(Path(folder).glob(pattern), key=image_id)
    if not paths:
        raise FileNotFoundError(f"No '{pattern}' images in {folder}")
    return paths


# ----------------------------------------------------------------------------- descriptor
def inner_crop(img, margin):
    h, w = img.shape[:2]
    dy, dx = int(margin * h), int(margin * w)
    return img[dy:h - dy, dx:w - dx]


def channel_mean_128(img):
    f = img.astype(np.float32)
    return np.clip(f * (128 / (f.reshape(-1, 3).mean(0) + EPS)), 0, 255).astype(np.uint8)


def prepare(img, m):
    x = channel_mean_128(inner_crop(img, m["margin"]))
    return cv2.cvtColor(x, SPACES[m["space"]][0])


def make_edges(m, bbdd_imgs):
    if m["scheme"] == "uniform":
        return [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(SPACES[m["space"]][1], m["bins"])]
    # 'range': equal-width bins over the 0.5-99.5 percentile range of the BBDD pixels of each channel
    P = np.concatenate([prepare(x, m).reshape(-1, 3) for x in bbdd_imgs])
    P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)].astype(np.float64)
    edges = []
    for c, n in enumerate(m["bins"]):
        lo, hi = np.percentile(P[:, c], [0.5, 99.5])
        e = np.linspace(lo, hi, n + 1); e[0], e[-1] = -1, 256
        edges.append(e)
    return edges


def pyramid_hist_3d(x, edges, levels, weights):
    """3D histogram of every block of every level (n x n grids), each level scaled by its weight, concatenated."""
    luts = [np.clip(np.searchsorted(e[1:-1], np.arange(256), side="right"), 0, len(e) - 2) for e in edges]
    nb = [len(e) - 1 for e in edges]
    idx = luts[0][x[..., 0]] * nb[1] * nb[2] + luts[1][x[..., 1]] * nb[2] + luts[2][x[..., 2]]
    h, w = idx.shape
    weights = np.asarray(weights, float) / np.sum(weights)
    out = []
    for n, wl in zip(levels, weights):
        ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
        for i in range(n):
            for j in range(n):
                hb = np.bincount(idx[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel(), minlength=nb[0] * nb[1] * nb[2]).astype(np.float32)
                out.append(wl * hb / (hb.sum() + EPS) / (n * n))
    return np.concatenate(out)


def describe(img, m):
    return pyramid_hist_3d(prepare(img, m), m["edges"], m["levels"], m["weights"])


def hellinger(q, D):
    return -np.sqrt(D * q).sum(1)


def retrieve(query_imgs, db_desc, db_ids, m, k=K):
    out = []
    for img in query_imgs:
        order = np.argsort(hellinger(describe(img, m), db_desc), kind="stable")[:k]
        out.append([int(db_ids[i]) for i in order])
    return out


# ----------------------------------------------------------------------------- background removal (colour only)
def otsu_threshold(v, nbins=256):
    hist, edges = np.histogram(v, nbins)
    p, c = hist / hist.sum(), (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    m1 = ((p * c).sum() - np.cumsum(p * c)) / np.maximum(w1, 1e-12)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def border_pixels(img, t=BORDER_PX):
    """Lab pixels of the t-pixel strip along the 4 borders of the full-resolution image (guaranteed to be wall)."""
    lab = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2LAB)
    strip = np.zeros(lab.shape[:2], bool)
    strip[:t], strip[-t:], strip[:, :t], strip[:, -t:] = True, True, True, True
    return lab[strip].astype(np.float64)


def compute_mask(img, border_px=BORDER_PX, keep=0.8):
    H, W = img.shape[:2]
    lab = cv2.cvtColor(cv2.medianBlur(resize_max(img, WORK_SIDE), 5), cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w = lab.shape[:2]
    P = border_pixels(img, border_px)                              # sampled at full resolution, not on the 400 px image
    d = np.linalg.norm(P - np.median(P, 0), axis=1)
    Pk = P[d <= np.quantile(d, keep)]                              # robust: dominant colour of the border = wall
    mu, cov = Pk.mean(0), np.cov(Pk.T) + np.eye(3)
    diff = lab.reshape(-1, 3) - mu
    ld = np.log1p(np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff))).reshape(h, w)
    m = (ld > otsu_threshold(ld)).astype(np.uint8)
    # post-processing of the binary mask
    s = min(h, w)
    se = lambda f: cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(f * s)) | 1,) * 2)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, se(0.03)), cv2.MORPH_OPEN, se(0.02))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    m = (pad[1:-1, 1:-1] != 2)                                     # painting + holes
    rows, cols = np.zeros_like(m), np.zeros_like(m)
    for y in np.where(m.any(1))[0]:
        xs_ = np.where(m[y])[0]; rows[y, xs_[0]:xs_[-1] + 1] = True
    for x in np.where(m.any(0))[0]:
        ys_ = np.where(m[:, x])[0]; cols[ys_[0]:ys_[-1] + 1, x] = True
    m = rows & cols
    if not m.any():                                                # nothing found: keep the whole image rather than crash
        m[:] = True
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def crop_to_mask(img, mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return resize_max(img[y:y + h, x:x + w])


# ----------------------------------------------------------------------------- evaluation
def apk(actual, predicted, k=10):
    predicted = predicted[:k]
    score, hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(actual, predicted, k=10):
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def prf(pred, gt):
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def save_pkl(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def check_format(preds, n):
    assert len(preds) == n and all(len(p) == K and all(type(i) is int for i in p) for p in preds)


# ----------------------------------------------------------------------------- main
def find_dir(*names):
    for n in map(Path, names):
        if n.is_dir():
            if any(n.glob("*.jpg")):
                return n
            for sub in sorted(p for p in n.iterdir() if p.is_dir()):
                if any(sub.glob("*.jpg")):
                    return sub
    return None


def main():
    ap = argparse.ArgumentParser(description="C1 week 2: QST1 / QST2 predictions")
    ap.add_argument("--bbdd", default="../week1/BBDD")
    ap.add_argument("--qsd1", default="qsd1_w2")
    ap.add_argument("--qsd2", default="qsd2_w2")
    ap.add_argument("--qst1", default="qst1_w2")
    ap.add_argument("--qst2", default="qst2_w2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--check", action="store_true", help="evaluate on QSD1 / QSD2 before predicting")
    ap.add_argument("--all-methods", action="store_true", help="also generate method2 (by default only method1 is submitted)")
    args = ap.parse_args()
    submit = list(METHODS) if args.all_methods else ["method1"]

    print("Loading BBDD ...")
    bb_paths = image_paths(args.bbdd, "bbdd_*.jpg")
    bb_ids = [image_id(p) for p in bb_paths]
    bb_imgs = [resize_max(cv2.imread(str(p))) for p in bb_paths]
    db = {}
    for name, m in METHODS.items():
        m["edges"] = make_edges(m, bb_imgs)
        db[name] = np.stack([describe(x, m) for x in bb_imgs])
    print(f"  {len(bb_imgs)} paintings, descriptors: " + ", ".join(f"{k} {v.shape[1]}-D" for k, v in db.items()))

    if args.check:
        qsd1, qsd2 = find_dir(args.qsd1), find_dir(args.qsd2)
        if qsd1:
            gt = pickle.load(open(qsd1 / "gt_corresps.pkl", "rb"))
            imgs = [resize_max(cv2.imread(str(p))) for p in image_paths(qsd1)]
            for name, m in METHODS.items():
                pr = retrieve(imgs, db[name], bb_ids, m)
                print(f"QSD1 {name}: mAP@1={mapk(gt, pr, 1):.3f}  mAP@5={mapk(gt, pr, 5):.3f}")
        if qsd2:
            gt = pickle.load(open(qsd2 / "gt_corresps.pkl", "rb"))
            crops, scores = [], []
            for p in image_paths(qsd2):
                img = cv2.imread(str(p)); mask = compute_mask(img)
                scores.append(prf(mask, cv2.imread(str(p.with_suffix(".png")), 0) > 0))
                crops.append(crop_to_mask(img, mask))
            P, R, F = np.mean(scores, 0)
            print(f"QSD2 masks: precision={P:.3f}  recall={R:.3f}  F1={F:.3f}")
            for name, m in METHODS.items():
                pr = retrieve(crops, db[name], bb_ids, m)
                print(f"QSD2 {name}: mAP@1={mapk(gt, pr, 1):.3f}  mAP@5={mapk(gt, pr, 5):.3f}")

    out = Path(args.out)
    qst1 = find_dir(args.qst1)
    if qst1:
        imgs = [resize_max(cv2.imread(str(p))) for p in image_paths(qst1)]
        for name in submit:
            pr = retrieve(imgs, db[name], bb_ids, METHODS[name]); check_format(pr, len(imgs))
            save_pkl(pr, out / "QST1" / name / "result.pkl")
        print(f"QST1: {len(imgs)} queries -> {out / 'QST1'}/{'/, '.join(submit)}/result.pkl")
    else:
        print(f"'{args.qst1}' not found: QST1 skipped")

    qst2 = find_dir(args.qst2)
    if qst2:
        crops = []
        for p in image_paths(qst2):
            img = cv2.imread(str(p)); mask = compute_mask(img)
            for name in submit:
                (out / "QST2" / name).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "QST2" / name / f"{p.stem}.png"), mask.astype(np.uint8) * 255)
            crops.append(crop_to_mask(img, mask))
        for name in submit:
            pr = retrieve(crops, db[name], bb_ids, METHODS[name]); check_format(pr, len(crops))
            save_pkl(pr, out / "QST2" / name / "result.pkl")
        print(f"QST2: {len(crops)} queries -> {out / 'QST2'}/{'/, '.join(submit)}/result.pkl + masks")
    else:
        print(f"'{args.qst2}' not found: QST2 skipped")


if __name__ == "__main__":
    main()
