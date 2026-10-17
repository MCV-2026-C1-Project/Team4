"""
C1 - Week 2 - Task 6: predictions for the blind test sets QST1 and QST2 (one method, method1).

Retrieval descriptor:
    1. remove a 5% inner margin on each side of the image (robust to imperfect crops)
    2. illumination normalisation: each RGB channel scaled to mean 128
    3. YCrCb, 3D colour histogram with 6x12x12 bins placed over the real range of each channel
       (equal-width bins between the 0.5 and 99.5 percentiles of the BBDD pixels)
    4. spatial pyramid: blocks 1x1 + 2x2 + 4x4 + 8x8, each block normalised, level weights 1:2:4:8
    5. Hellinger kernel to rank the 287 BBDD paintings

Background removal for QST2 (colour only):
    Gaussian model of the wall colour (robust mean / covariance in Lab of the 15-px strip along the border of the
    full-resolution image: the professors mentioned in class that there are at least 15 px of wall around every
    painting) -> Mahalanobis distance -> Otsu threshold ->
    closing + opening, largest component, hole filling, row/column filling. The painting is then cropped with the
    bounding box of the mask and described as above.

Outputs (format of the week-2 instructions):
    results/QST1/method1/result.pkl     list of lists of int: the K=10 best BBDD ids per query, in file order
    results/QST2/method1/result.pkl     same for QST2
    results/QST2/method1/*.png          binary masks (0 / 255), same name as the query (00001.jpg -> 00001.png)

The datasets must be in this folder: BBDD/, qst1_w2/, qst2_w2/ (and qsd1_w2/, qsd2_w2/ for --check).

Usage (from this folder):
    python run_w2.py --check     # results on QSD1 / QSD2 (mAP and mask precision / recall / F1) + test predictions
    python run_w2.py             # test predictions only
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

K = 10              # results per query required by the submission
MAX_SIDE = 512      # images are resized so that their longest side is at most 512 px before describing them
WORK_SIDE = 400     # background removal works on a copy with longest side 400 px
BORDER_PX = 15      # the professors mentioned in class at least 15 px of wall around every painting
MARGIN = 0.05       # inner margin removed on each side before describing an image
BINS = (6, 12, 12)  # Y, Cr, Cb
LEVELS, WEIGHTS = (1, 2, 4, 8), (1, 2, 4, 8)
EPS = 1e-10


# ----------------------------------------------------------------------------- data
def image_id(path):
    # 'bbdd_00120.jpg' -> 120, '00007.jpg' -> 7
    return int(Path(path).stem.split("_")[-1])


def image_paths(folder, pattern="*.jpg"):
    paths = sorted(Path(folder).glob(pattern), key=image_id)
    if not paths:
        raise FileNotFoundError(f"No '{pattern}' images found in {folder}")
    return paths


def resize_max(img, max_side=MAX_SIDE):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def find_dir(*candidates):
    """First candidate folder that holds .jpg images (directly or in a sub-folder, e.g. qsd2_w2/qsd2_w1)."""
    for c in map(Path, candidates):
        if c.is_dir():
            if any(c.glob("*.jpg")):
                return c
            for sub in sorted(p for p in c.iterdir() if p.is_dir()):
                if any(sub.glob("*.jpg")):
                    return sub
    return None


# ----------------------------------------------------------------------------- descriptor
def prepare(img):
    # 5% inner margin -> each RGB channel to mean 128 -> YCrCb
    h, w = img.shape[:2]
    dy, dx = int(MARGIN * h), int(MARGIN * w)
    f = img[dy:h - dy, dx:w - dx].astype(np.float32)
    f = np.clip(f * (128 / (f.reshape(-1, 3).mean(0) + EPS)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)


def range_edges(bbdd_imgs):
    # Equal-width bins between the 0.5 and 99.5 percentiles of the BBDD pixels of each channel;
    # the rare values outside go to the first / last bin
    P = np.concatenate([prepare(x).reshape(-1, 3) for x in bbdd_imgs])
    P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)].astype(np.float64)
    edges = []
    for c, n in enumerate(BINS):
        lo, hi = np.percentile(P[:, c], [0.5, 99.5])
        e = np.linspace(lo, hi, n + 1)
        e[0], e[-1] = -1, 256
        edges.append(e)
    return edges


def describe(img, edges):
    # 3D histogram of every block of every pyramid level, each block normalised, levels weighted, concatenated
    x = prepare(img)
    luts = [np.clip(np.searchsorted(e[1:-1], np.arange(256), side="right"), 0, len(e) - 2) for e in edges]
    idx = luts[0][x[..., 0]] * BINS[1] * BINS[2] + luts[1][x[..., 1]] * BINS[2] + luts[2][x[..., 2]]
    h, w = idx.shape
    weights = np.asarray(WEIGHTS, float) / np.sum(WEIGHTS)
    out = []
    for n, wl in zip(LEVELS, weights):
        ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
        for i in range(n):
            for j in range(n):
                hb = np.bincount(idx[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel(), minlength=int(np.prod(BINS)))
                out.append(wl * hb.astype(np.float32) / (hb.sum() + EPS) / (n * n))
    return np.concatenate(out)


def retrieve(query_imgs, db_desc, db_ids, edges, k=K):
    # Top-k BBDD ids (plain Python ints) for each query, Hellinger kernel (higher = more similar)
    out = []
    for img in query_imgs:
        sim = np.sqrt(db_desc * describe(img, edges)).sum(1)
        order = np.argsort(-sim, kind="stable")[:k]
        out.append([int(db_ids[i]) for i in order])
    return out


# ----------------------------------------------------------------------------- background removal (colour only)
def otsu_threshold(v, nbins=256):
    hist, e = np.histogram(v, nbins)
    p, c = hist / hist.sum(), (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    m1 = ((p * c).sum() - np.cumsum(p * c)) / np.maximum(w1, 1e-12)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def compute_mask(img, keep=0.8):
    """Full-resolution BGR photo -> boolean mask of the painting (same size as the photo)."""
    H, W = img.shape[:2]
    lab = cv2.cvtColor(cv2.medianBlur(resize_max(img, WORK_SIDE), 5), cv2.COLOR_BGR2LAB).astype(np.float64)
    h, w = lab.shape[:2]

    # wall model: robust mean / covariance of the 15-px border strip, taken at full resolution
    strip = np.zeros((H, W), bool)
    t = BORDER_PX
    strip[:t], strip[-t:], strip[:, :t], strip[:, -t:] = True, True, True, True
    P = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2LAB)[strip].astype(np.float64)
    d = np.linalg.norm(P - np.median(P, 0), axis=1)
    Pk = P[d <= np.quantile(d, keep)]                       # the 80% of strip pixels closest to the median colour
    mu, cov = Pk.mean(0), np.cov(Pk.T) + np.eye(3)

    # Mahalanobis distance to the wall, log, Otsu threshold
    diff = lab.reshape(-1, 3) - mu
    ld = np.log1p(np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff))).reshape(h, w)
    m = (ld > otsu_threshold(ld)).astype(np.uint8)

    # post-processing: closing + opening, largest component, fill holes, row / column fill
    s = min(h, w)
    se = lambda f: cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(f * s)) | 1,) * 2)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, se(0.03)), cv2.MORPH_OPEN, se(0.02))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    m = pad[1:-1, 1:-1] != 2                                # painting + its holes
    rows, cols = np.zeros_like(m), np.zeros_like(m)
    for y in np.where(m.any(1))[0]:
        xs = np.where(m[y])[0]
        rows[y, xs[0]:xs[-1] + 1] = True
    for x in np.where(m.any(0))[0]:
        ys = np.where(m[:, x])[0]
        cols[ys[0]:ys[-1] + 1, x] = True
    m = rows & cols
    if not m.any():                                         # nothing found: keep the whole image
        m[:] = True
    return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def crop_to_mask(img, mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return resize_max(img[y:y + h, x:x + w])


# ----------------------------------------------------------------------------- evaluation (for --check)
def apk(actual, predicted, k=10):
    # Average precision at k (benhamner/Metrics)
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
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def save_result(preds, n_queries, path):
    # Format required by the instructions: list of lists of int, one list of K ids per query
    assert len(preds) == n_queries
    assert all(len(p) == K and all(type(i) is int for i in p) for p in preds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preds, f)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="C1 week 2: QST1 / QST2 predictions (method1)")
    ap.add_argument("--bbdd", default="BBDD", help="museum database folder (default: BBDD)")
    ap.add_argument("--qsd1", default=None, help="development set 1 (default: qsd1_w2)")
    ap.add_argument("--qsd2", default=None, help="development set 2 (default: qsd2_w2)")
    ap.add_argument("--qst1", default=None, help="test set 1 (default: qst1_w2 or qst_w2/qst1_w2)")
    ap.add_argument("--qst2", default=None, help="test set 2 (default: qst2_w2 or qst_w2/qst2_w2)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--check", action="store_true", help="evaluate on QSD1 / QSD2 before predicting")
    args = ap.parse_args()

    bbdd = Path(args.bbdd)
    if not bbdd.is_dir():
        raise SystemExit(f"'{bbdd}' not found: put the BBDD folder in this folder (or pass --bbdd <folder>)")

    print(f"Loading BBDD from {bbdd} ...")
    bb_paths = image_paths(bbdd, "bbdd_*.jpg")
    bb_ids = [image_id(p) for p in bb_paths]
    bb_imgs = [resize_max(cv2.imread(str(p))) for p in bb_paths]
    edges = range_edges(bb_imgs)
    db = np.stack([describe(x, edges) for x in bb_imgs])
    print(f"  {len(bb_imgs)} paintings, descriptor of {db.shape[1]} values")

    if args.check:
        qsd1 = find_dir(*([args.qsd1] if args.qsd1 else ["qsd1_w2"]))
        qsd2 = find_dir(*([args.qsd2] if args.qsd2 else ["qsd2_w2"]))
        if qsd1:
            gt = pickle.load(open(qsd1 / "gt_corresps.pkl", "rb"))
            preds = retrieve([resize_max(cv2.imread(str(p))) for p in image_paths(qsd1)], db, bb_ids, edges)
            print(f"\nQSD1 ({len(preds)} queries): mAP@1 = {mapk(gt, preds, 1):.3f}   mAP@5 = {mapk(gt, preds, 5):.3f}")
        if qsd2:
            gt = pickle.load(open(qsd2 / "gt_corresps.pkl", "rb"))
            crops, scores = [], []
            for p in image_paths(qsd2):
                img = cv2.imread(str(p))
                mask = compute_mask(img)
                scores.append(prf(mask, cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_GRAYSCALE) > 0))
                crops.append(crop_to_mask(img, mask))
            P, R, F = np.mean(scores, 0)
            preds = retrieve(crops, db, bb_ids, edges)
            print(f"QSD2 masks ({len(crops)} images): precision = {P:.3f}   recall = {R:.3f}   F1 = {F:.3f}")
            print(f"QSD2 retrieval with our masks: mAP@1 = {mapk(gt, preds, 1):.3f}   mAP@5 = {mapk(gt, preds, 5):.3f}")

    out = Path(args.out)

    qst1 = find_dir(*([args.qst1] if args.qst1 else ["qst1_w2", "qst_w2/qst1_w2"]))
    if qst1:
        imgs = [resize_max(cv2.imread(str(p))) for p in image_paths(qst1)]
        preds = retrieve(imgs, db, bb_ids, edges)
        save_result(preds, len(imgs), out / "QST1" / "method1" / "result.pkl")
        print(f"\nQST1: {len(imgs)} queries from {qst1} -> {out / 'QST1' / 'method1' / 'result.pkl'}   (first: {preds[0]})")
    else:
        print("\nQST1 not found (put qst1_w2 in this folder or pass --qst1 <folder>): skipped")

    qst2 = find_dir(*([args.qst2] if args.qst2 else ["qst2_w2", "qst_w2/qst2_w2"]))
    if qst2:
        mask_dir = out / "QST2" / "method1"
        mask_dir.mkdir(parents=True, exist_ok=True)
        paths = image_paths(qst2)
        crops = []
        for p in paths:
            img = cv2.imread(str(p))
            mask = compute_mask(img)
            cv2.imwrite(str(mask_dir / f"{p.stem}.png"), mask.astype(np.uint8) * 255)   # 00001.jpg -> 00001.png
            crops.append(crop_to_mask(img, mask))
        preds = retrieve(crops, db, bb_ids, edges)
        save_result(preds, len(crops), mask_dir / "result.pkl")
        for p in paths:                                     # one 0/255 mask per query, same name and size
            m = cv2.imread(str(mask_dir / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            assert m is not None and m.shape == cv2.imread(str(p)).shape[:2] and set(np.unique(m)) <= {0, 255}
        print(f"QST2: {len(crops)} queries from {qst2} -> {mask_dir / 'result.pkl'} + {len(paths)} masks   (first: {preds[0]})")
    else:
        print("QST2 not found (put qst2_w2 in this folder or pass --qst2 <folder>): skipped")

    print("\nFormat checked. Upload to the Drive as TeamX/week2/QST1/method1/result.pkl, TeamX/week2/QST2/method1/result.pkl "
          "and TeamX/week2/QST2/method1/*.png")


if __name__ == "__main__":
    main()
