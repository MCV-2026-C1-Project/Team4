"""
C1 - Week 3 - Task 5: predictions for the blind test sets QST1 and QST2 (one method: method1).
Same pipeline as the notebook c1-week3.ipynb (all the experiments and the reasons behind each choice are there).

method1
  1. Noise (Task 1): an image is noisy if more than 0.5% of its pixels are isolated impulses (they differ by > 40 grey levels
     from all their 8 neighbours, measured at full resolution). Noisy images are filtered with a 3x3 median applied twice.
  2. Paintings (Task 3, QST2 only): Team 8's wall model (the wall chroma a,b as a quadratic function of the lightness L,
     robust fit on a 1% border strip) -> Mahalanobis distance <= 5.5 = wall-like -> wall = wall-like pixels connected to
     the border -> up to 2 components (>= 2% of the photo, the second >= 10% of the first) -> per painting: fill holes,
     trim sparse rows/columns, row/column span fill, quadrilateral fit -> edge recovery of paintings with the colour of the
     wall (only if fewer than 2 were found) -> paintings ordered left to right (top to bottom if stacked).
  3. Descriptors of each painting (bounding-box crop, longest side <= 512 px):
       HOG      128x128 grey, 16x16 cells, 2x2 blocks, 8 orientations, L2
       wavelet  256x256 grey, Haar, 3 levels, 8x8 blocks, mean|c| + std per sub-band + standardised approximation, L1
       colour   week-2 method1: 5% margin, light normalised, YCrCb 3D 6x12x12 range bins, pyramid 1+2+4+8, Hellinger
  4. Late fusion: per query, each distance vector is z-scored over the 287 BBDD images; the three are added.

Outputs (format of the week-3 instructions, with the extra list level):
    results/QST1/method1/result.pkl     [[[top-10]], [[top-10]], ...]          one list of paintings per query image
    results/QST2/method1/result.pkl     [[[p1 top-10], [p2 top-10]], [[...]], ...]
    results/QST2/method1/*.png          binary masks (0 / 255), same name and size as the query (00001.jpg -> 00001.png)

Folders expected next to this script: BBDD/ (or ../week2/week2_submission/BBDD, ../week1/BBDD), qst1_w3/, qst2_w3/
(and qsd1_w3/, qsd2_w3/ for --check). Sub-folders as downloaded (qst1_w3/qst1_w3/...) are found automatically.

Usage:
    python run_w3.py --check     # development results (QSD1-W3 / QSD2-W3) + test predictions
    python run_w3.py             # test predictions only
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pywt
from skimage.feature import hog

K = 10
EPS = 1e-10
NOISE_T = 0.005
RING = np.ones((3, 3), np.uint8); RING[1, 1] = 0


# ----------------------------------------------------------------------------- data
def image_id(p): return int(Path(p).stem.split('_')[-1])
def jpgs(folder): return sorted(Path(folder).glob('*.jpg'), key=image_id)
def resize_max(img, m=512):
    s = m / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img

def find_dir(*candidates):
    for c in map(Path, candidates):
        if c and c.is_dir():
            if any(c.glob('*.jpg')): return c
            for sub in sorted(p for p in c.iterdir() if p.is_dir()):
                if any(sub.glob('*.jpg')): return sub
    return None


# ----------------------------------------------------------------------------- Task 1: noise
def impulse_ratio(img, T=40):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mx = cv2.dilate(g, RING).astype(np.int16); mn = cv2.erode(g, RING).astype(np.int16); gi = g.astype(np.int16)
    return float(np.mean(np.maximum(gi - mx, mn - gi) > T))

def denoise(img):
    return cv2.medianBlur(cv2.medianBlur(img, 3), 3) if impulse_ratio(img) > NOISE_T else img


# ----------------------------------------------------------------------------- Task 3: paintings and masks
def border_strip(h, w, ratio):
    bh, bw = max(2, int(h * ratio)), max(2, int(w * ratio))
    b = np.zeros((h, w), bool); b[:bh] = b[-bh:] = True; b[:, :bw] = b[:, -bw:] = True; return b

def wall_distance(img, border_ratio=0.01, step=2, dark_gate=30.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32); h, w = lab.shape[:2]
    samp = np.zeros((h, w), bool); samp[::step, ::step] = True; samp &= border_strip(h, w, border_ratio)
    cols = lab[samp]; L = cols[:, 0] / 255.0; A = np.column_stack([np.ones_like(L), L, L ** 2]); T = cols[:, 1:]
    wts = np.ones(len(cols))
    for _ in range(8):
        sw = np.sqrt(wts)[:, None]; coef = np.linalg.lstsq(A * sw, T * sw, rcond=None)[0]
        res = np.linalg.norm(T - A @ coef, axis=1); med = np.median(res)
        sc = max(0.5, 1.4826 * np.median(np.abs(res - med))); wts = np.minimum(1.0, 1.5 * sc / np.maximum(res, 1e-6))
    err = T - A @ coef; c = np.median(err, 0); ce = err - c
    inv = np.linalg.inv((ce * wts[:, None]).T @ ce / max(wts.sum(), 1) + np.eye(2))
    Lp = np.clip(lab[..., 0] / 255.0, L.min(), L.max())
    d = lab[..., 1:] - (coef[0] + Lp[..., None] * coef[1] + Lp[..., None] ** 2 * coef[2]) - c
    dist = np.sqrt(np.maximum(0, np.einsum('...i,ij,...j->...', d, inv, d)))
    dist[lab[..., 0] < max(0.0, np.percentile(cols[:, 0], 2) - dark_gate)] = np.inf
    return dist

def wall_from(wall_like, connectivity=8):
    n, lbl = cv2.connectedComponents(wall_like.astype(np.uint8), connectivity=connectivity)
    keep = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])); keep = keep[keep > 0]
    return np.isin(lbl, keep)

def fill_holes(m):
    p = np.pad(m.astype(np.uint8), 1); ext = p.copy(); cv2.floodFill(ext, None, (0, 0), 1)
    return (p | (1 - ext)).astype(bool)[1:-1, 1:-1]

def span_fill(m):
    rows, cols = np.zeros_like(m), np.zeros_like(m)
    for y in np.where(m.any(1))[0]: xs = np.flatnonzero(m[y]); rows[y, xs[0]:xs[-1] + 1] = True
    for x in np.where(m.any(0))[0]: ys = np.flatnonzero(m[:, x]); cols[ys[0]:ys[-1] + 1, x] = True
    return rows & cols

def trim_sparse(m, frac=0.3):
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = m[y0:y1, x0:x1]; out = np.zeros_like(m)
    out[y0:y1, x0:x1] = sub & (sub.mean(1) >= frac)[:, None] & (sub.mean(0) >= frac)[None, :]
    return out if out.sum() > 0.5 * m.sum() else m

def quad_fit(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts)); peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) <= 4: break
    if len(ap) != 4: ap = cv2.boxPoints(cv2.minAreaRect(np.vstack(cnts))).astype(np.int32).reshape(-1, 1, 2)
    out = np.zeros(m.shape, np.uint8); cv2.fillPoly(out, [ap.reshape(-1, 2)], 1)
    return out.astype(bool)

def segment(img, work=512, sigma=5.5, min_area=0.02, second_ratio=0.1, max_paintings=2):
    """Denoised photo -> list of boolean masks (full size), one per painting, left to right."""
    H, W = img.shape[:2]; s = min(1.0, work / max(H, W))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy(); h, w = small.shape[:2]
    se = lambda f: cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(f * min(h, w))) | 1,) * 2)
    wall_like = wall_distance(small) <= sigma
    fg = cv2.morphologyEx((~wall_from(wall_like)).astype(np.uint8), cv2.MORPH_CLOSE, se(0.01)).astype(bool)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    cands = [lbl == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area * h * w]
    if len(cands) < max_paintings:      # edge recovery of paintings with the colour of the wall
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 1.2), 50, 150)
        e = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
        fe = cv2.morphologyEx((~wall_from(wall_like & ~e, 4)).astype(np.uint8), cv2.MORPH_OPEN, se(0.01)).astype(bool)
        taken = cv2.dilate(np.any(cands, 0).astype(np.uint8), se(0.03)).astype(bool) if cands else np.zeros((h, w), bool)
        n2, l2, st2, _ = cv2.connectedComponentsWithStats(fe.astype(np.uint8), connectivity=8)
        for i in range(1, n2):
            m = l2 == i
            if st2[i, cv2.CC_STAT_AREA] < min_area * h * w or (m & taken).any(): continue
            mf = fill_holes(m)
            if mf.sum() / max(quad_fit(mf).sum(), 1) >= 0.85: cands.append(mf)
    cands.sort(key=lambda m: -m.sum())
    cands = [m for m in cands if m.sum() >= second_ratio * cands[0].sum()][:max_paintings] if cands else []
    masks = [quad_fit(span_fill(trim_sparse(fill_holes(m)))) for m in cands]
    if len(masks) == 2:
        c = [np.argwhere(m).mean(0) for m in masks]
        ax = 1 if abs(c[0][1] - c[1][1]) >= abs(c[0][0] - c[1][0]) else 0
        masks = [masks[i] for i in np.argsort([ci[ax] for ci in c])]
    full = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
    return full if full else [np.ones((H, W), bool)]

def crop(img, mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8)); return resize_max(img[y:y + h, x:x + w])


# ----------------------------------------------------------------------------- descriptors
def grey_square(img, size):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)

def desc_hog(img):
    v = hog(grey_square(img, 128), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)
    return v / (np.linalg.norm(v) + EPS)

def desc_wavelet(img, size=256, level=3, grid=8):
    co = pywt.wavedec2(grey_square(img, size).astype(np.float32) / 255, 'haar', level=level)
    a = co[0]; bands = [(a - a.mean()) / (a.std() + 1e-6)] + [b for lv in co[1:] for b in lv]; out = []
    for b in bands:
        h, w = b.shape; n = min(grid, h); ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
        for i in range(n):
            for j in range(n):
                bb = b[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]; out += [np.mean(np.abs(bb)), np.std(bb)]
    v = np.array(out, np.float32); return v / (np.linalg.norm(v) + EPS)

MARGIN, BINS, LEVELS, LWEIGHTS = 0.05, (6, 12, 12), (1, 2, 4, 8), (1, 2, 4, 8)
def prepare_color(img):
    h, w = img.shape[:2]; f = img[int(MARGIN * h):h - int(MARGIN * h), int(MARGIN * w):w - int(MARGIN * w)].astype(np.float32)
    f = np.clip(f * (128 / (f.reshape(-1, 3).mean(0) + EPS)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)

class Colour:
    def __init__(self, bbdd_imgs):
        P = np.concatenate([prepare_color(x).reshape(-1, 3) for x in bbdd_imgs])
        P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)].astype(np.float64)
        self.luts = []
        for c, n in enumerate(BINS):
            lo, hi = np.percentile(P[:, c], [0.5, 99.5]); e = np.linspace(lo, hi, n + 1); e[0], e[-1] = -1, 256
            self.luts.append(np.clip(np.searchsorted(e[1:-1], np.arange(256), side='right'), 0, n - 1))
    def __call__(self, img):
        x = prepare_color(img); L = self.luts
        idx = L[0][x[..., 0]] * BINS[1] * BINS[2] + L[1][x[..., 1]] * BINS[2] + L[2][x[..., 2]]
        h, w = idx.shape; wts = np.asarray(LWEIGHTS, float) / np.sum(LWEIGHTS); out = []
        for n, wl in zip(LEVELS, wts):
            ys, xs = np.linspace(0, h, n + 1).astype(int), np.linspace(0, w, n + 1).astype(int)
            for i in range(n):
                for j in range(n):
                    hb = np.bincount(idx[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel(), minlength=int(np.prod(BINS))).astype(np.float32)
                    out.append(wl * hb / (hb.sum() + EPS) / (n * n))
        return np.concatenate(out)


class Retriever:
    def __init__(self, bbdd_imgs, bbdd_ids):
        self.ids = np.asarray(bbdd_ids)
        self.colour = Colour(bbdd_imgs)
        self.desc = {'hog': (desc_hog, 'L2'), 'wavelet': (desc_wavelet, 'L1'), 'colour': (self.colour, 'hellinger')}
        self.db = {k: np.stack([f(x) for x in bbdd_imgs]) for k, (f, _) in self.desc.items()}

    def top_k(self, img, k=K):
        total = 0
        for name, (f, kind) in self.desc.items():
            q, D = f(img), self.db[name]
            d = {'L2': lambda: np.linalg.norm(D - q, axis=1), 'L1': lambda: np.abs(D - q).sum(1),
                 'hellinger': lambda: -np.sqrt(D * q).sum(1)}[kind]()
            total = total + (d - d.mean()) / (d.std() + EPS)            # z-score late fusion
        return [int(i) for i in self.ids[np.argsort(total, kind='stable')[:k]]]

    def predict(self, photo, multi):
        img = denoise(photo)
        masks = segment(img) if multi else [np.ones(img.shape[:2], bool)]
        return [self.top_k(crop(img, m)) for m in masks], np.any(masks, 0)


# ----------------------------------------------------------------------------- evaluation helpers
def apk(actual, predicted, k=10):
    predicted = predicted[:k]; score, hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1; score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0

def map_multi(gt, preds, k):
    return float(np.mean([apk([g], preds[i][j], k) if j < len(preds[i]) else 0.0 for i in range(len(gt)) for j, g in enumerate(gt[i])]))

def prf(pred, gt):
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def save_result(preds, n, path):
    assert len(preds) == n
    for photo in preds:
        assert isinstance(photo, list) and 1 <= len(photo) <= 2
        assert all(len(p) == K and len(set(p)) == K and all(type(i) is int for i in p) for p in photo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(preds, f)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='C1 week 3: QST1 / QST2 predictions (method1)')
    ap.add_argument('--bbdd', default=None); ap.add_argument('--qst1', default=None); ap.add_argument('--qst2', default=None)
    ap.add_argument('--out', default='results'); ap.add_argument('--check', action='store_true', help='evaluate on QSD1-W3 / QSD2-W3 first')
    args = ap.parse_args()

    bbdd = find_dir(*([args.bbdd] if args.bbdd else ['BBDD', '../week2/week2_submission/BBDD', '../week1/BBDD']))
    if bbdd is None: raise SystemExit('BBDD not found: put the BBDD folder next to this script (or pass --bbdd <folder>)')
    print(f'Loading BBDD from {bbdd} ...')
    paths = sorted(bbdd.glob('bbdd_*.jpg'), key=image_id)
    R = Retriever([resize_max(denoise(cv2.imread(str(p)))) for p in paths], [image_id(p) for p in paths])
    print(f'  {len(paths)} paintings described (HOG + wavelet + colour)')

    if args.check:
        q1, q2 = find_dir('qsd1_w3'), find_dir('qsd2_w3')
        if q1:
            gt = pickle.load(open(q1 / 'gt_corresps.pkl', 'rb'))
            preds = [R.predict(cv2.imread(str(p)), False)[0] for p in jpgs(q1)]
            print(f'\nQSD1-W3: mAP@1 = {map_multi(gt, preds, 1):.3f}   mAP@5 = {map_multi(gt, preds, 5):.3f}')
        if q2:
            gt = pickle.load(open(q2 / 'gt_corresps.pkl', 'rb')); preds, sc = [], []
            for p in jpgs(q2):
                pr, m = R.predict(cv2.imread(str(p)), True); preds.append(pr)
                sc.append(prf(m, cv2.imread(str(p.with_suffix('.png')), cv2.IMREAD_GRAYSCALE) > 0))
            P, Rc, F = np.mean(sc, 0)
            print(f'QSD2-W3 masks: precision = {P:.3f}   recall = {Rc:.3f}   F1 = {F:.3f}')
            print(f'QSD2-W3 retrieval: mAP@1 = {map_multi(gt, preds, 1):.3f}   mAP@5 = {map_multi(gt, preds, 5):.3f}')

    out = Path(args.out)
    for name, cand, multi in [('QST1', [args.qst1, 'qst1_w3', 'qst_w3/qst1_w3'], False), ('QST2', [args.qst2, 'qst2_w3', 'qst_w3/qst2_w3'], True)]:
        d = find_dir(*[c for c in cand if c])
        if d is None:
            print(f'\n{name} not found (put {name.lower()}_w3 in this folder or pass --{name.lower()} <folder>): skipped'); continue
        outd = out / name / 'method1'; outd.mkdir(parents=True, exist_ok=True)
        qpaths = jpgs(d); preds = []
        for p in qpaths:
            img = cv2.imread(str(p)); pr, mask = R.predict(img, multi); preds.append(pr)
            if multi:
                cv2.imwrite(str(outd / f'{p.stem}.png'), mask.astype(np.uint8) * 255)
                m = cv2.imread(str(outd / f'{p.stem}.png'), cv2.IMREAD_GRAYSCALE)
                assert m.shape == img.shape[:2] and set(np.unique(m)) <= {0, 255}
        save_result(preds, len(qpaths), outd / 'result.pkl')
        print(f'\n{name}: {len(qpaths)} images, {sum(map(len, preds))} paintings -> {outd / "result.pkl"}'
              + (f' + {len(qpaths)} masks' if multi else '') + f'   (first: {preds[0]})')
    print('\nUpload to the Drive as TeamX/week3/QST1/method1/result.pkl, TeamX/week3/QST2/method1/result.pkl and TeamX/week3/QST2/method1/*.png')


if __name__ == '__main__':
    main()
