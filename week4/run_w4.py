"""
C1 - Week 4: retrieval of museum paintings with keypoints + local descriptors, with unknown queries.

method1
  1. Noise and paintings: same functions as week 3 (run_w3.py). Noisy photos are filtered with a 3x3 median applied twice.
  2. Task 1: keypoints + descriptors (keypoints.py) on the BBDD and on the query, longest side <= WORK px.
  3. Task 2: tentative matches, ratio test + RANSAC homography (matching.py). Score of a BBDD image = number of inliers.
  4. Task 2 (II): a query whose best score is below T is unknown (unknowns.py) and returned as [[-1]].

Outputs (format of the week-4 instructions):
    results/QST1/method1/result.pkl     [[[p1 top-10], [p2 top-10]], [[-1]], [[top-10]], ...]
    results/dev_summary.md              development table (--check only)

Folders expected next to this script: qsd1_w4/ (for --check) and qst1_w4/. BBDD/ is optional: ../week3/BBDD is used otherwise.
Nested downloaded folders are found automatically.

Usage:
    python run_w4.py --check     # development results (QSD1-W4, all the options, week 3) + test predictions
    python run_w4.py             # test predictions only, with the chosen method and threshold
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'week3'))
from run_w3 import denoise, find_dir, image_id, jpgs, map_multi, resize_max, segment  # noqa: E402

import keypoints as kp  # noqa: E402
from matching import MIN_MATCHES, Matcher  # noqa: E402
from unknowns import best_threshold, loo_prf, prf  # noqa: E402

K = 10
WORK = 800
SECOND_MIN = 10     # a second painting needs at least these inliers, even with no discard
FULL, SEG, HYBRID = 'full', 'seg', 'hybrid'
# Chosen with --check: T ties in (7, 53] on QSD1-W4, and 25 is the best of that range on the independent check (see README.md)
METHOD, MODE, CROSS, T = 'sift', HYBRID, False, 25.0


# ----------------------------------------------------------------------------- data
def prepare(img): return resize_max(denoise(img), WORK)

class Database:
    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob('bbdd_*.jpg'), key=image_id)
        self.ids = [image_id(p) for p in self.paths]
        self.shapes = [prepare(cv2.imread(str(p))).shape[:2] for p in self.paths]

    def matcher(self, method, cross):
        m = kp.METHODS[method]
        feats = kp.load_bbdd(self.paths, self.ids, method, WORK, prepare, Path(__file__).parent / 'cache')
        return Matcher(feats, self.shapes, m['norm'], m['dtype'], m['ratio'], cross)


# ----------------------------------------------------------------------------- query description
def slim(region, center):
    return {'inliers': region['inliers'], 'matches': region['matches'], 'center': center}

def describe(photo, matcher, method, mode):
    """Photo -> list of region records (centres in pixels of the denoised photo).
    full    the two paintings found by sequential RANSAC on the whole photo
    seg     the week-3 paintings, each one searched on its own
    hybrid  full + the week-3 painting not covered by the first match, kept as a third record to fill the layout"""
    img = denoise(photo)
    def search_box(box):
        x, y, w, h = box; c = resize_max(img[y:y + h, x:x + w], WORK); xy, desc = kp.extract(c, method)
        return slim(matcher.search(xy, desc, c.shape[:2]), np.array([x + w / 2, y + h / 2]))
    boxes = [cv2.boundingRect(m.astype(np.uint8)) for m in segment(img)] if mode != FULL else []
    if mode == SEG: return [search_box(b) for b in boxes]
    small = resize_max(img, WORK); xy, desc = kp.extract(small, method); s = small.shape[:2]; f = max(img.shape[:2]) / max(s)
    centre = lambda r, b: None if r['quads'][b] is None else r['quads'][b].mean(0) * f
    first = matcher.search(xy, desc, s); best = int(np.argmax(first['inliers']))
    out = [slim(first, centre(first, best))]
    if first['inliers'][best] < MIN_MATCHES or out[0]['center'] is None: return out
    second = matcher.search_second(first, s); out.append(slim(second, centre(second, int(np.argmax(second['inliers'])))))
    if mode == HYBRID and len(boxes) == 2:
        inside = lambda b: b[0] <= out[0]['center'][0] <= b[0] + b[2] and b[1] <= out[0]['center'][1] <= b[1] + b[3]
        other = [b for b in boxes if not inside(b)]
        if len(other) == 1: out.append(search_box(other[0]))
    return out

def score(regions): return int(max(r['inliers'].max() for r in regions[:2]))

def ranking(region, ids):
    return [int(ids[i]) for i in np.lexsort((-region['matches'], -region['inliers']))[:K]]

def in_order(regions):
    """Left to right, top to bottom if stacked (order of the ground truth and of week 3)."""
    if len(regions) < 2: return regions
    a, b = regions[0]['center'], regions[1]['center']
    ax = 0 if abs(a[0] - b[0]) >= abs(a[1] - b[1]) else 1
    return regions if a[ax] <= b[ax] else regions[::-1]

def predict(regions, ids, thr, mode):
    """Region records -> [[top-K], ...] or [[-1]]. thr = 0 means no discard."""
    if mode == SEG:
        out = [ranking(r, ids) if r['inliers'].max() >= thr else [-1] for r in regions]
        return out if any(p != [-1] for p in out) else [[-1]]
    if regions[0]['inliers'].max() < thr: return [[-1]]
    if len(regions) < 2: return [ranking(regions[0], ids)]
    second = regions[1]
    if second['inliers'].max() >= max(thr, SECOND_MIN) and second['center'] is not None:
        return [ranking(r, ids) for r in in_order(regions[:2])]
    if len(regions) == 3:       # the week-3 layout says there is a second painting, but its match was not verified
        return [ranking(r, ids) for r in in_order([regions[0], regions[2]])]
    return [ranking(regions[0], ids)]


# ----------------------------------------------------------------------------- evaluation helpers
def is_unknown(g): return g == [-1]

def scores_table(gt, preds):
    known = [i for i, g in enumerate(gt) if not is_unknown(g)]
    return {'map1': map_multi(gt, preds, 1), 'map5': map_multi(gt, preds, 5),
            'map1_known': map_multi([gt[i] for i in known], [preds[i] for i in known], 1),
            'map5_known': map_multi([gt[i] for i in known], [preds[i] for i in known], 5),
            'count': float(np.mean([len(p) == len(g) for p, g in zip(preds, gt)]))}

def evaluate(name, gt, photo_scores, make_preds):
    """make_preds(thr) -> predictions. Returns a row of the development table."""
    unk = [is_unknown(g) for g in gt]
    thr, f1, lo, hi = best_threshold(photo_scores, unk)
    p, r, _ = prf(np.asarray(photo_scores) < thr, unk); lp, lr, lf = loo_prf(photo_scores, unk)
    plain, disc = scores_table(gt, make_preds(None)), scores_table(gt, make_preds(thr))
    return dict(name=name, T=thr, lo=lo, hi=hi, P=p, R=r, F1=f1, loo_F1=lf, plain=plain, disc=disc)

def holdout(best_per_region, gt):
    """Independent check with the week-3 sets, all known: the best score simulates a known query, and the best score
    with the right paintings removed from the BBDD simulates an unknown one. best_per_region(photo index, excluded ids)."""
    known = [best_per_region(i, []) for i in range(len(gt))]
    unknown = [best_per_region(i, gt[i]) for i in range(len(gt))]
    return np.array(known), np.array(unknown)

def row_holdout(name, known, unknown, thr):
    return (f'| {name} | {thr:.2f} | {np.mean(known < thr):.3f} | {np.mean(unknown >= thr):.3f} | {known.min():.2f} | '
            f'{np.percentile(known, 5):.2f} | {np.percentile(unknown, 95):.2f} | {unknown.max():.2f} |')

def row_md(r, t=None):
    d, n = r['disc'], r['plain']
    return (f"| {r['name']} | {r['T']:.2f} ({r['lo']:.2f}, {r['hi']:.2f}] | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | {r['loo_F1']:.3f} | "
            f"{n['map1']:.3f} | {n['map5']:.3f} | {d['map1']:.3f} | {d['map5']:.3f} | {d['map1_known']:.3f} | {d['map5_known']:.3f} | {d['count']:.3f} | {'' if t is None else f'{t:.1f}'} |")

def save_result(preds, n, path):
    assert len(preds) == n
    for photo in preds:
        assert isinstance(photo, list) and 1 <= len(photo) <= 2
        for p in photo:
            assert p == [-1] or (len(p) == K and len(set(p)) == K and all(type(i) is int and i >= 0 for i in p))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(preds, f)


# ----------------------------------------------------------------------------- development
def check(db, qsd, out):
    from baseline_w3 import BaselineW3
    gt = pickle.load(open(qsd / 'gt_corresps.pkl', 'rb')); photos = [cv2.imread(str(p)) for p in jpgs(qsd)]
    rows, chosen, thrs = [], None, {}
    for method in kp.METHODS:
        for cross in (False, True):
            matcher = db.matcher(method, cross)
            for mode in (FULL, SEG) + (() if cross else (HYBRID,)):
                t = time.time(); recs = [describe(p, matcher, method, mode) for p in photos]; t = (time.time() - t) / len(photos)
                name = f"{method} {'ratio+cross' if cross else 'ratio'} {mode}"
                r = evaluate(name, gt, [score(x) for x in recs], lambda thr: [predict(x, db.ids, thr or 0, mode) for x in recs])
                rows.append((r, t)); print(row_md(r, t), flush=True)
                if (method, mode, cross) == (METHOD, MODE, CROSS): chosen = r
                if mode == HYBRID: thrs[method] = r['T']
            matcher.pool.shutdown()
    print('\nWeek 3 on QSD1-W4 ...', flush=True)
    base = BaselineW3(db.paths, db.ids); t = time.time()
    res = [base.describe(p) for p in photos]; t = (time.time() - t) / len(photos)
    r3 = evaluate('week 3 texture fusion seg', gt, [s for _, s in res], lambda thr: [p if thr is None or s >= thr else [[-1]] for p, s in res])
    rows.append((r3, t)); print(row_md(r3, t))
    head = ('| Option | T (range) | P | R | F1 | F1 LOO | mAP@1 no discard | mAP@5 no discard | mAP@1 | mAP@5 | mAP@1 known | mAP@5 known | paintings ok | s/query |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    out.mkdir(parents=True, exist_ok=True)
    text = head + '\n'.join(row_md(*x) for x in rows) + '\n'

    w3gt, w3photos = [], []
    for name in ('../week3/qsd1_w3', '../week3/qsd2_w3'):
        d = find_dir(Path(__file__).parent / name)
        if d is not None:
            w3gt += pickle.load(open(d / 'gt_corresps.pkl', 'rb')); w3photos += [cv2.imread(str(p)) for p in jpgs(d)]
    if w3photos:
        print(f'\nIndependent check on the week-3 development sets ({len(w3photos)} photos) ...', flush=True)
        lines = []; idx = {i: n for n, i in enumerate(db.ids)}
        for method in kp.METHODS:
            matcher = db.matcher(method, False); recs = [describe(p, matcher, method, FULL) for p in w3photos]; matcher.pool.shutdown()
            def best(i, excl):
                drop = [idx[g] for g in excl]
                return max(float(np.delete(r['inliers'], drop).max()) for r in recs[i][:2])
            known, unknown = holdout(best, w3gt)
            lines.append(row_holdout(method, known, unknown, thrs[method])); print(lines[-1], flush=True)
            if method == METHOD: lines.append(row_holdout(f'{method} (chosen T)', known, unknown, T)); print(lines[-1], flush=True)
        fused = [[base.fused(crop_box) for crop_box in base.crops(p)] for p in w3photos]
        def best3(i, excl):
            drop = [idx[g] for g in excl]
            return max(float(-np.delete(t, drop).min()) for t in fused[i])
        lines.append(row_holdout('week 3 texture fusion', *holdout(best3, w3gt), r3['T'])); print(lines[-1], flush=True)
        text += ('\n## Independent check: week-3 development sets, unknowns simulated by removing the right paintings from the BBDD\n\n'
                 '| Option | T of QSD1-W4 | known missed | false known | known min | known p5 | unknown p95 | unknown max |\n|---|---|---|---|---|---|---|---|\n' + '\n'.join(lines) + '\n')
    (out / 'dev_summary.md').write_text(text, encoding='utf-8')
    if chosen is not None and not chosen['lo'] < T <= chosen['hi']:
        print(f"\nWARNING: T = {T} is outside the best range ({chosen['lo']}, {chosen['hi']}] of the chosen method")
    print(f"\nTable saved in {out / 'dev_summary.md'}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='C1 week 4: QST1 predictions (method1)')
    ap.add_argument('--bbdd', default=None); ap.add_argument('--qst1', default=None); ap.add_argument('--qsd1', default=None)
    ap.add_argument('--out', default='results'); ap.add_argument('--check', action='store_true', help='evaluate on QSD1-W4 first')
    args = ap.parse_args()

    bbdd = find_dir(*([args.bbdd] if args.bbdd else ['BBDD', '../week3/BBDD']))
    if bbdd is None: raise SystemExit('BBDD not found: put the BBDD folder next to this script (or pass --bbdd <folder>)')
    print(f'BBDD from {bbdd}'); db = Database(bbdd); out = Path(args.out)

    if args.check:
        qsd = find_dir(*([args.qsd1] if args.qsd1 else ['qsd1_w4']))
        if qsd is None: print('QSD1-W4 not found (put qsd1_w4 in this folder or pass --qsd1 <folder>): skipped')
        else: check(db, qsd, out)

    d = find_dir(*[c for c in [args.qst1, 'qst1_w4', 'qst_w4/qst1_w4'] if c])
    if d is None:
        print('\nQST1 not found (put qst1_w4 in this folder or pass --qst1 <folder>): skipped'); return
    matcher = db.matcher(METHOD, CROSS); qpaths = jpgs(d)
    preds = [predict(describe(cv2.imread(str(p)), matcher, METHOD, MODE), db.ids, T, MODE) for p in qpaths]
    outp = out / 'QST1' / 'method1' / 'result.pkl'; save_result(preds, len(qpaths), outp)
    print(f'\nQST1: {len(qpaths)} images, {sum(map(len, preds))} paintings, {sum(p == [[-1]] for p in preds)} unknown -> {outp}   (first: {preds[0]})')
    print('Upload to the Drive as TeamX/week4/QST1/method1/result.pkl')


if __name__ == '__main__':
    main()
