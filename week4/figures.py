"""
C1 - Week 4: figures for the slides, written to results/figures/ (needs qsd1_w4/ and the BBDD).

    matches_known.png    inlier matches of a known query
    matches_two.png      inlier matches of a photo with two paintings
    matches_unknown.png  best verified match of an unknown query
    scores.png           best number of inliers per query, with the threshold T
    fail_*.png           the two hardest paintings: query crop and BBDD image

Usage:
    python figures.py
"""

import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import keypoints as kp
from matching import RANSAC_CONF, RANSAC_ITERS, RANSAC_PX
from run_w4 import CROSS, METHOD, MODE, T, WORK, Database, denoise, describe, jpgs, prepare, resize_max, score

OUT = Path('results/figures')
SINGLE, DOUBLE, UNKNOWN = 4, 2, 26


def draw(photo, db, feats, matcher, name):
    """Query on the left, its best BBDD images on the right, inlier matches in green."""
    small = resize_max(denoise(photo), WORK); xy, desc = kp.extract(small, METHOD)
    first = matcher.search(xy, desc, small.shape[:2]); regions = [first]
    if first['inliers'].max() >= 4:
        second = matcher.search_second(first, small.shape[:2])
        if second['inliers'].max() >= 10: regions.append(second)
    panels, H = [], 600
    q = cv2.resize(small, None, fx=H / small.shape[0], fy=H / small.shape[0]); s = H / small.shape[0]; x0 = q.shape[1] + 20
    lines = []
    for r in regions:
        b = int(np.argmax(r['inliers'])); dxy = feats[b][0]; p = r['pairs'][b]
        im = prepare(cv2.imread(str(db.paths[b]))); f = H / im.shape[0]; panels.append(cv2.resize(im, None, fx=f, fy=f))
        _, mask = cv2.findHomography(dxy[p[:, 1]], xy[p[:, 0]], cv2.RANSAC, RANSAC_PX, maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
        lines.append((p[mask.ravel() == 1], dxy, f, x0)); x0 += panels[-1].shape[1] + 20
    gap = np.full((H, 20, 3), 255, np.uint8); canvas = q
    for pn in panels: canvas = np.hstack([canvas, gap, pn])
    canvas = canvas.copy()
    for (p, dxy, f, x), colour in zip(lines, [(60, 200, 60), (40, 150, 255)]):
        for a, c in p[:: max(1, len(p) // 30)]:
            cv2.line(canvas, tuple(np.int32(xy[a] * s)), (int(dxy[c, 0] * f) + x, int(dxy[c, 1] * f)), colour, 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / name), canvas)
    print(name, [(db.ids[int(np.argmax(r['inliers']))], int(r['inliers'].max())) for r in regions])


def scores_plot(values, gt):
    known = [v for v, g in zip(values, gt) if g != [-1]]; unknown = [v for v, g in zip(values, gt) if g == [-1]]
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=200)
    jitter = lambda n, c: c + np.linspace(-0.13, 0.13, n)
    ax.scatter(np.maximum(known, 1), jitter(len(known), 1), s=30, color='#2F6F5E', linewidths=0)
    ax.scatter(np.maximum(unknown, 1), jitter(len(unknown), 0), s=30, color='#C8553D', linewidths=0)
    ax.axvline(T, color='#444444', lw=1); ax.text(T * 1.05, 1.42, f'T = {T:.0f}', color='#444444', fontsize=13)
    ax.set_xscale('log'); ax.set_xticks([1, 3, 10, 30, 100, 300, 1000]); ax.set_xticklabels(['1', '3', '10', '30', '100', '300', '1000'], fontsize=13)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Unknown', 'In the BBDD'], fontsize=13); ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel('Inliers of the best BBDD image (log scale)', fontsize=13)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.tight_layout(); fig.savefig(OUT / 'scores.png'); plt.close(fig)


def failures(paths, folder):
    """Painting 128 (image 8) and painting 11 (left of image 16): query and BBDD image."""
    q8 = denoise(cv2.imread(str(paths[8])))
    x, y, w, h = 13, 57, 339, 439                   # frame of painting 11 in frames.pkl
    q16 = cv2.imread(str(paths[16]))[y:y + h, x:x + w]
    for name, im in [('fail_128_query', q8), ('fail_128_bbdd', cv2.imread(str(folder / 'bbdd_00128.jpg'))),
                     ('fail_11_query', q16), ('fail_11_bbdd', cv2.imread(str(folder / 'bbdd_00011.jpg')))]:
        cv2.imwrite(str(OUT / f'{name}.png'), resize_max(im, 700))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    db = Database('../week3/BBDD'); matcher = db.matcher(METHOD, CROSS); feats = matcher.feats
    q = Path('qsd1_w4'); gt = pickle.load(open(q / 'gt_corresps.pkl', 'rb')); paths = jpgs(q)
    for i, name in [(SINGLE, 'matches_known.png'), (DOUBLE, 'matches_two.png'), (UNKNOWN, 'matches_unknown.png')]:
        draw(cv2.imread(str(paths[i])), db, feats, matcher, name)
    scores_plot([score(describe(cv2.imread(str(p)), matcher, METHOD, MODE)) for p in paths], gt)
    failures(paths, db.paths[0].parent)
    print(f'Figures saved in {OUT}')


if __name__ == '__main__':
    main()
