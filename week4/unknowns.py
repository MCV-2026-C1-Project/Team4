"""
C1 - Week 4 - Task 2 (II): discard queries that are not in the BBDD.

A query is unknown when the score of its best BBDD image is below a threshold T (unknown = positive class).
T maximises F1 on the development set; among the thresholds that tie, the middle of the widest range is taken,
so that T stays as far as possible from both classes.
"""

import numpy as np


def prf(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp, fp, fn = np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(scores, unknown):
    """Returns (T, best F1, low, high): every t in (low, high] gives the same decisions and the best F1."""
    scores = np.asarray(scores, float)
    u = np.unique(scores)
    high = np.append(u, u[-1] + 1); low = np.append(u[0] - 1, u)
    f1 = np.array([prf(scores < t, unknown)[2] for t in high])
    i = 0; best = (-1.0, 0, 0)
    while i < len(f1):
        j = i
        if np.isclose(f1[i], f1.max()):
            while j + 1 < len(f1) and np.isclose(f1[j + 1], f1.max()): j += 1
            if high[j] - low[i] > best[0]: best = (high[j] - low[i], i, j)
        i = j + 1
    _, i, j = best
    return float((low[i] + high[j]) / 2), float(f1.max()), float(low[i]), float(high[j])


def loo_prf(scores, unknown):
    """Leave-one-out estimate: each query is decided with the threshold fitted on the other ones."""
    scores, unknown = np.asarray(scores, float), np.asarray(unknown, bool); pred = []
    for i in range(len(scores)):
        rest = np.arange(len(scores)) != i
        pred.append(scores[i] < best_threshold(scores[rest], unknown[rest])[0])
    return prf(pred, unknown)
