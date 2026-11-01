"""
C1 - Week 4 - Task 2: tentative matches and their verification.

  1. Tentative matches: exhaustive nearest neighbours (deterministic), Lowe's ratio test, optional cross-check.
  2. Verification: RANSAC homography from the BBDD image to the query, with a fixed seed; implausible models are rejected.
  3. The score of a BBDD image is its number of inliers (its number of tentative matches breaks the ties).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SEED = 42
MIN_MATCHES = 4
RANSAC_PX = 5.0
RANSAC_ITERS = 2000
RANSAC_CONF = 0.995
MAX_SCALE = 8.0
MIN_AREA = 0.01


def tentative_matches(q_desc, d_desc, norm, dtype, ratio, cross=False):
    """Query and BBDD descriptors -> M x 2 array of (query index, BBDD index)."""
    if len(q_desc) < 1 or len(d_desc) < 2:
        return np.zeros((0, 2), np.int32)
    dist, nn = cv2.batchDistance(q_desc, d_desc, dtype, normType=norm, K=2)
    keep = np.flatnonzero(dist[:, 0] < ratio * dist[:, 1])
    pairs = np.column_stack([keep, nn[keep, 0]]).astype(np.int32)
    if cross and len(pairs):
        _, back = cv2.batchDistance(d_desc, q_desc, dtype, normType=norm, K=1)
        pairs = pairs[back[pairs[:, 1], 0] == pairs[:, 0]]
    if len(pairs):      # each BBDD keypoint is used once (repeated textures send many query keypoints to the same one)
        d = dist[pairs[:, 0], 0]; order = np.argsort(d, kind='stable')
        _, first = np.unique(pairs[order, 1], return_index=True)
        pairs = pairs[np.sort(order[first])]
    return pairs


def plausible(H, shape, q_shape):
    """The BBDD outline projected on the query must be a convex, not mirrored, reasonably sized quadrilateral."""
    h, w = shape; Q, W = q_shape
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    if not cv2.isContourConvex(quad.astype(np.float32)): return None
    a = np.roll(quad, -1, 0) - quad; b = np.roll(a, -1, 0)
    if (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).min() <= 0: return None       # mirrored or twisted
    area = cv2.contourArea(quad); scale = np.sqrt(area / (w * h))
    if not 1 / MAX_SCALE <= scale <= MAX_SCALE or area < MIN_AREA * Q * W: return None
    return quad


def verify(q_xy, d_xy, pairs, d_shape, q_shape):
    """Tentative matches -> (number of inliers, projected BBDD outline or None)."""
    if len(pairs) < MIN_MATCHES: return 0, None
    cv2.setRNGSeed(SEED)
    H, mask = cv2.findHomography(d_xy[pairs[:, 1]], q_xy[pairs[:, 0]], cv2.RANSAC, RANSAC_PX, maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
    if H is None: return 0, None
    quad = plausible(H, d_shape, q_shape)
    return (int(mask.sum()), quad) if quad is not None else (0, None)


class Matcher:
    def __init__(self, feats, shapes, norm, dtype, ratio, cross=False):
        self.feats, self.shapes = feats, shapes
        self.params = (norm, dtype, ratio, cross)
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def match_all(self, q_desc):
        return list(self.pool.map(lambda f: tentative_matches(q_desc, f[1], *self.params), self.feats))

    def verify_all(self, q_xy, all_pairs, q_shape):
        res = list(self.pool.map(lambda a: verify(q_xy, a[0][0], a[1], a[2], q_shape), zip(self.feats, all_pairs, self.shapes)))
        return np.array([r[0] for r in res]), [r[1] for r in res]

    def search(self, q_xy, q_desc, q_shape):
        """Query features -> region record: inliers and tentative matches per BBDD image, plus the outlines."""
        pairs = self.match_all(q_desc)
        inliers, quads = self.verify_all(q_xy, pairs, q_shape)
        return {'inliers': inliers, 'matches': np.array([len(p) for p in pairs]), 'quads': quads, 'pairs': pairs, 'xy': q_xy}

    def search_second(self, first, q_shape, grow=1.05):
        """Sequential RANSAC: remove the keypoints inside the painting already found and verify again."""
        best = int(np.argmax(first['inliers'])); quad = first['quads'][best]
        if quad is None: return None
        c = quad.mean(0); mask = np.zeros(q_shape, np.uint8)
        cv2.fillConvexPoly(mask, ((quad - c) * grow + c).astype(np.int32), 1)
        x = np.clip(first['xy'][:, 0].astype(int), 0, q_shape[1] - 1); y = np.clip(first['xy'][:, 1].astype(int), 0, q_shape[0] - 1)
        outside = mask[y, x] == 0
        pairs = [p[outside[p[:, 0]]] for p in first['pairs']]
        inliers, quads = self.verify_all(first['xy'], pairs, q_shape)
        inliers[best] = 0
        return {'inliers': inliers, 'matches': np.array([len(p) for p in pairs]), 'quads': quads, 'pairs': pairs, 'xy': first['xy']}
