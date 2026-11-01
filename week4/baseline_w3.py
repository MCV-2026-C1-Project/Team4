"""
C1 - Week 4 - Task 3: the best system of week 3 (texture fusion + up to 2 paintings) on the same query set.

The week-3 descriptors, denoising and segmentation are reused as they are. Week 3 has no unknown queries, so its answer
is always a ranking: that is the reference. A discarded version is also given to compare like with like: the score of a
painting is minus its best fused distance, and the threshold comes from the same F1 optimisation as week 4.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'week3'))
from run_w3 import K, crop, denoise, resize_max, segment  # noqa: E402
from task2_texture import desc_dct, desc_gabor, desc_lbp, desc_wavelet  # noqa: E402

DESCRIPTORS = (desc_lbp, desc_dct, desc_wavelet, desc_gabor)


class BaselineW3:
    def __init__(self, paths, ids):
        self.ids = np.asarray(ids)
        imgs = [resize_max(denoise(cv2.imread(str(p)))) for p in paths]
        self.db = [np.stack([f(x) for x in imgs]) for f in DESCRIPTORS]

    def fused(self, img):
        total = 0
        for f, db in zip(DESCRIPTORS, self.db):
            d = np.abs(db - f(img)).sum(1)
            total = total + (d - d.mean()) / (d.std() + 1e-10)
        return total

    def crops(self, photo):
        img = denoise(photo)
        return [crop(img, m) for m in segment(img)]

    def describe(self, photo):
        """Photo -> (rankings of its paintings, photo score)."""
        fused = [self.fused(c) for c in self.crops(photo)]
        return [[int(i) for i in self.ids[np.argsort(t, kind='stable')[:K]]] for t in fused], float(max(-t.min() for t in fused))
