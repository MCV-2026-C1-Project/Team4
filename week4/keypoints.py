"""
C1 - Week 4 - Task 1: keypoints + local descriptors (the three options tested) and the BBDD cache.

  sift      DoG keypoints + SIFT descriptor (128 floats), L2
  rootsift  same keypoints, descriptor L1-normalised + square root (Arandjelovic & Zisserman, 2012), L2
  orb       FAST keypoints on a pyramid + rotated BRIEF (256 bits), Hamming

A feature set is a tuple (xy, desc): xy is N x 2 (float32), in pixels of the image it was extracted from.
"""

from pathlib import Path

import cv2
import numpy as np

METHODS = {
    'sift': {'detector': 'sift', 'norm': cv2.NORM_L2, 'dtype': cv2.CV_32F, 'root': False, 'ratio': 0.75},
    'rootsift': {'detector': 'sift', 'norm': cv2.NORM_L2, 'dtype': cv2.CV_32F, 'root': True, 'ratio': 0.75},
    'orb': {'detector': 'orb', 'norm': cv2.NORM_HAMMING, 'dtype': cv2.CV_32S, 'root': False, 'ratio': 0.8},
}
DETECTORS = {
    'sift': dict(nfeatures=2000, contrastThreshold=0.02),
    'orb': dict(nfeatures=3000, scaleFactor=1.2, nlevels=8, fastThreshold=10),
}


def root_sift(desc, eps=1e-7):
    d = desc / (np.abs(desc).sum(1, keepdims=True) + eps)
    return np.sqrt(d).astype(np.float32)


def detect(img, detector, params=None):
    """BGR image -> (xy, raw descriptors) with the given detector ('sift' or 'orb')."""
    params = params or DETECTORS[detector]
    make = cv2.SIFT_create if detector == 'sift' else cv2.ORB_create
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = make(**params).detectAndCompute(grey, None)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128 if detector == 'sift' else 32), np.float32 if detector == 'sift' else np.uint8)
    return np.float32([k.pt for k in kps]), desc


def extract(img, method):
    xy, desc = detect(img, METHODS[method]['detector'])
    return xy, (root_sift(desc) if METHODS[method]['root'] else desc)


# ----------------------------------------------------------------------------- BBDD cache
def cache_name(detector, work):
    tag = '_'.join(f'{k}{v}' for k, v in DETECTORS[detector].items())
    return f'bbdd_{detector}_w{work}_{tag}.npz'


def load_bbdd(paths, ids, method, work, prepare, cache_dir='cache'):
    """Features of every BBDD image. The raw detector output is cached, so rootsift reuses the sift cache."""
    detector = METHODS[method]['detector']
    f = Path(cache_dir) / cache_name(detector, work)
    feats = None
    if f.exists():
        z = np.load(f)
        if np.array_equal(z['ids'], ids):
            cuts, xy, desc = z['offsets'], z['xy'], z['desc']       # each z[...] reads the whole array from disk
            feats = [(xy[a:b], desc[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    if feats is None:
        feats = [detect(prepare(cv2.imread(str(p))), detector) for p in paths]
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, ids=np.asarray(ids), offsets=np.cumsum([0] + [len(x) for x, _ in feats]),
                 xy=np.vstack([x for x, _ in feats]), desc=np.vstack([d for _, d in feats]))
    return [(x, root_sift(d)) for x, d in feats] if METHODS[method]['root'] else feats
