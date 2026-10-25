"""Strictly texture-based descriptors used in Week 3 Task 2.

This module is intentionally separate from ``run_w3.py``: Task 2 is evaluated
on cropped QSD1-W3 queries, whereas the blind Task 5 pipeline keeps its own
retrieval method.  The four Task-2 families are LBP, block DCT, Haar wavelets
and a Gabor filter bank.  HOG is deliberately not part of this module because
it is an edge/shape descriptor rather than a strict texture descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern

EPS = 1e-10


def grey_square(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Convert a BGR/RGB image to a fixed-size greyscale square."""
    if img.ndim == 2:
        grey = img
    else:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + EPS)


def desc_lbp(img: np.ndarray, size: int = 256, points: int = 8, radius: float = 2,
             grid: int = 10) -> np.ndarray:
    """Spatial LBP histograms retaining all 256 codes."""
    grey = grey_square(img, size)
    codes = local_binary_pattern(grey, points, radius, method="default")
    edges = np.linspace(0, size, grid + 1, dtype=int)
    out: list[float] = []
    for i in range(grid):
        for j in range(grid):
            block = codes[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            # ``P`` neighbours produce ``2**P`` possible binary codes.  Using
            # ``P**2`` would make the histogram length depend on whether a
            # particular block happens to contain the largest code.
            hist = np.bincount(block.astype(np.uint8).ravel(), minlength=2 ** points).astype(np.float32)
            out.extend((hist / (hist.sum() + EPS)).tolist())
    return np.asarray(out, np.float32)


def _zigzag_indices(n: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for diagonal in range(2 * n - 1):
        coords = [(i, diagonal - i) for i in range(n) if 0 <= diagonal - i < n]
        order.extend(coords if diagonal % 2 == 0 else coords[::-1])
    return order


def desc_dct(img: np.ndarray, size: int = 256, grid: int = 4, keep: int = 36) -> np.ndarray:
    """Block DCT texture descriptor: zig-zag coefficients, no DC, block-normalised."""
    grey = grey_square(img, size).astype(np.float32) / 255.0
    edges = np.linspace(0, size, grid + 1, dtype=int)
    block_size = size // grid
    zz = _zigzag_indices(block_size)
    out: list[float] = []
    for i in range(grid):
        for j in range(grid):
            block = grey[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            coeff = cv2.dct(block)
            vals = np.asarray([coeff[a, b] for a, b in zz[:keep + 1]][1:], np.float32)
            vals /= np.linalg.norm(vals) + EPS
            out.extend(vals.tolist())
    return np.asarray(out, np.float32)


def desc_wavelet(img: np.ndarray, size: int = 256, level: int = 3, grid: int = 8) -> np.ndarray:
    """Haar wavelet energy statistics over spatial blocks."""
    grey = grey_square(img, size).astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(grey, "haar", level=level)
    approx = coeffs[0]
    bands = [(approx - approx.mean()) / (approx.std() + 1e-6)]
    bands.extend(band for level_coeffs in coeffs[1:] for band in level_coeffs)
    out: list[float] = []
    for band in bands:
        h, w = band.shape
        n = min(grid, h, w)
        ys, xs = np.linspace(0, h, n + 1, dtype=int), np.linspace(0, w, n + 1, dtype=int)
        for i in range(n):
            for j in range(n):
                block = band[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                out.extend((float(np.mean(np.abs(block))), float(np.std(block))))
    return _l2(np.asarray(out, np.float32))


@dataclass(frozen=True)
class GaborConfig:
    """Selected Gabor bank from the Task-2 sweep."""

    size: int = 256
    scales: tuple[float, ...] = (4.0, 8.0, 16.0)
    orientations: int = 6
    grid: int = 12
    include_std: bool = True
    sigma_factor: float = 0.56
    gamma: float = 0.5


def desc_gabor(img: np.ndarray, config: GaborConfig = GaborConfig()) -> np.ndarray:
    """Gabor filter-bank energy statistics, a classical texture descriptor.

    For each wavelength/orientation response, the descriptor stores the mean
    absolute response and (by default) its standard deviation in each spatial
    block.  The selected sweep setting is 256x256, three scales, six
    orientations and a 12x12 grid: 12*12*3*6*2 = 5,184 values.
    """
    grey = grey_square(img, config.size).astype(np.float32) / 255.0
    theta_values = np.arange(config.orientations, dtype=np.float32) * np.pi / config.orientations
    edges = np.linspace(0, config.size, config.grid + 1, dtype=int)
    out: list[float] = []
    for wavelength in config.scales:
        sigma = config.sigma_factor * wavelength
        ksize = int(max(7, round(6 * sigma))) | 1
        for theta in theta_values:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, float(theta), wavelength,
                                        config.gamma, 0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            response = cv2.filter2D(grey, cv2.CV_32F, kernel)
            energy = np.abs(response)
            for i in range(config.grid):
                for j in range(config.grid):
                    block = energy[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
                    out.append(float(block.mean()))
                    if config.include_std:
                        out.append(float(block.std()))
    return _l2(np.asarray(out, np.float32))


def zscore_fusion(distance_vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Equal-weight late fusion for texture descriptors.

    Descriptors have different dimensionalities and metrics, so their ranked
    distance vectors are standardised over the database before summation.
    """
    return zscore_fusion_sequence(list(distance_vectors))


def zscore_fusion_sequence(distance_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not distance_vectors:
        raise ValueError("at least one distance vector is required")
    result = np.zeros_like(np.asarray(distance_vectors[0], dtype=np.float32))
    for distances in distance_vectors:
        d = np.asarray(distances, np.float32)
        result += (d - d.mean()) / (d.std() + EPS)
    return result


if __name__ == "__main__":
    demo = np.zeros((128, 128, 3), np.uint8)
    print("LBP", desc_lbp(demo).shape)
    print("DCT", desc_dct(demo).shape)
    print("wavelet", desc_wavelet(demo).shape)
    print("Gabor", desc_gabor(demo).shape)
