# C1 – Week 3 · Team 4

Noise filtering, texture descriptors and retrieval of up to two paintings per photo.

## Method (method1)

### Task 1 – Noise removal
1. Convert the photo to grey, at its original size.
2. For each pixel, take the maximum and the minimum of its 8 neighbours (grey-level dilation / erosion with a 3×3 ring).
3. A pixel is an **impulse** if it is more than 40 grey levels above that maximum or below that minimum.
4. If more than **0.5 %** of the pixels are impulses, the photo is noisy.
5. Only noisy photos are filtered, with a **3×3 median applied twice** (best PSNR / SSIM among 12 linear and non-linear filters).

### Task 2 – Texture descriptors
Only texture descriptors are used (HOG and colour are left out: HOG describes edges / shape, not texture).
All of them work on the grey image, resized to 256×256 and split into blocks.
1. **LBP**: each pixel is compared with its 8 neighbours → an 8-bit code; histogram of the 256 codes per block (P=8, R=2, 10×10 blocks).
2. **DCT**: per block, the first 36 coefficients in zig-zag order, without the DC term, block-normalised (4×4 blocks).
3. **Wavelet**: Haar, 3 levels; per sub-band and block, mean |c| and standard deviation, plus the approximation (8×8 blocks).
4. **Gabor**: bank of 18 filters (3 wavelengths × 6 orientations); per filter and block, mean and std of the response energy (12×12 blocks).

### Task 3 – Background removal (up to two paintings)
1. Take a thin strip (1 %) along the border of the photo as samples of the wall.
2. Model the wall: its colour (a, b in Lab) as a function of its brightness (L), with a robust fit, so lighting changes do not matter.
3. A pixel is **wall-like** if its colour is close to the colour the wall would have at its brightness (Mahalanobis distance ≤ 5.5).
4. **Wall** = wall-like pixels connected to the border of the photo.
5. Everything else is a painting candidate; keep the **2 largest** regions (≥ 2 % of the photo, the 2nd at least 10 % of the 1st).
6. Clean each painting: fill holes, remove sparse rows / columns, fill each row and column, fit a quadrilateral.
7. If fewer than 2 paintings were found, look for nearly rectangular regions closed by edges (paintings with the colour of the wall).
8. Order the paintings from left to right.

### Task 4 – Retrieval
1. Remove the noise (Task 1) and find the paintings (Task 3).
2. Crop each painting by its bounding box.
3. Describe it with the four texture descriptors: **LBP + DCT + wavelet + Gabor** (Task 2).
4. For each descriptor, compute the L1 distances to the 287 BBDD images and z-score them (subtract the mean, divide by the std).
5. Add the four z-scored distances and return the 10 closest paintings.

## Results on the development sets

| | mAP@1 | mAP@5 |
|---|---|---|
| QSD1-W3, texture only (LBP, DCT, wavelet or Gabor alone) | 1.000 | 1.000 |
| QSD1-W3, final method | 1.000 | 1.000 |
| QSD2-W3 with our masks (39 paintings) | 1.000 | 1.000 |

| QSD2-W3 masks (mean over 30 photos) | Precision | Recall | F1 |
|---|---|---|---|
| method1 | 0.988 | 0.965 | 0.975 |

## Files

```
week3_submission/
├── run_w3.py           main script: Tasks 1, 3, 4 and 5 (run this one)
├── task2_texture.py    the four texture descriptors of Task 2 (used by run_w3.py)
├── requirements.txt
├── BBDD/               museum database (287 paintings)
├── qsd1_w3/, qsd2_w3/  development sets (as downloaded)
├── qst_w3/             test sets: qst1_w3/ (50 queries), qst2_w3/ (30 photos)
└── results/            outputs of method1, ready to upload
```

* **`run_w3.py`** is the only file you need to run. It loads the BBDD, removes the noise, finds the paintings, describes them and
  writes the results.
* **`task2_texture.py`** only contains the descriptors (LBP, DCT, wavelet, Gabor). `run_w3.py` imports it, so both files must be in
  the same folder with these names.

## How to run

```bash
pip install -r requirements.txt
python run_w3.py --check     # QSD1-W3 / QSD2-W3 results + test predictions
python run_w3.py             # test predictions only (< 2 min)
```

Running it again overwrites the files in `results/` (same names, same content).

## Output (format of the week-3 instructions, with the extra list level)

```
results/QST1/method1/result.pkl    [[[top-10]], [[top-10]], ...]
results/QST2/method1/result.pkl    [[[painting 1 top-10], [painting 2 top-10]], [[...]], ...]
results/QST2/method1/*.png         binary masks 0 / 255, same name and size as the query (00001.jpg -> 00001.png)
```
All ids are Python `int`; the paintings of a photo are given left to right. `run_w3.py` checks the format before saving.
