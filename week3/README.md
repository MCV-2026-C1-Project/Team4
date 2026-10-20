# C1 – Week 3: noise filtering, texture descriptors and retrieval of up to two paintings per photo

All the experiments and the reasons behind every choice are in `c1-week3.ipynb`; `explanation.md` explains the work in Spanish;
`C1_W3_all_tests.xlsx` contains every experiment table.

## Method (method1)

**Task 1 – noise.** An image is noisy if more than 0.5 % of its pixels are *isolated impulses* (they differ by more than 40 grey
levels from all their 8 neighbours; measured at full resolution with a dilation / erosion by a 3×3 ring). Only those images are
filtered, with a **3×3 median applied twice** (best PSNR / SSIM among 12 linear and non-linear filters).

**Task 2 – texture descriptors** (grey image resized to a fixed square, per block): LBP (uniform, multi-scale and all 256 codes),
block DCT (zig-zag, first N coefficients, no DC, block-normalised), Haar wavelets (mean |c| and std per sub-band and block) and HOG.

**Task 3 – background removal, up to two paintings.** Team 8's W2 wall model (the wall chroma a, b as a quadratic function of the
lightness L, robust fit on a **1 %** border strip, Mahalanobis distance ≤ 5.5) → wall = wall-like pixels connected to the border →
up to 2 components → per painting: hole filling, trimming of sparse rows/columns, row/column span filling and quadrilateral fit →
paintings whose colour equals the wall are recovered from enclosed edges → left-to-right order.

**Retrieval.** For each painting (bounding-box crop): **HOG** (128×128, 16-px cells, 2×2 blocks, 8 orientations, L2) +
**Haar wavelet** (256×256, 3 levels, 8×8 blocks, L1) + our **W2 colour descriptor** (YCrCb 3D pyramid, Hellinger). For each query
the three distance vectors are z-scored over the 287 BBDD images and added (late fusion).

## Results on the development sets

| | mAP@1 | mAP@5 |
|---|---|---|
| QSD1-W3, texture only (HOG; wavelet and DCT tie) | 1.000 | 1.000 |
| QSD1-W3, final method | 1.000 | 1.000 |
| QSD2-W3 with our masks (39 paintings) | 1.000 | 1.000 |

| QSD2-W3 masks (mean over 30 photos) | Precision | Recall | F1 |
|---|---|---|---|
| method1 | 0.988 | 0.965 | 0.975 |

## How to run

```
week3/
├── c1-week3.ipynb      all the experiments (≈20 min)
├── run_w3.py           test predictions only (< 2 min)
├── requirements.txt
├── BBDD/               museum database (or ../week2/week2_submission/BBDD, ../week1/BBDD)
├── qsd1_w3/, qsd2_w3/  development sets (as downloaded)
├── qsd1_w2/, qsd2_w2/  optional, extra validation in the notebook (or ../week2/...)
└── qst1_w3/, qst2_w3/  test sets (published Thursday 22 Oct)
```

```bash
pip install -r requirements.txt
python run_w3.py --check     # QSD1-W3 / QSD2-W3 results + test predictions
python run_w3.py             # test predictions only
```

## Output (format of the week-3 instructions, with the extra list level)

```
results/QST1/method1/result.pkl    [[[top-10]], [[top-10]], ...]
results/QST2/method1/result.pkl    [[[painting 1 top-10], [painting 2 top-10]], [[...]], ...]
results/QST2/method1/*.png         binary masks 0 / 255, same name and size as the query (00001.jpg -> 00001.png)
```
All ids are Python `int`; the paintings of a photo are given left to right. `run_w3.py` checks the format before saving.
