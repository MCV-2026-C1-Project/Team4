# C1 – Week 3: noise filtering, texture descriptors and retrieval of up to two paintings per photo

All the experiments and the reasons behind every choice are in `c1-week3.ipynb`; `explanation.md` explains the work in Spanish;
`C1_W3_all_tests.xlsx` contains every experiment table.

## Method (method1)

**Task 1 – noise.** An image is noisy if more than 0.5 % of its pixels are *isolated impulses* (they differ by more than 40 grey
levels from all their 8 neighbours; measured at full resolution with a dilation / erosion by a 3×3 ring). Only those images are
filtered, with a **3×3 median applied twice** (best PSNR / SSIM among 12 linear and non-linear filters).

**Task 2 – strictly texture descriptors** (grey image resized to a fixed square, per block): LBP (uniform, multi-scale and all 256 codes),
block DCT (zig-zag, first N coefficients, no DC, block-normalised), Haar wavelets (mean |c| and std per sub-band and block) and a Gabor filter bank (mean energy + spread per scale/orientation/block). HOG is deliberately excluded from Task 2 because it is an edge/shape descriptor.

**Task 3 – background removal, up to two paintings.** Team 8's W2 wall model (the wall chroma a, b as a quadratic function of the
lightness L, robust fit on a **1 %** border strip, Mahalanobis distance ≤ 5.5) → wall = wall-like pixels connected to the border →
up to 2 components → per painting: hole filling, trimming of sparse rows/columns, row/column span filling and quadrilateral fit →
paintings whose colour equals the wall are recovered from enclosed edges → left-to-right order.

**Retrieval.** For each painting (bounding-box crop): **LBP + block DCT + Haar wavelets + Gabor**, all computed on grey images. For each query
the four texture distance vectors are z-scored over the 287 BBDD images and added (late fusion).

## Results on the development sets

| | mAP@1 | mAP@5 |
|---|---|---|
| QSD1-W3, texture only (LBP, DCT, wavelets or Gabor) | 1.000 | 1.000 |
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
├── BBDD/               museum database (287 jpg files)
├── qsd1_w3/, qsd2_w3/  development sets (as downloaded)
├── qst_w3/
│   ├── qst1_w3/        50 blind cropped queries
│   └── qst2_w3/        30 blind photos with background
└── results/             ready-to-upload method1 outputs
```

```bash
pip install -r requirements.txt
python run_w3.py --check     # if qsd1_w3/ and qsd2_w3/ are present
python run_w3.py             # uses BBDD/ and qst_w3/ automatically
```

The script can also be run with explicit paths from this directory:

```bash
python run_w3.py \
  --bbdd BBDD \
  --qst1 qst_w3/qst1_w3 \
  --qst2 qst_w3/qst2_w3 \
  --out results
```

## Output (format of the week-3 instructions, with the extra list level)

```
results/QST1/method1/result.pkl    [[[top-10]], [[top-10]], ...]
results/QST2/method1/result.pkl    [[[painting 1 top-10], [painting 2 top-10]], [[...]], ...]
results/QST2/method1/*.png         binary masks 0 / 255, same name and size as the query (00001.jpg -> 00001.png)
```
All ids are Python `int`; the paintings of a photo are given left to right. `run_w3.py` checks the format before saving.
