# C1 – Week 2: 3D block histograms and colour-based background removal

## Method (method1)

**Retrieval descriptor**
1. Remove a 5% margin on each side of the image (robust to imperfect crops).
2. Normalise the illumination: scale each RGB channel to mean 128.
3. Convert to YCrCb.
4. Compute a 3D colour histogram with 6×12×12 bins (Y × Cr × Cb).
   - The bins cover only the real range of each channel: equal-width bins between the 0.5 and 99.5 percentiles of the
     museum (BBDD) pixels, instead of the whole 0–255 scale, where most colour bins would stay empty.
5. Spatial pyramid: one histogram per block in 1×1 + 2×2 + 4×4 + 8×8 grids, each block normalised, level weights 1:2:4:8.
6. Rank the BBDD paintings with the Hellinger kernel and keep the 10 best.

**Background removal (QST2), colour only**
1. Wall model: the professors mentioned in class that there are at least 15 px of wall around every painting, so we take
   the 15-px strip along the border of the photo as wall and compute its mean colour and covariance in Lab (with the 80%
   of strip pixels closest to the median colour, to ignore labels or shadows).
2. Distance of every pixel to the wall model (Mahalanobis distance, in units of the wall's own variation).
3. Threshold chosen automatically for each photo with Otsu.
4. Clean the mask: closing + opening, keep the largest connected component, fill holes, fill rows / columns.
5. Crop the painting with the bounding box of the mask and describe it as above.

## Results on the development sets

| | mAP@1 | mAP@5 |
|---|---|---|
| QSD1-W2 | 0.967 | 0.967 |
| QSD2-W2 (cropped with our masks) | 1.000 | 1.000 |

| QSD2-W2 masks (mean over the 30 images) | Precision | Recall | F1 |
|---|---|---|---|
| method1 | 0.953 | 0.963 | 0.954 |

## How to run

The datasets must be **in the same folder as `run_w2.py`**:

```
week2/
├── run_w2.py
├── requirements.txt
├── BBDD/          museum database (bbdd_00000.jpg ... bbdd_00286.jpg)
├── qst1_w2/       test set 1
├── qst2_w2/       test set 2
├── qsd1_w2/       development set 1 (for --check)
└── qsd2_w2/       development set 2 (for --check)
```

Then, from this folder:

```bash
pip install -r requirements.txt
python run_w2.py --check     # results on QSD1 / QSD2 (mAP and mask precision / recall / F1) + test predictions
python run_w2.py             # test predictions only
```

## Output

```
results/QST1/method1/result.pkl
results/QST2/method1/result.pkl
results/QST2/method1/*.png
```

- `result.pkl`: a Python list of lists of integers, one list per query (in file order) with the 10 most similar BBDD
  ids, best first. For example `[[35, 68, ...], [131, 69, ...], ...]`, where `35` is `bbdd_00035.jpg`.
- `*.png`: one binary mask per QST2 image (0 = background, 255 = painting), same size and same name as the query
  (`00001.jpg` → `00001.png`).

`run_w2.py` checks this format before saving the files.
