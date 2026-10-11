# C1 – Week 2: 3D block / pyramid colour histograms and colour-based background removal

## Methods

The week-2 instructions ask for a single method, so we submit **method1**. method2 is the alternative we compared it with
(`--all-methods` also generates it).

**Retrieval descriptor.** Remove a 5% inner margin on each side of the image (robustness to imperfect crops). Normalise the illumination by scaling each RGB channel to mean 128. Then compute a **3D colour histogram per
block**, normalise every block to sum 1, concatenate the blocks, and compare descriptors with the Hellinger kernel.

| | Colour space | Bins (per channel) | Spatial layout | Dim. |
|---|---|---|---|---|
| **method1** | YCrCb | 6×12×12, equal-width over the real range of each channel (0.5–99.5 percentile of the BBDD pixels) | spatial pyramid 1×1 + 2×2 + 4×4 + 8×8, level weights 1:2:4:8 | 73 440 |
| method2 (not submitted) | HSV | 8×6×6, uniform | 5×5 blocks | 7 200 |

**Background removal (QST2), colour only.** We build a Gaussian model of the wall colour for each image: the robust mean
and covariance, in Lab, of a strip along the image border. Each pixel gets its Mahalanobis distance to that model, and an
Otsu threshold separates painting from wall. The binary mask is post-processed with morphology, the largest connected
component, hole filling and row/column filling. The painting is then cropped with the bounding box of the mask.

## Results on the development sets

| | QSD1 mAP@1 / @5 | QSD2 (our masks) mAP@1 / @5 |
|---|---|---|
| Week 1 best (HSV 1D, χ²) | 0.900 / 0.950 | 0.500 / 0.578 |
| **method1** | **0.967 / 0.967** | **0.967 / 0.983** |
| method2 (not submitted) | 0.967 / 0.967 | 0.933 / 0.961 |

QSD2 masks: precision 0.953, recall 0.966, F1 0.956 (mean over the 30 images).

All the experiments and the reasons behind each choice are in `c1-week2.ipynb`.

## How to run

Put `qsd1_w2/`, `qsd2_w2/` (and, when published, `qst1_w2/`, `qst2_w2/`) inside `week2/`. The BBDD is read from
`../week1/BBDD`. Then run from `week2/`:

```bash
pip install -r requirements.txt
python run_w2.py --check   # optional: mAP on QSD1 / QSD2 and mask P/R/F1
python run_w2.py           # generates the QST1 / QST2 predictions
```

## Output

```
results/QST1/method1/result.pkl
results/QST2/method1/result.pkl
results/QST2/method1/*.png
```

- Each `result.pkl` is a list of lists of `int`: one list per query (in file order) with the 10 most similar BBDD ids,
  best first.
- The masks are `.png` files with values 0 / 255 and the same name as the query (`00001.jpg` → `00001.png`).
