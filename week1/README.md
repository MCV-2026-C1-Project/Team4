# Team4

## C1 – Week 1: Museum painting retrieval with 1D color histograms

### Methods
Both methods share the same pipeline:

1. **Resize** the image so its longest side is 512 px.
2. **Normalize the illumination**: scale each RGB channel so its mean is 128. This removes the brightness and color cast of each photo.
3. **Convert** to the method's color space and compute a **1D histogram per channel**, normalized to sum 1 so the image size does not matter. The three histograms are concatenated into one descriptor.
4. **Compare** the query descriptor with every BBDD descriptor and return the 10 closest paintings.

**Method 1 – HSV, 16 bins, χ² distance (48-D descriptor).**
HSV separates the color (H, S) from the brightness (V). 16 bins are coarse enough to tolerate small color shifts between photos. χ² divides each bin difference by the bin size, so small but distinctive colors also count, not only the dominant background.

**Method 2 – CIELab, 32 bins, Hellinger kernel (96-D descriptor).**
CIELab is perceptually uniform: equal distances mean similar perceived color differences. The Hellinger kernel, Σ√(h·H), measures the overlap between two distributions. Its square root reduces the weight of large bins, with the same effect as χ².

| | Color space | Bins / channel | Distance | mAP@1 | mAP@5 |
|---|---|---|---|---|---|
| Method 1 | HSV | 16 | χ² | 0.900 | 0.950 |
| Method 2 | CIELab | 32 | Hellinger | 0.900 | 0.922 |

mAP values evaluated on QSD1 against its ground truth (`gt_corresps.pkl`). Both methods were selected after testing 5 color spaces, 5 bin sizes (8–128) and 5 distances on QSD1. Without the illumination normalization, the best mAP@1 was 0.50.

### How to run
Place `BBDD/`, `qsd1_w1/` and `qst1_w1/` inside `week1/`, then run from `week1/`:

```bash
pip install -r requirements.txt
python run_qst1.py --check   # optional: prints mAP@1 / mAP@5 on QSD1 (no files saved)
python run_qst1.py           # generates the QST1 predictions
```

### Output
The predictions are saved in `week1/results/QST1/` (created when `qst1_w1/` exists):

```
results/QST1/method1/result.pkl
results/QST1/method2/result.pkl
```

Each `result.pkl` is a pickled Python list of lists of integers: one list per query (in file order) with the 10 most similar BBDD ids, best first. For example, `[[120, 182, ...], [161, 170, ...], ...]`, where `120` refers to `bbdd_00120.jpg`.
