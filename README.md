# Team4

## C1 – Week 1: Museum painting retrieval with 1D color histograms

### Methods
Before computing the histograms, each RGB channel is scaled to mean 128 to normalize the illumination. Each descriptor is a set of per-channel 1D histograms, normalized to sum 1 and concatenated.

| | Color space | Bins / channel | Distance | mAP@1 | mAP@5 |
|---|---|---|---|---|---|
| Method 1 | HSV | 16 | χ² | 0.900 | 0.950 |
| Method 2 | CIELab | 32 | Hellinger | 0.900 | 0.922 |

mAP values evaluated on QSD1 against its ground truth (`gt_corresps.pkl`).

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
