# C1 – Week 4 · Team 4

Retrieval of museum paintings with keypoints and local descriptors, including queries that are not in the BBDD.

## Method (method1)

### Task 1 – Keypoints and descriptors
Three options were tested. The chosen one is **SIFT**.
1. **SIFT**: DoG keypoints + 128-float descriptor, L2 distance.
2. **RootSIFT**: same keypoints; each descriptor is L1-normalised and square-rooted.
3. **ORB**: FAST keypoints on a pyramid + rotated BRIEF, Hamming distance.

Before extraction, noisy images are filtered as in week 3 and the longest side is limited to 800 px.
SIFT keeps up to 2000 keypoints (contrast threshold 0.02). The features of the BBDD are cached in `cache/`.

### Task 2 – Matching and verification
1. For each query keypoint, find its two nearest neighbours in each BBDD image (exhaustive search, so it is deterministic).
2. Keep the match if it passes **Lowe's ratio test** (0.75). Each BBDD keypoint is used only once.
3. Fit a **homography with RANSAC** (5 px, fixed seed). Reject it if the projected painting is not convex, is mirrored, or has an implausible size.
4. The score of a BBDD image is its **number of inliers**. Its number of tentative matches breaks ties.

Cross-check was also tested. It gave no improvement and takes more time, so it is not used.

### Up to two paintings per photo (hybrid)
1. Match the whole photo: the best BBDD image is the first painting.
2. Remove the keypoints inside that painting and verify again (sequential RANSAC). The new best image is the second painting if it has enough inliers.
3. If no second painting is verified but the week-3 segmentation finds two regions, search the other region on its own and keep its ranking. This keeps the paintings in the right positions.
4. Order the paintings from left to right, or top to bottom if stacked.

### Task 2 (II) – Unknown queries
A photo whose best score is below **T = 25 inliers** is returned as `[[-1]]`.
On QSD1-W4 every T in (7, 53] gives the best F1, so F1 alone does not fix T. The value 25 is the best of that range on the independent check below.

## Results on QSD1-W4 (30 photos, 39 entries: 29 paintings + 10 unknown photos)

| Option | Unknowns: P | R | F1 | F1 leave-one-out | mAP@1 / @5 no discard | mAP@1 / @5 discard | mAP@1 known only |
|---|---|---|---|---|---|---|---|
| **SIFT hybrid (chosen)** | 0.909 | 1.000 | 0.952 | 0.952 | 0.718 / 0.718 | **0.974 / 0.974** | 0.966 |
| RootSIFT hybrid | 0.909 | 1.000 | 0.952 | 0.952 | 0.718 / 0.718 | 0.974 / 0.974 | 0.966 |
| ORB hybrid | 1.000 | 1.000 | 1.000 | 0.952 | 0.718 / 0.718 | 0.974 / 0.974 | 0.966 |
| SIFT whole photo | 0.909 | 1.000 | 0.952 | 0.952 | 0.667 / 0.679 | 0.923 / 0.936 | 0.897 |
| SIFT week-3 segmentation | 0.909 | 1.000 | 0.952 | 0.952 | 0.667 / 0.667 | 0.897 / 0.897 | 0.862 |
| Week 3 (texture fusion) | 0.909 | 1.000 | 0.952 | 0.952 | 0.692 / 0.692 | 0.923 / 0.923 | 0.897 |

Unknown is the positive class. Week 3 has no discard of its own: its discard column uses a threshold on its best fused distance, optimised in the same way.
`--check` writes all the options, with cross-check, to `results/dev_summary.md`.

### Choosing the method: independent check
QSD1-W4 cannot separate the three descriptors: their mAP is the same, and their leave-one-out F1 too. ORB's F1 of 1.000 depends on one photo with 9 inliers, against 7 for the worst unknown.

So the method was chosen on the QSD1-W3 and QSD2-W3 sets (60 photos, all known). A known query is simulated with the full BBDD, and an unknown one by removing its right paintings from the BBDD.

| Option | T | Known missed | False known |
|---|---|---|---|
| **SIFT (chosen)** | 25 | 3.3 % | 0.0 % |
| SIFT | 30 | 6.7 % | 0.0 % |
| RootSIFT | 30 | 6.7 % | 1.7 % |
| ORB | 8 | 1.7 % | 8.3 % |
| Week 3 | 8.94 | 30.0 % | 6.7 % |

The difference between SIFT and RootSIFT is a single photo, so it is weak evidence. The BBDD also contains near-duplicates (179 / 219, 13 / 52), which explain the highest scores among the simulated unknowns.

### Known limitations
* Paintings with almost no local structure fail. Painting 128 (plastic with reflections) is lost in QSD1-W4 and returned as unknown.
* Painting 11 (image 16) is correct with only 7 inliers, below T. It is kept by the hybrid rule, and this single hit is worth 0.026 of mAP.
* A photo with one known and one unknown painting would get a ranking for the unknown one, not `[-1]`. QSD1-W4 has no such photo.
* With 10 unknown photos, two of them repeated paintings, these figures have a large uncertainty.

## Files

```
week4/
├── run_w4.py           main script (run this one)
├── keypoints.py        Task 1: SIFT, RootSIFT, ORB and the BBDD cache
├── matching.py         Task 2: ratio test, cross-check, RANSAC, sequential RANSAC
├── unknowns.py         Task 2 (II): F1 threshold and leave-one-out
├── baseline_w3.py      Task 3: week-3 system on the same queries
├── requirements.txt
├── qsd1_w4/, qst1_w4/  development and test sets (not in the repository)
├── cache/              BBDD features (not in the repository)
└── results/            development table and test predictions
```

The code also imports `../week3/run_w3.py` and `../week3/task2_texture.py`. The BBDD is read from a `BBDD/` folder here or, failing that, from `../week3/BBDD`.

## How to run

From the `week4/` folder:

```bash
pip install -r requirements.txt
python run_w4.py --check     # QSD1-W4, all the options, week 3 + test predictions (about 25 min)
python run_w4.py             # test predictions only (about 1 min for 30 photos, once the cache exists)
```

Results are reproducible: the matching is exhaustive and the RANSAC seed is fixed.

## Output (format of the week-4 instructions)

```
results/QST1/method1/result.pkl    [[[painting 1 top-10], [painting 2 top-10]], [[-1]], [[top-10]], ...]
```
All ids are Python `int`; `run_w4.py` checks the format before saving. Upload it to the Drive as `TeamX/week4/QST1/method1/result.pkl`.
