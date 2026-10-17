"""
C1 - Week 1 - Task 4: predictions for the blind test set QST1.

Methods:
    method1: HSV,    16 bins per channel, chi-square distance
    method2: CIELab, 32 bins per channel, Hellinger kernel

For every query image in qst1_w1/, retrieve the K=10 most similar BBDD paintings with our two
final methods and save them as a list of lists of integers (one list per query, in file order):
    results/QST1/method1/result.pkl
    results/QST1/method2/result.pkl

Usage (from the week1 folder, next to BBDD/ and qst1_w1/):
    python run_qst1.py                  # generate the QST1 predictions
    python run_qst1.py --check          # sanity check on QSD1 first (prints mAP@1 / mAP@5)
    python run_qst1.py --qst qst1_w1    # if the test folder has a different name
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

K = 10            # results per query required by the submission
MAX_SIDE = 512    
EPS = 1e-10

METHODS = {
    "method1": {"space": "HSV", "bins": 16, "measure": "chi2"},
    "method2": {"space": "LAB", "bins": 32, "measure": "hellinger"},
}

# color space -> (OpenCV conversion from BGR, value range of each channel)
COLOR_SPACES = {
    "HSV": (cv2.COLOR_BGR2HSV, [(0, 180), (0, 256), (0, 256)]),   # OpenCV hue is in [0, 180)
    "LAB": (cv2.COLOR_BGR2LAB, [(0, 256)] * 3),
}


# ----------------------------------------------------------------------------- data
def image_id(path: Path) -> int:
    # 'bbdd_00120.jpg' -> 120, '00007.jpg' -> 7
    return int(path.stem.split("_")[-1])


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read {path}")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def load_folder(folder: Path, pattern: str = "*.jpg"):
    paths = sorted(folder.glob(pattern), key=image_id)
    if not paths:
        raise FileNotFoundError(f"No '{pattern}' images found in {folder}")
    return [image_id(p) for p in paths], [load_image(p) for p in paths]


# ----------------------------------------------------------------------------- descriptor
def channel_mean_128(img, target=128):
    # Illumination normalization: scale each RGB channel so that its mean is 128
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    return np.clip(f * (target / (means + EPS)), 0, 255).astype(np.uint8)


def color_histogram(img_bgr, space, bins):
    # Concatenation of the per-channel 1D histograms, each normalized to sum 1
    conversion, ranges = COLOR_SPACES[space]
    x = cv2.cvtColor(img_bgr, conversion)
    hists = []
    for ch, (lo, hi) in enumerate(ranges):
        h = cv2.calcHist([x], [ch], None, [bins], [lo, hi]).ravel()
        hists.append(h / (h.sum() + EPS))
    return np.concatenate(hists).astype(np.float32)


def descriptors(images, space, bins):
    return np.stack([color_histogram(channel_mean_128(img), space, bins) for img in images])


# ----------------------------------------------------------------------------- measures (lower = more similar)
def chi2(h, H):
    return np.sum((H - h) ** 2 / (H + h + EPS), axis=1)


def hellinger(h, H):
    return -np.sum(np.sqrt(H * h), axis=1)


MEASURES = {"chi2": chi2, "hellinger": hellinger}


# ----------------------------------------------------------------------------- retrieval / evaluation
def retrieve(query_desc, db_desc, db_ids, measure, k=K):
    # Top-k BBDD ids (plain Python ints) for each query -> list of lists
    out = []
    for q in query_desc:
        order = np.argsort(measure(q, db_desc), kind="stable")[:k]
        out.append([int(db_ids[i]) for i in order])
    return out


def apk(actual, predicted, k=10):
    # Average precision at k (benhamner/Metrics)
    predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(actual, predicted, k=10):
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Generate QST1 predictions (C1 week 1, Task 4)")
    parser.add_argument("--bbdd", default="BBDD", help="museum database folder")
    parser.add_argument("--qst", default="qst1_w1", help="blind test query folder")
    parser.add_argument("--qsd", default="qsd1_w1", help="development query folder (for --check)")
    parser.add_argument("--out", default="results/QST1", help="output folder")
    parser.add_argument("--check", action="store_true", help="evaluate the methods on QSD1 before predicting")
    args = parser.parse_args()

    print("Loading BBDD ...")
    bbdd_ids, bbdd_imgs = load_folder(Path(args.bbdd), "bbdd_*.jpg")
    print(f"  {len(bbdd_imgs)} museum images")

    if args.check:
        qsd_ids, qsd_imgs = load_folder(Path(args.qsd))
        with open(Path(args.qsd) / "gt_corresps.pkl", "rb") as f:
            gt = pickle.load(f)
        print(f"\nSanity check on QSD1 ({len(qsd_imgs)} queries):")
        for name, cfg in METHODS.items():
            db = descriptors(bbdd_imgs, cfg["space"], cfg["bins"])
            qs = descriptors(qsd_imgs, cfg["space"], cfg["bins"])
            preds = retrieve(qs, db, bbdd_ids, MEASURES[cfg["measure"]], k=K)
            print(f"  {name}: mAP@1 = {mapk(gt, preds, 1):.3f}   mAP@5 = {mapk(gt, preds, 5):.3f}")

    qst_dir = Path(args.qst)
    if not qst_dir.exists():
        print(f"\n'{qst_dir}' not found. Put the QST1 folder next to BBDD/ (or pass --qst <folder>).")
        return

    qst_ids, qst_imgs = load_folder(qst_dir)
    print(f"\nQST1: {len(qst_imgs)} queries ({qst_dir})")

    for name, cfg in METHODS.items():
        db = descriptors(bbdd_imgs, cfg["space"], cfg["bins"])
        qs = descriptors(qst_imgs, cfg["space"], cfg["bins"])
        preds = retrieve(qs, db, bbdd_ids, MEASURES[cfg["measure"]], k=K)

        # Format required by the instructions: list of lists of int ids, one list of K results per query
        assert len(preds) == len(qst_imgs)
        assert all(len(p) == K and all(isinstance(i, int) for i in p) for p in preds)

        out = Path(args.out) / name / "result.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(preds, f)
        print(f"  {name}: saved {out}   (first query -> {preds[0]})")

    print("\nDone. Upload to the Drive as TeamX/week1/QST1/method1/result.pkl and TeamX/week1/QST1/method2/result.pkl")


if __name__ == "__main__":
    main()
