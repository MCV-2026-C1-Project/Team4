"""
Figures for the week-2 slides. Reads slides/data.json and slides/data_seg.json, writes slides/fig/*.png.

Run from the week2 folder:  python slides/figures.py
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_w2 as R  # noqa: E402

FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)
D = json.load(open(HERE / "data.json"))
S = json.load(open(HERE / "data_seg.json"))

TEAL, TEAL_L, ORANGE, GREY, GREY_L, INK, MUTED = "#1B7F79", "#A9D3CF", "#C8702A", "#9AA5AB", "#E3E7E9", "#263238", "#6B7880"
RED, BLUE = "#C0392B", "#2E6FBF"
plt.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#FCFCFA", "axes.facecolor": "#FCFCFA", "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlecolor": INK, "text.color": INK})


def px(w, h, **kw):
    """A figure that fills a w x h px box of the 1920x1080 slide canvas (100 px per inch, rendered at 2x)."""
    return plt.subplots(figsize=(w / 100, h / 100), layout="constrained", **kw)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, facecolor="#FCFCFA")
    plt.close(fig)
    print("saved", name)


bb_paths = R.image_paths(HERE.parent.parent / "week1" / "BBDD", "bbdd_*.jpg")
BB = {R.image_id(p): p for p in bb_paths}
q1, q2 = R.find_dir(HERE.parent / "qsd1_w2"), R.find_dir(HERE.parent / "qsd2_w2")
Q1P, Q2P = R.image_paths(q1), R.image_paths(q2)
rgb = lambda x: x[..., ::-1]


# ============================================================================= Task 1
def fig_bins_curve():
    """mAP@5 vs bins per channel, 3D global YCrCb, uniform vs range-adapted edges."""
    fig, ax = px(760, 330)
    for key, col, lab in (("YCRCB/uniform", GREY, "uniform bins over [0, 255]"), ("YCRCB/range", TEAL, "bins over the real range")):
        pts = D["bins_curve"][key]
        x = [p["n"] for p in pts]; y = [p["map1"] for p in pts]
        ax.plot(x, y, "-o", color=col, lw=2, ms=6, label=lab, zorder=3)
    best = max(D["bins_curve"]["YCRCB/range"], key=lambda p: p["map1"])
    ax.annotate(f"{best["map1"]:.2f}", (best["n"], best["map1"]), textcoords="offset points", xytext=(0, 8), ha="center",
                color=TEAL, fontweight="bold")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 3, 4, 6, 8, 12, 16, 24]); ax.set_xticklabels([2, 3, 4, 6, 8, 12, 16, 24])
    ax.set_xlabel("bins per channel (3D: bins³ in total)"); ax.set_ylabel("mAP@1")
    ax.grid(axis="y", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    save(fig, "t1_bins_curve")


def fig_chroma_range():
    """Where the Cr values of the BBDD really are, with uniform vs range-adapted edges (8 bins)."""
    c = dict(margin=0.05)
    vals = []
    for p in list(BB.values())[::3]:
        x = R.channel_mean_128(R.inner_crop(R.resize_max(cv2.imread(str(p))), 0.05))
        vals.append(cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)[..., 1].ravel())
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [0.5, 99.5])
    fig, ax = px(900, 250)
    h, e = np.histogram(v, 256, (0, 256))
    ax.fill_between(np.arange(256), h / h.max(), step="mid", color=TEAL_L, lw=0)
    for k in np.linspace(0, 256, 9):
        ax.axvline(k, color=GREY, lw=1.2, ls=(0, (3, 2)), ymin=0.0, ymax=0.5)
    for k in np.linspace(lo, hi, 9):
        ax.axvline(k, color=TEAL, lw=1.4, ymin=0.5, ymax=1.0)
    ax.text(2, 0.30, "uniform edges", color=MUTED, fontsize=13, va="center")
    ax.text(2, 0.80, "range edges", color=TEAL, fontsize=13, va="center", fontweight="bold")
    ax.set_xlim(0, 255); ax.set_ylim(0, 1.05); ax.set_yticks([])
    ax.set_xlabel("Cr value of the museum pixels (0–255)")
    ax.spines["left"].set_visible(False)
    save(fig, "t1_chroma_range")
    return float(lo), float(hi)


def fig_blocks_curve():
    fig, ax = px(760, 380)
    for kind, col, lab in (("1d", ORANGE, "1D per channel"), ("3d", TEAL, "3D joint")):
        pts = [p for p in D["blocks_curve"][kind] if p["n"] != "pyramid"]
        x = [p["n"] for p in pts]; y = [p["map1"] for p in pts]
        ax.plot(x, y, "-o", color=col, lw=2, ms=6, label=lab, zorder=3)
        ax.annotate(f"{y[0]:.2f}", (x[0], y[0]), textcoords="offset points", xytext=(10, -4), color=col, fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 8]); ax.set_xticklabels(["1×1", "2×2", "3×3", "4×4", "5×5", "6×6", "8×8"])
    ax.set_xlabel("blocks"); ax.set_ylabel("mAP@1")
    ax.set_ylim(0.6, 1.02); ax.grid(axis="y", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    save(fig, "t1_blocks_curve")


def palette(img_bgr_norm, mask=None, bins=(6, 12, 12)):
    """(colours, weights) of the 3D histogram of a region, each bin drawn with the mean colour of its pixels."""
    ycc = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2YCrCb)
    e = [np.linspace(0, 256, n + 1) for n in bins]
    q = [np.clip(np.searchsorted(e[k][1:-1], ycc[..., k], side="right"), 0, bins[k] - 1) for k in range(3)]
    idx = (q[0] * bins[1] * bins[2] + q[1] * bins[2] + q[2]).ravel()
    px = img_bgr_norm.reshape(-1, 3).astype(float)
    n = np.bincount(idx, minlength=np.prod(bins)).astype(float)
    cols = np.stack([np.bincount(idx, weights=px[:, c], minlength=np.prod(bins)) for c in range(3)], 1) / np.maximum(n, 1)[:, None]
    keep = n > 0.01 * n.sum()
    cols, w = cols[keep], n[keep] / n[keep].sum()
    order = np.argsort(cols @ [0.114, 0.587, 0.299])        # dark to light
    return cols[order][:, ::-1] / 255, w[order]


def draw_palette(ax, x0, y0, w, h, cols, wts):
    x = x0
    for c, wt in zip(cols, wts):
        ax.add_patch(Rectangle((x, y0), w * wt, h, color=c, lw=0)); x += w * wt
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="white", lw=1.2))


def fig_block_example(qset, qi, gt, confuser, rank_global):
    if qset == "QSD1":
        q = R.resize_max(cv2.imread(str(Q1P[qi])))
    else:
        im = cv2.imread(str(Q2P[qi])); gm = cv2.imread(str(Q2P[qi].with_suffix(".png")), 0) > 0
        q = R.crop_to_mask(im, gm)
    imgs = [("query", q), (f"correct painting", R.resize_max(cv2.imread(str(BB[gt])))),
            ("wrong painting", R.resize_max(cv2.imread(str(BB[confuser]))))]
    fig, axes = px(900, 620, nrows=3, ncols=3, gridspec_kw=dict(height_ratios=[2.2, 0.45, 2.0]))
    for j, (name, x) in enumerate(imgs):
        xn = R.channel_mean_128(R.inner_crop(x, 0.05))
        a = axes[0, j]; a.imshow(rgb(xn)); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_color({0: INK, 1: TEAL, 2: RED}[j]); s.set_linewidth(2.5)
        a.set_title(name, color={0: INK, 1: TEAL, 2: RED}[j], fontsize=15)
        a = axes[1, j]; a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        cols, w = palette(xn); draw_palette(a, 0, 0.05, 1, 0.9, cols, w)
        a = axes[2, j]; a.set_xlim(0, 2); a.set_ylim(0, 2); a.axis("off"); a.invert_yaxis()
        hh, ww = xn.shape[:2]
        for bi in range(2):
            for bj in range(2):
                blk = xn[bi * hh // 2:(bi + 1) * hh // 2, bj * ww // 2:(bj + 1) * ww // 2]
                cols, w = palette(blk); draw_palette(a, bj + 0.03, bi + 0.03, 0.94, 0.94, cols, w)
    axes[1, 0].text(-0.04, 0.5, "whole\nimage", transform=axes[1, 0].transAxes, ha="right", va="center", fontsize=13, color=MUTED)
    axes[2, 0].text(-0.04, 0.5, "2×2\nblocks", transform=axes[2, 0].transAxes, ha="right", va="center", fontsize=13, color=MUTED)
    save(fig, "t1_block_example")


def fig_ablation():
    A = D["ablation"]
    rows = [("final descriptor", "final"), ("– light normalisation", "no light normalisation"), ("– blocks (one global histogram)", "no blocks (global)"),
            ("– 3D (1D per channel)", "1D instead of 3D"), ("– range bins (uniform)", "uniform bins"),
            ("– pyramid (8×8 only)", "no pyramid (8x8 only)"), ("HSV instead of YCrCb", "HSV instead of YCrCb"),
            ("Lab instead of YCrCb", "Lab instead of YCrCb")]
    vals = [A[k]["dev60"]["map1"] for _, k in rows]
    base = vals[0]
    fig, ax = px(940, 560)
    y = np.arange(len(rows))[::-1]
    cols = [TEAL] + [RED if base - v > 0.05 else GREY for v in vals[1:]]
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(max(v, base) + 0.008, yi, f"{v:.2f}", va="center", fontsize=13, color=INK, fontweight="bold" if v < base - 0.05 or v == base else "normal")
    ax.axvline(base, color=TEAL, lw=1, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlim(0.5, 1.06); ax.set_xlabel("mAP@1 (60 queries)")
    ax.grid(axis="x", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    save(fig, "t1_ablation")


# ============================================================================= Task 3
def seg_pre(q):
    img = cv2.imread(str(Q2P[q])); gt = cv2.imread(str(Q2P[q].with_suffix(".png")), 0) > 0
    small = cv2.medianBlur(R.resize_max(img, R.WORK_SIDE), 5)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
    P = R.border_pixels(img)
    d = np.linalg.norm(P - np.median(P, 0), axis=1); Pk = P[d <= np.quantile(d, 0.8)]
    mu, cov = Pk.mean(0), np.cov(Pk.T) + np.eye(3)
    diff = lab.reshape(-1, 3) - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)).reshape(lab.shape[:2])
    gts = cv2.resize(gt.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    t_small = max(1, int(round(R.BORDER_PX * small.shape[0] / img.shape[0])))
    return img, small, md, gts, t_small


def fig_seg_steps(q=8):
    img, small, md, gts, t = seg_pre(q)
    ld = np.log1p(md); thr = R.otsu_threshold(ld)
    fig, axes = px(1728, 360, ncols=4, gridspec_kw=dict(width_ratios=[1, 1, 1.5, 1]))
    show = small.copy()
    strip = np.zeros(small.shape[:2], bool); s = max(3, t)
    strip[:s], strip[-s:], strip[:, :s], strip[:, -s:] = True, True, True, True
    show[strip] = (0.35 * show[strip] + 0.65 * np.array([0x79, 0x7F, 0x1B])).astype(np.uint8)
    axes[0].imshow(rgb(show)); axes[0].set_title("1 · 15-px border = wall")
    axes[1].imshow(ld, cmap="magma"); axes[1].set_title("2 · distance to the wall")
    a = axes[2]
    hist, e = np.histogram(ld.ravel(), 120)
    c = (e[:-1] + e[1:]) / 2
    a.bar(c, hist / hist.max(), width=e[1] - e[0], color=np.where(c > thr, TEAL, GREY), lw=0)
    a.axvline(thr, color=INK, lw=2); a.text(thr, 1.02, " Otsu", color=INK, fontsize=13, fontweight="bold", va="bottom")
    for k in (3, 8):
        a.axvline(np.log1p(k), color=ORANGE, lw=1.2, ls=(0, (3, 2)))
        a.text(np.log1p(k), 1.02, f"fixed {k}σ " if k == 3 else f" fixed {k}σ", color=ORANGE, fontsize=12,
               ha="right" if k == 3 else "left", va="bottom")
    a.text(0.02, 0.97, "wall", transform=a.transAxes, color=MUTED, fontsize=13, va="top", fontweight="bold")
    a.text(0.99, 0.97, "painting", transform=a.transAxes, color=TEAL, fontsize=13, ha="right", va="top", fontweight="bold")
    a.set_ylim(0, 1.25); a.set_yticks([]); a.spines["left"].set_visible(False); a.set_xlabel("log(1 + Mahalanobis distance)", fontsize=13)
    a.set_title("3 · cut at the valley")
    axes[3].imshow(ld > thr, cmap="gray"); axes[3].set_title("4 · raw mask")
    for a in axes[[0, 1, 3]]:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t3_steps")


def fig_thresholds():
    T = S["thresholds"]
    rows = [("Otsu on log distance  (ours)", "Otsu on log(1+d) (final)", TEAL),
            ("Otsu on raw distance", "Otsu on d (no log)", GREY),
            ("fixed: d > 3σ", "fixed d > 3", GREY), ("fixed: d > 5σ", "fixed d > 5", GREY), ("fixed: d > 8σ", "fixed d > 8", GREY),
            ("99th pct of the border", "border percentile 99", GREY), ("95th pct of the border", "border percentile 95", GREY),
            ("GMM of the border (3 Gauss.)", "M3 GMM (3 Gaussians)", GREY),
            ("Euclidean, fixed T = 25", "M1 Euclidean, T = 25", GREY)]
    vals = [T[k]["F1"] for _, k, _ in rows]; mins = [T[k]["F1_min"] for _, k, _ in rows]
    fig, ax = px(920, 360)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=[c for *_, c in rows], height=0.62, zorder=3)
    for yi, v, mn in zip(y, vals, mins):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=13, fontweight="bold" if yi == y[0] else "normal")
        ax.text(1.055, yi, f"{mn:.2f}", va="center", ha="right", fontsize=13, color=MUTED)
    ax.text(1.055, y[0] + 0.85, "worst", ha="right", fontsize=13, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=13); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.8, 1.06); ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel("mean F1 of the masks (QSD2, 30 images)")
    ax.grid(axis="x", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    save(fig, "t3_thresholds")


def post_steps(fg):
    h, w = fg.shape; s = min(h, w)
    se = lambda f: cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(f * s)) | 1,) * 2)
    out = [fg.copy()]
    m = cv2.morphologyEx(cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, se(0.03)), cv2.MORPH_OPEN, se(0.02)); out.append(m > 0)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (lbl == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    out.append(m > 0)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2); m = pad[1:-1, 1:-1] != 2; out.append(m)
    rows, cols = np.zeros_like(m), np.zeros_like(m)
    for y in np.where(m.any(1))[0]:
        xs = np.where(m[y])[0]; rows[y, xs[0]:xs[-1] + 1] = True
    for x in np.where(m.any(0))[0]:
        ys = np.where(m[:, x])[0]; cols[ys[0]:ys[-1] + 1, x] = True
    out.append(rows & cols)
    return out


def overlay(small, m, gt):
    o = (small.astype(float) * 0.55).astype(np.uint8)
    o[m & gt] = (0.45 * small[m & gt] + 0.55 * np.array([0x79, 0x7F, 0x1B])).astype(np.uint8)
    o[m & ~gt] = (0x2B, 0x39, 0xC0); o[~m & gt] = (0xBF, 0x6F, 0x2E)
    return o


def fig_post_steps(q=29):
    img, small, md, gts, _ = seg_pre(q)
    ld = np.log1p(md); fg = ld > R.otsu_threshold(ld)
    st = post_steps(fg)
    names = ["raw threshold", "closing + opening", "largest component", "fill holes", "row / column fill"]
    fig, axes = px(1728, 300, ncols=6)
    axes[0].imshow(rgb(small)); axes[0].set_title(f"QSD2 query {q}", fontsize=13)
    for a, m, nm in zip(axes[1:], st, names):
        p, r, f = R.prf(m, gts)
        a.imshow(m, cmap="gray", vmin=0, vmax=1); a.set_title(nm, fontsize=13)
        a.text(0.5, -0.09, f"F1 {f:.2f}", transform=a.transAxes, ha="center", va="top", fontsize=13,
               color=TEAL if f > 0.9 else (RED if f < 0.5 else INK), fontweight="bold")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t3_post_steps")


def fig_limits():
    fig, axes = px(860, 320, ncols=2)
    for a, q, title in ((axes[0], 14, "cast shadow (q14)"), (axes[1], 20, "white on white (q20)")):
        img, small, md, gts, _ = seg_pre(q)
        ld = np.log1p(md); m = post_steps(ld > R.otsu_threshold(ld))[-1]
        a.imshow(rgb(overlay(small, m, gts))); a.set_xticks([]); a.set_yticks([])
        p, r, f = R.prf(m, gts)
        a.set_title(f"{title} · F1 {f:.2f}", fontsize=13)
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t3_limits")


# ============================================================================= Task 5
def fig_t5_bars():
    T5 = D["t5"]
    rows = [("8×8\nblocks", "8x8, no margin"), ("8×8\n+ margin", "8x8 + 5% margin"),
            ("pyramid", "pyramid, no margin"), ("pyramid\n+ margin", "pyramid + 5% margin (final)")]
    fig, ax = px(764, 420)
    x = np.arange(len(rows)); w = 0.36
    a = [T5[k]["res"]["QSD2-pred"]["map1"] for _, k in rows]; b = [T5[k]["res"]["QSD2-15px"]["map1"] for _, k in rows]
    ax.bar(x - w / 2, a, w, color=TEAL, label="our masks", zorder=3)
    ax.bar(x + w / 2, b, w, color=ORANGE, label="worst case: 15 px of wall", zorder=3)
    for xi, v in zip(x - w / 2, a):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=13)
    for xi, v in zip(x + w / 2, b):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=13)
    gt = T5["pyramid + 5% margin (final)"]["res"]["QSD2-GT"]["map1"]
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=13)
    ax.set_ylim(0.8, 1.04); ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0]); ax.set_ylabel("mAP@1 on QSD2")
    ax.grid(axis="y", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    fig.legend(frameon=False, loc="outside upper center", fontsize=13, ncol=1)
    save(fig, "t5_bars")


def fig_t5_mechanism(q):
    im = cv2.imread(str(Q2P[q])); gm = cv2.imread(str(Q2P[q].with_suffix(".png")), 0) > 0
    pm = R.compute_mask(im)
    x, y, w, h = cv2.boundingRect(pm.astype(np.uint8))
    crop, gcrop = im[y:y + h, x:x + w], gm[y:y + h, x:x + w]
    show = crop.copy()
    show[~gcrop] = (0.4 * show[~gcrop] + 0.6 * np.array([0x2B, 0x39, 0xC0])).astype(np.uint8)
    xg, yg, wg, hg = cv2.boundingRect(gm.astype(np.uint8))
    fig, axes = px(900, 300, ncols=3)
    axes[0].imshow(rgb(R.resize_max(show))); axes[0].set_title("our crop (red: wall)", fontsize=13)
    axes[1].imshow(rgb(R.channel_mean_128(R.resize_max(crop)))); axes[1].set_title("normalised, our crop", fontsize=13)
    axes[2].imshow(rgb(R.channel_mean_128(R.resize_max(im[yg:yg + hg, xg:xg + wg])))); axes[2].set_title("normalised, GT crop", fontsize=13)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t5_mechanism")


def fig_t4_per_image():
    f = np.array(S["final"]["per_image"])
    fig, ax = px(1728, 300)
    x = np.arange(len(f))
    ax.bar(x, f, color=[TEAL if v >= 0.9 else ORANGE for v in f], width=0.7, zorder=3)
    ax.axhline(f.mean(), color=INK, lw=1.2, ls=(0, (3, 2)))
    ax.text(-0.45, f.mean() + 0.008, f"mean {f.mean():.3f}", ha="left", va="bottom", fontsize=13, fontweight="bold")
    for i in np.where(f < 0.9)[0]:
        ax.text(i, f[i] + 0.01, f"{f[i]:.2f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=13)
    ax.set_xlabel("QSD2 query"); ax.set_ylabel("F1"); ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    save(fig, "t4_per_image")

# ============================================================================= v2 figures (slide redesign)
BLUE_L = "#4A6FA5"


def fig_t1_kinds():
    """The same painting described as 1D, 2D and 3D histograms (what Task 1 compares)."""
    gt2 = pickle.load(open(q2 / "gt_corresps.pkl", "rb"))
    x = R.channel_mean_128(R.inner_crop(R.resize_max(cv2.imread(str(BB[gt2[13][0]]))), 0.05))
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    lo = [np.percentile(ycc[..., k], 0.5) for k in range(3)]; hi = [np.percentile(ycc[..., k], 99.5) for k in range(3)]
    q = lambda k, n: np.clip(((ycc[..., k].astype(float) - lo[k]) / (hi[k] - lo[k] + 1e-9) * n).astype(int), 0, n - 1)
    fig, axes = px(1728, 260, ncols=4, gridspec_kw=dict(width_ratios=[0.75, 1.5, 0.9, 1.6]))
    axes[0].imshow(rgb(x)); axes[0].set_title("one museum painting")
    a = axes[1]
    for k, (name, col) in enumerate((("Y", GREY), ("Cr", "#C0392B"), ("Cb", BLUE_L))):
        h = np.bincount(q(k, 16).ravel(), minlength=16) / q(k, 16).size
        a.bar(np.arange(16) + k * 18, h, color=col, width=0.85)
        a.text(k * 18 + 7.5, -0.05, name, ha="center", va="top", fontsize=13, color=INK, transform=a.get_xaxis_transform())
    a.set_xticks([]); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_title("1D · each channel apart (3 × 16)")
    a = axes[2]
    h2 = np.zeros((12, 12)); np.add.at(h2, (q(1, 12).ravel(), q(2, 12).ravel()), 1)
    a.imshow(h2.T, origin="lower", cmap="Greens", interpolation="nearest")
    a.set_xlabel("Cr", fontsize=13); a.set_ylabel("Cb", fontsize=13); a.set_xticks([]); a.set_yticks([])
    a.set_title("2D · chroma only (12 × 12)")
    a = axes[3]
    b = (6, 12, 12); idx = (q(0, 6) * 144 + q(1, 12) * 12 + q(2, 12)).ravel()
    n = np.bincount(idx, minlength=864).astype(float)
    pxl = x.reshape(-1, 3).astype(float)
    cols = np.stack([np.bincount(idx, weights=pxl[:, c], minlength=864) for c in range(3)], 1) / np.maximum(n, 1)[:, None]
    top = np.argsort(n)[::-1][:40]
    a.bar(np.arange(40), n[top] / n.sum(), color=cols[top][:, ::-1] / 255, width=0.85)
    a.set_xticks([]); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_title("3D · Y × Cr × Cb (6 × 12 × 12): one bar per colour")
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    save(fig, "t1_kinds")


def fig_t1_types():
    T = D["types"]
    kinds = [("1d", "1D"), ("2d", "2D chroma"), ("2d1d", "2D + 1D"), ("3d", "3D")]
    fig, ax = px(860, 310)
    ax.axvspan(2.6, 3.4, color=TEAL_L, alpha=0.35, lw=0)
    for sp, lab, col, lw in (("YCRCB", "YCrCb", TEAL, 3), ("HSV", "HSV", ORANGE, 2), ("LAB", "CIELab", BLUE_L, 2), ("RGB", "RGB", GREY, 2)):
        xs = [i for i, (k, _) in enumerate(kinds) if f"{sp}/{k}" in T]
        ys = [T[f"{sp}/{kinds[i][0]}"]["map1"] for i in xs]
        ax.plot(xs, ys, "-o" if sp != "RGB" else "o", color=col, lw=lw, ms=7 if sp != "RGB" else 9, label=lab if sp != "RGB" else "RGB (1D, 3D only)", zorder=3)
    ax.text(3, 0.915, "0.90 in\nall four", ha="center", va="bottom", fontsize=13, color=TEAL, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([k[1] for k in kinds])
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("mAP@1")
    ax.grid(axis="y", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=13)
    save(fig, "t1_types")


def fig_t1_delta():
    A = D["ablation"]; base = A["final"]["dev60"]["map1"]
    rows = [("light normalisation", "no light normalisation"), ("blocks (vs one global)", "no blocks (global)"),
            ("bins over the real range", "uniform bins"), ("3D (vs 1D per channel)", "1D instead of 3D"),
            ("pyramid (vs 8×8 only)", "no pyramid (8x8 only)"), ("YCrCb (vs HSV)", "HSV instead of YCrCb"),
            ("YCrCb (vs Lab)", "Lab instead of YCrCb")]
    lost = [max(0.0, (base - A[k]["dev60"]["map1"]) * 100) for _, k in rows]
    fig, ax = px(900, 420)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, lost, color=[RED if v > 5 else GREY for v in lost], height=0.6, zorder=3)
    for yi, v in zip(y, lost):
        ax.text(v + 0.6, yi, f"−{v:.0f}" if v >= 0.5 else "0", va="center", fontsize=14,
                fontweight="bold" if v > 5 else "normal", color=RED if v > 5 else MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 38); ax.set_xlabel("mAP@1 points lost when the piece is removed")
    ax.grid(axis="x", color=GREY_L, lw=0.8); ax.set_axisbelow(True)
    save(fig, "t1_delta")


def fig_t3_strip(q=6):
    """Corner of a photo cut so that only 15 px of wall remain: a 4% strip reaches the painting, the 15-px strip does not."""
    img = cv2.imread(str(Q2P[q])); gt = cv2.imread(str(Q2P[q].with_suffix(".png")), 0) > 0
    x, y, w, h = cv2.boundingRect(gt.astype(np.uint8)); pad = 15
    im, g = img[y - pad:y + h + pad, x - pad:x + w + pad], gt[y - pad:y + h + pad, x - pad:x + w + pad]
    H, W = im.shape[:2]; sc = min(1, R.WORK_SIDE / max(H, W))
    t4 = int(round(max(2, round(0.04 * min(int(H * sc), int(W * sc)))) / sc))
    c = 4 * t4                                                   # zoom on the top-left corner
    fig, axes = px(752, 250, ncols=2)
    for a, t, title in ((axes[0], t4, f"4% of the image = {t4} px"), (axes[1], 15, "15 px = always wall")):
        strip = np.zeros((H, W), bool); strip[:t], strip[-t:], strip[:, :t], strip[:, -t:] = True, True, True, True
        o = im.astype(float) * 0.7
        o[strip & ~g] = 0.35 * o[strip & ~g] + 0.65 * np.array([0x79, 0x7F, 0x1B])
        o[strip & g] = 0.35 * o[strip & g] + 0.65 * np.array([0x2B, 0x39, 0xC0])
        a.imshow(rgb(o[:c, :c].astype(np.uint8)), interpolation="nearest"); a.set_title(title, color=RED if t != 15 else TEAL)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t3_strip")


def fig_t5_grid(q=14):
    """What the descriptor sees for q14: our crop, the same after the 5% margin, and the GT crop; 8x8 grid on top."""
    im = cv2.imread(str(Q2P[q])); gm = cv2.imread(str(Q2P[q].with_suffix(".png")), 0) > 0
    pm = R.compute_mask(im)
    x, y, w, h = cv2.boundingRect(pm.astype(np.uint8)); xg, yg, wg, hg = cv2.boundingRect(gm.astype(np.uint8))
    ours, gtc = im[y:y + h, x:x + w], im[yg:yg + hg, xg:xg + wg]
    panels = [("our crop", R.channel_mean_128(ours)), ("our crop − 5% margin", R.channel_mean_128(R.inner_crop(ours, 0.05))),
              ("GT crop − 5% (≈ museum)", R.channel_mean_128(R.inner_crop(gtc, 0.05)))]
    fig, axes = px(900, 330, ncols=3)
    for a, (title, z) in zip(axes, panels):
        z = cv2.resize(z, (400, int(400 * z.shape[0] / z.shape[1])))
        a.imshow(rgb(z)); a.set_title(title)
        for k in range(1, 8):
            a.axhline(k * z.shape[0] / 8, color="white", lw=0.8, alpha=0.8); a.axvline(k * z.shape[1] / 8, color="white", lw=0.8, alpha=0.8)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    save(fig, "t5_grid")


if __name__ == "__main__":
    lo, hi = fig_chroma_range()
    print("Cr range", lo, hi)
    fig_bins_curve(); fig_blocks_curve(); fig_ablation()
    ex = D["block_examples"]
    pick = next(e for e in ex if e["set"] == "QSD2-GT" and e["q"] == 11)     # gold frame vs gold jacket: same palette, other layout
    print("block example:", pick)
    fig_block_example(pick["set"], pick["q"], pick["gt"], pick["confuser"], pick["rank_global"])
    fig_seg_steps(); fig_thresholds(); fig_post_steps(); fig_limits()
    fig_t5_bars()
    T5 = D["t5"]
    lost = [i for i, (a, b) in enumerate(zip(T5["pyramid, no margin"]["ranks"]["QSD2-pred"], T5["pyramid + 5% margin (final)"]["ranks"]["QSD2-pred"])) if a > 1 and b == 1]
    print("lost without margin, recovered with it:", lost)
    mq = max(lost, key=lambda i: D["t5_mech"][i]["wall_frac"])
    fig_t5_mechanism(mq)
    fig_t4_per_image()
    m = D["t5_mech"][mq]
    json.dump(dict(example=pick, mech=m, lost_without_margin=lost), open(FIG / "choices.json", "w"), indent=1)
    print("mechanism:", m)
    fig_t1_kinds(); fig_t1_types(); fig_t1_delta(); fig_t3_strip(); fig_t5_grid()
