"""Genera los plots del reporte best_methods_w1.md."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
HERE      = Path(__file__).resolve().parent
ROOT      = HERE.parent.parent.parent  # workspace root
BBDD_DIR  = ROOT / "Project/DATA/BBDD"
QSD1_DIR  = ROOT / "Project/DATA/QUERY/qsd1_w1"
GT_FILE   = QSD1_DIR / "gt_corresps.pkl"
OUT_DIR   = HERE / "img" / "w1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_DIM   = 512
EPS       = 1e-10
DEMO_Q    = 0  # query index used for pipeline visualisation


def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    s = max_dim / m
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------- Color-space converters ----------
def cs_lab(img):  return cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
def cs_hsv(img):  return cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
def cs_hmmd(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].astype(np.float32)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    return np.stack([h, mx, mn, mx - mn], axis=-1)


# ---------- Metrics ----------
def dist_l1(q, D):    return np.abs(q - D).sum(axis=1)
def dist_chi2(q, D):  return ((q - D) ** 2 / (q + D + EPS)).sum(axis=1)
def dist_canb(q, D):  return (np.abs(q - D) / (np.abs(q) + np.abs(D) + EPS)).sum(axis=1)


# ---------- Descriptor / evaluation helpers ----------
def global_ranges(imgs):
    n_ch = imgs[0].shape[-1]
    out = []
    for c in range(n_ch):
        vals = np.concatenate([x[..., c].ravel() for x in imgs])
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            hi = lo + 1.0
        out.append((lo, hi + 1e-6))
    return out


def per_channel_hist(img_conv, nbins, ranges):
    n_ch = img_conv.shape[-1]
    return [
        np.histogram(img_conv[..., c].ravel(), bins=nbins, range=ranges[c])[0].astype(np.float64)
        for c in range(n_ch)
    ]


def descriptors(imgs_conv, nbins, ranges):
    n_ch = imgs_conv[0].shape[-1]
    D = np.zeros((len(imgs_conv), n_ch * nbins), dtype=np.float64)
    for i, x in enumerate(imgs_conv):
        D[i] = np.concatenate(per_channel_hist(x, nbins, ranges))
    return D


def ap_at_k(order, relevant, k):
    hits, s = 0, 0.0
    for i, idx in enumerate(order[:k], start=1):
        if idx in relevant:
            hits += 1
            s += hits / i
    return s / (min(len(relevant), k) if relevant else 1)


def map_at_k(orders, gt, k):
    return float(np.mean([
        ap_at_k(o, set(g) if isinstance(g, (list, tuple)) else {g}, k)
        for o, g in zip(orders, gt)
    ]))


# ---------- Methods ----------
METHODS = [
    dict(id="v1_top1", label="V1 · Top 1",
         title="CIELab + $L_1$ + 32 bins + norm",
         color="tab:blue",
         converter=cs_lab, ch_names=("L*", "a*", "b*"),
         nbins=32, metric_name="L1", metric_fn=dist_l1, direction="distance"),
    dict(id="v1_top2", label="V1 · Top 2",
         title="HSV + $\\chi^{2}$ + 16 bins + norm",
         color="tab:orange",
         converter=cs_hsv, ch_names=("H", "S", "V"),
         nbins=16, metric_name="chi2", metric_fn=dist_chi2, direction="distance"),
    dict(id="v2_top1", label="V2 · Top 1",
         title="HMMD + Canberra + 16 bins + norm",
         color="tab:green",
         converter=cs_hmmd, ch_names=("H", "Max", "Min", "Diff"),
         nbins=16, metric_name="canberra", metric_fn=dist_canb, direction="distance"),
    dict(id="v2_top2", label="V2 · Top 2",
         title="HSV + Canberra + 16 bins + norm",
         color="tab:red",
         converter=cs_hsv, ch_names=("H", "S", "V"),
         nbins=16, metric_name="canberra", metric_fn=dist_canb, direction="distance"),
]


# ---------- Load data once ----------
print("Loading data...")
bbdd_paths = sorted(BBDD_DIR.glob("bbdd_*.jpg"))
qsd_paths  = sorted(QSD1_DIR.glob("*.jpg"))
gt         = pickle.load(open(GT_FILE, "rb"))
bbdd = [resize_max(cv2.imread(str(p))) for p in bbdd_paths]
qsd  = [resize_max(cv2.imread(str(p))) for p in qsd_paths]
print(f"  BBDD={len(bbdd)}  QSD1={len(qsd)}")


# ---------- Compute results for each method ----------
for m in METHODS:
    print(f"\n>>> {m['id']}  {m['title']}")
    conv_b = [m["converter"](x) for x in bbdd]
    conv_q = [m["converter"](x) for x in qsd]
    ranges = global_ranges(conv_b + conv_q)
    n_ch = conv_b[0].shape[-1]
    Db_raw = descriptors(conv_b, m["nbins"], ranges)
    Dq_raw = descriptors(conv_q, m["nbins"], ranges)
    Db = Db_raw / (Db_raw.sum(axis=1, keepdims=True) + EPS)
    Dq = Dq_raw / (Dq_raw.sum(axis=1, keepdims=True) + EPS)

    # Rankings on ALL queries -> mAP@k
    orders = []
    for q in Dq:
        sc = m["metric_fn"](q, Db)
        order = np.argsort(sc) if m["direction"] == "distance" else np.argsort(-sc)
        orders.append(order[:5].tolist())
    m["mAP@1"] = map_at_k(orders, gt, 1)
    m["mAP@5"] = map_at_k(orders, gt, 5)
    print(f"  mAP@1={m['mAP@1']:.3f}  mAP@5={m['mAP@5']:.3f}")

    # Cache for plotting
    m["_qidx"]     = DEMO_Q
    m["_qconv"]    = conv_q[DEMO_Q]
    m["_qhists"]   = per_channel_hist(conv_q[DEMO_Q], m["nbins"], ranges)
    m["_qconcat"]  = Dq_raw[DEMO_Q]
    m["_qnorm"]    = Dq[DEMO_Q]
    m["_top5"]     = orders[DEMO_Q]
    m["_gt"]       = gt[DEMO_Q]
    m["_ranges"]   = ranges
    m["_n_ch"]     = n_ch


# ---------- Plot 1: pipeline figure per method ----------
def plot_pipeline(m, qsd_bgr, bbdd_bgr, out_path):
    q_rgb = cv2.cvtColor(qsd_bgr[m["_qidx"]], cv2.COLOR_BGR2RGB)
    conv  = m["_qconv"]
    hists = m["_qhists"]
    concat_raw  = m["_qconcat"]
    concat_norm = m["_qnorm"]
    n_ch = m["_n_ch"]

    ch_colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    nbins = m["nbins"]

    fig = plt.figure(figsize=(4 * n_ch + 2, 15))
    gs = gridspec.GridSpec(5, n_ch, height_ratios=[2, 1.5, 1.2, 1.0, 1.0], hspace=0.55, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(q_rgb)
    ax0.set_title(f"Query original — {m['label']}: {m['title']}\n(qsd1_w1/{m['_qidx']:05d}.jpg)", fontsize=13)
    ax0.axis("off")

    for c in range(n_ch):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(conv[..., c], cmap="gray")
        ax.set_title(f"Canal {m['ch_names'][c]}", fontsize=11)
        ax.axis("off")

    for c in range(n_ch):
        ax = fig.add_subplot(gs[2, c])
        ax.bar(np.arange(nbins), hists[c], color=ch_colors[c], edgecolor="none")
        ax.set_title(f"Hist. {m['ch_names'][c]} ({nbins} bins)", fontsize=11)
        ax.set_xlabel("bin"); ax.set_ylabel("cuentas")
        ax.grid(alpha=0.3)

    ax_c = fig.add_subplot(gs[3, :])
    xs = np.arange(len(concat_raw))
    colors_concat = np.repeat(ch_colors[:n_ch], nbins)
    ax_c.bar(xs, concat_raw, color=colors_concat, edgecolor="none")
    for c in range(1, n_ch):
        ax_c.axvline(c * nbins - 0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_c.set_title(f"Descriptor concatenado (dim = {n_ch}·{nbins} = {len(concat_raw)}), sin normalizar", fontsize=11)
    ax_c.set_xlabel("bin (concatenado)"); ax_c.set_ylabel("cuentas")
    ax_c.grid(alpha=0.3)

    ax_n = fig.add_subplot(gs[4, :])
    ax_n.bar(xs, concat_norm, color=colors_concat, edgecolor="none")
    for c in range(1, n_ch):
        ax_n.axvline(c * nbins - 0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_n.set_title(f"Descriptor normalizado a suma 1 (probabilidad). $\\sum h = {concat_norm.sum():.3f}$", fontsize=11)
    ax_n.set_xlabel("bin (concatenado)"); ax_n.set_ylabel("prob.")
    ax_n.grid(alpha=0.3)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 2: retrieval strip per method ----------
def plot_retrieval(m, qsd_bgr, bbdd_bgr, out_path):
    q_rgb = cv2.cvtColor(qsd_bgr[m["_qidx"]], cv2.COLOR_BGR2RGB)
    top5  = m["_top5"]
    rel   = set(m["_gt"]) if isinstance(m["_gt"], (list, tuple)) else {m["_gt"]}

    fig, axes = plt.subplots(1, 6, figsize=(18, 4.2))
    axes[0].imshow(q_rgb)
    axes[0].set_title(f"Query {m['_qidx']:05d}\n(GT={sorted(rel)})", fontsize=11)
    axes[0].axis("off")
    for k in range(5):
        idx = top5[k]
        img = cv2.cvtColor(bbdd_bgr[idx], cv2.COLOR_BGR2RGB)
        axes[k + 1].imshow(img)
        border = "tab:green" if idx in rel else "tab:red"
        symbol = "✓" if idx in rel else "✗"
        axes[k + 1].set_title(f"Rank {k + 1}: bbdd_{idx:05d} {symbol}", fontsize=11, color=border)
        for spine in axes[k + 1].spines.values():
            spine.set_edgecolor(border); spine.set_linewidth(3)
        axes[k + 1].set_xticks([]); axes[k + 1].set_yticks([])
    fig.suptitle(f"{m['label']}: {m['title']} — Top-5 sobre BBDD", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- Plot 3: mAP comparison ----------
def plot_map_comparison(methods, out_path):
    labels = [f"{m['label']}\n{m['title']}" for m in methods]
    m1 = [m["mAP@1"] for m in methods]
    m5 = [m["mAP@5"] for m in methods]
    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))
    b1 = ax.bar(x - w / 2, m1, w, label="mAP@1", color="tab:blue")
    b5 = ax.bar(x + w / 2, m5, w, label="mAP@5", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean Average Precision")
    ax.set_title("mAP@1 y mAP@5 en QSD1_w1 (30 queries vs 287 BBDD)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in list(zip(b1, m1)) + list(zip(b5, m5)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------- Generate ----------
print("\nGenerating figures...")
for m in METHODS:
    plot_pipeline(m, qsd, bbdd, OUT_DIR / f"{m['id']}_pipeline.png")
    plot_retrieval(m, qsd, bbdd, OUT_DIR / f"{m['id']}_retrieval.png")
    print(f"  {m['id']}: pipeline + retrieval saved")
plot_map_comparison(METHODS, OUT_DIR / "map_comparison.png")
print(f"  map_comparison.png saved")
print(f"\nAll figures in {OUT_DIR}")
