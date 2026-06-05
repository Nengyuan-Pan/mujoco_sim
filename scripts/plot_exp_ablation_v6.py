"""v6 消融实验图表生成：Softmin × 随挥 4 条件对比。"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATE = "20260603"
DATA_DIR = Path(__file__).resolve().parent.parent / "results" / f"exp_ablation_v6_{DATE}"
OUT_DIR = DATA_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

with open(DATA_DIR / "raw_data.json", "r", encoding="utf-8") as f:
    all_results = json.load(f)

CONDITIONS = ["full_v6", "no_softmin", "no_follow_thru", "baseline"]
LABELS = {
    "full_v6": "Full v6\n(softmin+follow)",
    "no_softmin": "No Softmin\n(follow only)",
    "no_follow_thru": "No Follow-through\n(softmin only)",
    "baseline": "Baseline\n(neither)",
}
COLORS = {
    "full_v6": "#2196F3",
    "no_softmin": "#FF9800",
    "no_follow_thru": "#4CAF50",
    "baseline": "#9C27B0",
}


def get_stats(name):
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs]
    face_speeds = [r["result"].get("max_face", 0) for r in runs]
    return {
        "rate": rate,
        "n_hits": len(hits),
        "n_total": len(runs),
        "pos_mean": np.mean(pos_errs),
        "pos_std": np.std(pos_errs),
        "face_mean": np.mean(face_speeds),
        "face_std": np.std(face_speeds),
    }


# ── 图1: 成功率柱状图 ──
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(CONDITIONS))
stats = [get_stats(c) for c in CONDITIONS]
rates = [s["rate"] for s in stats]
bars = ax.bar(x, rates, color=[COLORS[c] for c in CONDITIONS], width=0.6, edgecolor="black", linewidth=0.5)
for bar, s in zip(bars, stats):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{s['rate']:.0f}%\n({s['n_hits']}/{s['n_total']})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9)
ax.set_ylabel("Success Rate (%)")
ax.set_title("v6 Ablation: Softmin × Follow-through")
ax.set_ylim(0, 105)
ax.axhline(y=85, color="gray", linestyle="--", alpha=0.5, label="Baseline ref")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_v6_bar.pdf")
fig.savefig(OUT_DIR / "ablation_v6_bar.png")
plt.close(fig)

# ── 图2: 位置误差箱线图 ──
fig, ax = plt.subplots(figsize=(7, 4.5))
box_data = []
for c in CONDITIONS:
    runs = [r for r in all_results[c] if r["result"] is not None]
    box_data.append([r["result"]["pos_error"] * 100 for r in runs])
bp = ax.boxplot(box_data, labels=[LABELS[c].replace("\n", " ") for c in CONDITIONS],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], CONDITIONS):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.7)
ax.set_ylabel("Position Error (cm)")
ax.set_title("v6 Ablation: Position Error Distribution")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_v6_boxplot.pdf")
fig.savefig(OUT_DIR / "ablation_v6_boxplot.png")
plt.close(fig)

# ── 打印汇总 ──
print("=== v6 消融实验 汇总 ===")
print(f"{'条件':>20s} {'成功率':>10s} {'位置误差(cm)':>14s} {'Face速度(m/s)':>14s}")
print("-" * 62)
for c in CONDITIONS:
    s = get_stats(c)
    print(f"{LABELS[c].replace(chr(10), ' '):>20s} "
          f"{s['n_hits']:>2d}/{s['n_total']:>2d} ({s['rate']:>3.0f}%) "
          f"{s['pos_mean']:>6.1f}±{s['pos_std']:.1f} "
          f"{s['face_mean']:>6.1f}±{s['face_std']:.1f}")
print(f"\n图表已保存到 {OUT_DIR}")
