"""v6 球速实验图表生成：5-11 m/s 成功率与位置误差。"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATE = "20260603"
DATA_DIR = Path(__file__).resolve().parent.parent / "results" / f"exp_speed_v6_{DATE}"
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

SPEEDS = [5, 6, 7, 8, 9, 10, 11]


def get_speed_stats(speed):
    name = f"speed_{speed}"
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


stats_list = [get_speed_stats(s) for s in SPEEDS]

# ── 图1: 成功率 vs 球速 ──
fig, ax1 = plt.subplots(figsize=(8, 5))
rates = [s["rate"] for s in stats_list]
ax1.plot(SPEEDS, rates, "o-", color="#2196F3", linewidth=2, markersize=8, label="Success Rate")
for i, (sp, r) in enumerate(zip(SPEEDS, rates)):
    ax1.annotate(f"{r:.0f}%", (sp, r), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
ax1.set_xlabel("Ball Speed (m/s)")
ax1.set_ylabel("Success Rate (%)", color="#2196F3")
ax1.set_ylim(50, 100)
ax1.tick_params(axis="y", labelcolor="#2196F3")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
pos_errs = [s["pos_mean"] for s in stats_list]
ax2.plot(SPEEDS, pos_errs, "s--", color="#FF5722", linewidth=1.5, markersize=7, label="Pos Error")
ax2.set_ylabel("Position Error (cm)", color="#FF5722")
ax2.tick_params(axis="y", labelcolor="#FF5722")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
ax1.set_title("v6 Ball Speed Experiment (No Softmin, No Follow-through)")
fig.tight_layout()
fig.savefig(OUT_DIR / "speed_v6_rate.pdf")
fig.savefig(OUT_DIR / "speed_v6_rate.png")
plt.close(fig)

# ── 图2: 位置误差箱线图 ──
fig, ax = plt.subplots(figsize=(8, 5))
box_data = []
for sp in SPEEDS:
    name = f"speed_{sp}"
    runs = [r for r in all_results[name] if r["result"] is not None]
    box_data.append([r["result"]["pos_error"] * 100 for r in runs])
bp = ax.boxplot(box_data, labels=[f"{s} m/s" for s in SPEEDS], patch_artist=True, widths=0.5)
cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(SPEEDS)))
for patch, c in zip(bp["boxes"], cmap):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(y=12, color="red", linestyle="--", alpha=0.5, label="12cm threshold")
ax.set_ylabel("Position Error (cm)")
ax.set_xlabel("Ball Speed (m/s)")
ax.set_title("v6 Position Error vs Ball Speed")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "speed_v6_boxplot.pdf")
fig.savefig(OUT_DIR / "speed_v6_boxplot.png")
plt.close(fig)

# ── 打印汇总 ──
print("=== v6 球速实验 汇总 ===")
print(f"{'球速(m/s)':>10s} {'成功率':>10s} {'位置误差(cm)':>14s} {'Face速度(m/s)':>14s}")
print("-" * 52)
for sp, s in zip(SPEEDS, stats_list):
    print(f"{sp:>10.0f} {s['n_hits']:>2d}/{s['n_total']:>2d} ({s['rate']:>3.0f}%) "
          f"{s['pos_mean']:>6.1f}±{s['pos_std']:.1f} "
          f"{s['face_mean']:>6.1f}±{s['face_std']:.1f}")
print(f"\n图表已保存到 {OUT_DIR}")
