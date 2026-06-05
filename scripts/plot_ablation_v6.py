"""v6 消融实验对比图生成。

读取 results/exp_ablation_v6_20260604/raw_data.json，
生成命中率、位置误差、球拍速度对比图。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "exp_ablation_v6_20260604"
OUT_DIR = DATA_DIR

HIT_THRESHOLD = 0.12  # 12cm

with open(DATA_DIR / "raw_data.json", "r", encoding="utf-8") as f:
    all_results = json.load(f)

CONDITIONS = [
    {"name": "full_v6",        "label": "Full v6\n(Softmin + Follow-through)", "color": "#2196F3"},
    {"name": "no_softmin",     "label": "No Softmin\n(+ Follow-through)",      "color": "#FF9800"},
    {"name": "no_follow_thru", "label": "No Follow-through\n(+ Softmin)",      "color": "#4CAF50"},
    {"name": "baseline",       "label": "Baseline\n(Neither)",                 "color": "#F44336"},
]

# 提取每组的统计数据
stats = {}
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    n_valid = len(runs)
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    active_hits = [r for r in runs if r["result"].get("hit_type") == "active"]

    pos_errs_all = [r["result"]["pos_error"] * 100 for r in runs]
    pos_errs_hit = [r["result"]["pos_error"] * 100 for r in hits]
    v_rackets = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    v_rackets_active = [r["result"].get("v_racket_at_hit", 0) for r in active_hits]

    stats[name] = {
        "hit_rate": len(hits) / n_valid * 100,
        "active_rate": len(active_hits) / n_valid * 100,
        "n_valid": n_valid,
        "n_hit": len(hits),
        "pos_err_all": pos_errs_all,
        "pos_err_hit": pos_errs_hit,
        "v_racket": v_rackets,
        "v_racket_active": v_rackets_active,
        "hit_time_errors": [r["result"].get("hit_time_error_ms", 0) for r in hits],
    }

# 汇总表格打印
print(f"\n{'条件':>16s}  {'命中率':>8s}  {'主动率':>8s}  {'位置误差(cm)':>14s}  {'v_racket(m/s)':>14s}")
print("-" * 70)
for cond in CONDITIONS:
    s = stats[cond["name"]]
    pe = s["pos_err_hit"]
    vr = s["v_racket"]
    print(f"{cond['name']:>16s}  {s['n_hit']}/{s['n_valid']} ({s['hit_rate']:>4.0f}%)  "
          f"{len(s['v_racket_active'])}/{s['n_valid']} ({s['active_rate']:>4.0f}%)  "
          f"{np.mean(pe):>6.1f}±{np.std(pe):.1f}      "
          f"{np.mean(vr):>6.2f}±{np.std(vr):.2f}")


# ============ 绘图 ============
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

names = [c["name"] for c in CONDITIONS]
labels = [c["label"] for c in CONDITIONS]
colors = [c["color"] for c in CONDITIONS]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# --- 图1: 命中率 ---
ax1 = axes[0]
rates = [stats[n]["hit_rate"] for n in names]
bars1 = ax1.bar(range(4), rates, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
ax1.set_xticks(range(4))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Hit Rate (%)")
ax1.set_title("(a) Hit Rate")
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.4)
for bar, rate in zip(bars1, rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{rate:.0f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

# --- 图2: 位置误差箱线图 ---
ax2 = axes[1]
bp_data = [stats[n]["pos_err_all"] for n in names]
bp = ax2.boxplot(bp_data, patch_artist=True, widths=0.5,
                  medianprops=dict(color='black', linewidth=1.5),
                  whiskerprops=dict(linewidth=1.2),
                  capprops=dict(linewidth=1.2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Position Error (cm)")
ax2.set_title("(b) Position Error")
ax2.axhline(y=HIT_THRESHOLD * 100, color='red', linestyle='--', alpha=0.6, label=f"Hit threshold ({HIT_THRESHOLD*100:.0f}cm)")
ax2.legend(fontsize=9)

# --- 图3: 球拍速度箱线图（仅命中样本）---
ax3 = axes[2]
vr_data = [stats[n]["v_racket"] for n in names]
bp2 = ax3.boxplot(vr_data, patch_artist=True, widths=0.5,
                   medianprops=dict(color='black', linewidth=1.5),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2))
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("Racket Speed at Hit (m/s)")
ax3.set_title("(c) Racket Speed (hit samples only)")
ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label="Active threshold (0.3 m/s)")
ax3.legend(fontsize=9)

plt.tight_layout()
fig_path = OUT_DIR / "ablation_v6_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存到 {fig_path}")

# ============ 额外：散点图（位置误差 vs 球拍速度）============
fig2, ax = plt.subplots(figsize=(8, 6))
for cond in CONDITIONS:
    s = stats[cond["name"]]
    hits_data = [(r["result"]["pos_error"] * 100, r["result"].get("v_racket_at_hit", 0))
                 for r in all_results[cond["name"]]
                 if r["result"] is not None and r["result"].get("hit_type") in ("active", "passive")]
    if hits_data:
        pe_arr, vr_arr = zip(*hits_data)
        ax.scatter(pe_arr, vr_arr, c=cond["color"], label=cond["label"].replace("\n", " "),
                   alpha=0.7, s=40, edgecolors='gray', linewidths=0.5)

ax.axvline(x=HIT_THRESHOLD * 100, color='red', linestyle='--', alpha=0.5, label=f"Hit threshold ({HIT_THRESHOLD*100:.0f}cm)")
ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label="Active threshold (0.3 m/s)")
ax.set_xlabel("Position Error at Hit (cm)")
ax.set_ylabel("Racket Speed at Hit (m/s)")
ax.set_title("v6 Ablation: Position Error vs Racket Speed")
ax.legend(fontsize=9)
ax.set_xlim(0, 30)
ax.set_ylim(-0.1, 2.0)
plt.tight_layout()
fig2_path = OUT_DIR / "ablation_v6_scatter.png"
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"散点图已保存到 {fig2_path}")
