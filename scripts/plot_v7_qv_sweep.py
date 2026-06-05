"""生成 V7 Q_v/Q_p 调参实验的图表。"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "exp_v7_qv_sweep_20260604"
OUT_DIR.mkdir(parents=True, exist_ok=True)

conditions = ["baseline", "Q_v×4", "Q_v_near×2", "Q_p降低", "Q_v×4+Q_p降低"]
hit_counts = [34, 34, 34, 35, 35]
total = 47
hit_rates = [c / total * 100 for c in hit_counts]
avg_v = [1.283, 1.332, 1.284, 1.308, 1.293]
std_v = [0.224, 0.265, 0.224, 0.296, 0.219]
min_v = [0.798, 0.484, 0.798, 0.362, 0.712]
max_v = [1.722, 1.778, 1.722, 1.741, 1.710]
avg_pos = [0.0420, 0.0348, 0.0420, 0.0506, 0.0566]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(conditions))
width = 0.35

bars1 = ax1.bar(x - width / 2, hit_rates, width, label="命中率 (%)", color="#4CAF50", alpha=0.8)
ax1.set_ylabel("命中率 (%)", fontsize=13)
ax1.set_ylim(60, 85)
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=11)
ax1.tick_params(axis="y", labelsize=11)
for bar, val in zip(bars1, hit_rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, avg_v, width, label="球拍速度 (m/s)", color="#2196F3", alpha=0.8)
ax2.set_ylabel("击球瞬间球拍速度 (m/s)", fontsize=13)
ax2.set_ylim(0, 2.5)
ax2.tick_params(axis="y", labelsize=11)
for bar, val, sv in zip(bars2, avg_v, std_v):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
             f"{val:.2f}±{sv:.2f}", ha="center", va="bottom", fontsize=9)

fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=11)
ax1.set_title("V7 (终端=击球点, 偏移=0, 硬约束) Q_v/Q_p 调参: 命中率 vs 球拍速度\n(50 seeds, ball_speed=7m/s, TCP≤1.8m/s, exempt=0)", fontsize=13, pad=15)
fig.tight_layout()
fig.savefig(OUT_DIR / "v7_qv_sweep_hitrate_speed.png", dpi=150, bbox_inches="tight")
plt.close()

fig2, ax3 = plt.subplots(figsize=(10, 6))
box_data = []
colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
for i in range(len(conditions)):
    q1 = max(min_v[i], avg_v[i] - std_v[i])
    med = avg_v[i]
    q3 = min(max_v[i], avg_v[i] + std_v[i])
    box_data.append({"med": med, "q1": q1, "q3": q3, "whislo": min_v[i], "whishi": max_v[i], "label": conditions[i]})

bp = ax3.bxp(box_data, positions=np.arange(len(conditions)), showfliers=False, patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp["medians"]:
    median.set(color="black", linewidth=2)

ax3.axhline(y=1.8, color="red", linestyle="--", alpha=0.5, label="TCP 限速 1.8 m/s")
ax3.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="主动击球阈值 0.3 m/s")
ax3.set_xticks(np.arange(len(conditions)))
ax3.set_xticklabels(conditions, fontsize=11)
ax3.set_ylabel("击球瞬间球拍速度 (m/s)", fontsize=13)
ax3.set_title("V7 球拍速度分布 (min/avg±std/max)", fontsize=14, pad=15)
ax3.legend(fontsize=10)
ax3.tick_params(axis="y", labelsize=11)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "v7_qv_sweep_speed_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

fig3, ax4 = plt.subplots(figsize=(8, 6))
for i, cond in enumerate(conditions):
    ax4.scatter(avg_pos[i], avg_v[i], s=200, c=colors[i], label=cond, zorder=5, edgecolors="black", linewidth=0.5)
    ax4.annotate(cond, (avg_pos[i], avg_v[i]), textcoords="offset points", xytext=(10, 5), fontsize=10)
ax4.set_xlabel("命中 pos_error avg (m) — 越小越好 →", fontsize=12)
ax4.set_ylabel("击球瞬间球拍速度 avg (m/s) — 越大越好 ↑", fontsize=12)
ax4.set_title("V7 Q_v/Q_p 调参: 位置精度 vs 球拍速度 (帕累托分析)", fontsize=14, pad=15)
ax4.invert_xaxis()
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=11)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "v7_qv_sweep_pareto.png", dpi=150, bbox_inches="tight")
plt.close()

csv_path = OUT_DIR / "v7_qv_sweep_results.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("condition,hit_rate,pct,avg_pos_error,avg_v_racket,std_v_racket,min_v_racket,max_v_racket\n")
    for i in range(len(conditions)):
        f.write(f"{conditions[i]},{hit_counts[i]}/{total},{hit_rates[i]:.1f},"
                f"{avg_pos[i]:.4f},{avg_v[i]:.3f},{std_v[i]:.3f},{min_v[i]:.3f},{max_v[i]:.3f}\n")

print(f"图表已保存到: {OUT_DIR}")
