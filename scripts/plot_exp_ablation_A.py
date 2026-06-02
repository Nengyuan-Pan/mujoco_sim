"""消融实验A 图表生成: Bar chart + corridor ratio trend."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATE = "20260602"
OUT_DIR = ROOT / "results" / f"exp_ablation_corridor_{DATE}"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'cm'

with open(OUT_DIR / "raw_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

COND_ORDER = [
    "no_tube", "corridor_1.0", "corridor_0.3",
    "softmin_only", "s+c0.1", "s+c0.3", "s+c0.5", "s+c1.0",
]
LABELS = [
    "No-Tube", "Corridor\nonly", "Corridor\n(r=0.3)",
    "Softmin\nonly", "S+0.1C", "S+0.3C\n(default)", "S+0.5C", "S+1.0C",
]

stats = {}
for name in COND_ORDER:
    runs = [r for r in data[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    stats[name] = {
        "rate": rate, "n_hit": len(hits), "n_total": len(runs),
        "pe_mean": np.mean(pe), "pe_std": np.std(pe),
    }

# ===== 图1: 柱状图 (成功率 + 位置误差) =====
fig, ax1 = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(COND_ORDER))
w = 0.55

colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4', '#4a90d9', '#aec7e8']

rates = [stats[c]["rate"] for c in COND_ORDER]
pe_mean = [stats[c]["pe_mean"] for c in COND_ORDER]
pe_std = [stats[c]["pe_std"] for c in COND_ORDER]

bars = ax1.bar(x, rates, w, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
for i, (b, r) in enumerate(zip(bars, rates)):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
             f'{r:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_ylabel('Success Rate (%)', fontsize=11)
ax1.set_ylim(0, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=8)
ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)

ax2 = ax1.twinx()
ax2.errorbar(x, pe_mean, yerr=pe_std, fmt='ko-', markersize=5, capsize=3,
             linewidth=1.2, zorder=5, label='Pos. Error')
ax2.set_ylabel('Position Error (cm)', fontsize=11)
ax2.set_ylim(0, 45)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
from matplotlib.patches import Patch
legend_bars = [Patch(facecolor=c, edgecolor='black', linewidth=0.5) for c in colors]
legend_labels = LABELS
ax1.legend(legend_bars + lines2, legend_labels + [labels2[0]],
           loc='upper left', fontsize=7, ncol=3, framealpha=0.9)

plt.title('Ablation: Softmin vs Corridor Contribution', fontsize=12, pad=10)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f"ablation_bar.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved ablation_bar.pdf/png")

# ===== 图2: Corridor Ratio Trend (softmin ON) =====
fig2, ax = plt.subplots(figsize=(5.5, 4))

softmin_conditions = ["softmin_only", "s+c0.1", "s+c0.3", "s+c0.5", "s+c1.0"]
ratios = [0.0, 0.1, 0.3, 0.5, 1.0]
s_rates = [stats[c]["rate"] for c in softmin_conditions]
s_pe = [stats[c]["pe_mean"] for c in softmin_conditions]
s_pe_std = [stats[c]["pe_std"] for c in softmin_conditions]

ax.plot(ratios, s_rates, 'bs-', markersize=8, linewidth=1.5, label='Success Rate', zorder=5)
ax.set_xlabel('Corridor Cost Ratio $r$', fontsize=11)
ax.set_ylabel('Success Rate (%)', fontsize=11, color='blue')
ax.set_ylim(0, 110)
ax.tick_params(axis='y', labelcolor='blue')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax_r = ax.twinx()
ax_r.errorbar(ratios, s_pe, yerr=s_pe_std, fmt='ro--', markersize=6, capsize=3,
              linewidth=1.2, label='Pos. Error')
ax_r.set_ylabel('Position Error (cm)', fontsize=11, color='red')
ax_r.tick_params(axis='y', labelcolor='red')
ax_r.set_ylim(0, 40)

lines_a, labels_a = ax.get_legend_handles_labels()
lines_b, labels_b = ax_r.get_legend_handles_labels()
ax.legend(lines_a + lines_b, labels_a + labels_b, loc='center right', fontsize=8)

plt.title('Softmin + Varying Corridor Ratio', fontsize=12, pad=10)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig2.savefig(OUT_DIR / f"ablation_trend.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved ablation_trend.pdf/png")

# ===== 图3: No-Softmin baseline comparison =====
fig3, ax3 = plt.subplots(figsize=(7, 4))
no_soft_conds = ["corridor_0.3", "corridor_1.0"]
soft_conds = ["s+c0.3", "s+c1.0"]
compare_labels = [
    "Corridor r=0.3\n(no softmin)", "Corridor r=1.0\n(no softmin)",
    "Softmin+Corridor r=0.3", "Softmin+Corridor r=1.0",
]
compare_rates = [stats[c]["rate"] for c in no_soft_conds + soft_conds]
compare_pe = [stats[c]["pe_mean"] for c in no_soft_conds + soft_conds]
compare_pe_std = [stats[c]["pe_std"] for c in no_soft_conds + soft_conds]

xc = np.arange(4)
c_colors = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4']
bc = ax3.bar(xc, compare_rates, 0.55, color=c_colors, edgecolor='black', linewidth=0.5, zorder=3)
for i, (b, r) in enumerate(zip(bc, compare_rates)):
    ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
             f'{r:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('Success Rate (%)', fontsize=11)
ax3.set_ylim(0, 115)
ax3.set_xticks(xc)
ax3.set_xticklabels(compare_labels, fontsize=8)
ax3.yaxis.grid(True, alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)

ax3r = ax3.twinx()
ax3r.errorbar(xc, compare_pe, yerr=compare_pe_std, fmt='ko-', markersize=5,
              capsize=3, linewidth=1.2, zorder=5)
ax3r.set_ylabel('Position Error (cm)', fontsize=11)
ax3r.set_ylim(0, 45)

plt.title('Softmin is the Key Factor', fontsize=12, pad=10)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig3.savefig(OUT_DIR / f"ablation_softmin_key.{ext}", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved ablation_softmin_key.pdf/png")

print(f"\nAll figures saved to {OUT_DIR}")
