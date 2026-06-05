"""v8 ablation experiment: summarize data + generate plots."""

import sys
import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 12

HIT_THRESHOLD = 0.153
ACTIVE_THRESHOLD = 0.3
HIT_LABEL = "15.3cm"


@dataclass
class ExperimentResult:
    name: str
    label: str
    n_valid: int
    n_hit: int
    n_active: int
    n_passive: int
    n_miss: int
    hit_rate: float
    active_rate: float
    avg_pos_error: float
    std_pos_error: float
    avg_v_racket: float
    std_v_racket: float
    avg_tube_ready_ms: float
    pos_errors: list
    v_rackets: list
    hit_types: list


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize(name, label, rows):
    valid_rows = [r for r in rows if r["hit_type"] != "error"]
    n_valid = len(valid_rows)
    pos_errors = [float(r["pos_error"]) for r in valid_rows]
    v_rackets = [float(r["v_racket"]) for r in valid_rows]
    tube_readys = [float(r["tube_ready_ms"]) for r in valid_rows]
    hit_types = [r["hit_type"] for r in valid_rows]

    n_hit = sum(1 for t in hit_types if t in ("active", "passive"))
    n_active = sum(1 for t in hit_types if t == "active")
    n_passive = sum(1 for t in hit_types if t == "passive")
    n_miss = sum(1 for t in hit_types if t == "miss")

    return ExperimentResult(
        name=name, label=label, n_valid=n_valid, n_hit=n_hit,
        n_active=n_active, n_passive=n_passive, n_miss=n_miss,
        hit_rate=n_hit / n_valid * 100 if n_valid > 0 else 0,
        active_rate=n_active / n_valid * 100 if n_valid > 0 else 0,
        avg_pos_error=np.mean(pos_errors) * 100 if pos_errors else 0,
        std_pos_error=np.std(pos_errors) * 100 if pos_errors else 0,
        avg_v_racket=np.mean(v_rackets) if v_rackets else 0,
        std_v_racket=np.std(v_rackets) if v_rackets else 0,
        avg_tube_ready_ms=np.mean(tube_readys) if tube_readys else 0,
        pos_errors=pos_errors, v_rackets=v_rackets, hit_types=hit_types,
    )


def plot_hit_rate_3methods(results, save_path):
    """3 方法消融图（Softmin-only = Tube+Softmin 时合并展示）"""
    fig, ax = plt.subplots(figsize=(9, 5))
    no_perturb = results[::2]
    perturb = results[1::2]
    labels = ["w/o Tube\nw/o Softmin", "w/ Softmin\n(Tube+/-)",
              "w/ Tube\nw/o Softmin"]
    x = np.arange(len(labels))
    width = 0.32

    bars1 = ax.bar(x - width / 2, [r.hit_rate for r in no_perturb], width,
                   label="Nominal", color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, [r.hit_rate for r in perturb], width,
                   label="Perturbed (±300ms/±15cm)", color="#ED7D31",
                   edgecolor="black", linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 标注 softmin-only = Tube+Softmin
    ax.annotate("Tube+Softmin ≡ Softmin-only\n(Tube corridor effect ≈ 0)",
                xy=(1, perturb[1].hit_rate), xytext=(1.6, perturb[1].hit_rate + 20),
                fontsize=9, fontstyle="italic", color="#C00000",
                arrowprops=dict(arrowstyle="->", color="#C00000", lw=1.2))

    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=11, loc="upper left")
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pos_error_3methods(results, save_path):
    """3 方法位置误差箱线图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["w/o Tube\nw/o Softmin", "w/ Softmin\n(Tube+/-)", "w/ Tube\nw/o Softmin"]
    colors = ["#4472C4", "#FFC000", "#70AD47"]
    for ax_idx, (title, group) in enumerate([
        ("Nominal", results[::2]), ("Perturbed (±300ms/±15cm)", results[1::2]),
    ]):
        cur_ax = axes[ax_idx]
        data = []
        for r in group:
            hit_pos = [pe * 100 for pe, ht in zip(r.pos_errors, r.hit_types)
                       if ht in ("active", "passive")]
            data.append(hit_pos if hit_pos else [0])

        bp = cur_ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

        cur_ax.set_title(title, fontsize=14, fontweight="bold")
        cur_ax.set_ylabel("Position Error (cm)" if ax_idx == 0 else "", fontsize=13)
        cur_ax.axhline(y=HIT_THRESHOLD * 100, color="red", linestyle="--",
                       alpha=0.7, linewidth=1.2, label=f"Contact threshold ({HIT_LABEL} cm)")
        cur_ax.legend(fontsize=10, loc="upper right")
        cur_ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pos_error_boxplot(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax_idx, (title, group) in enumerate([
        ("Nominal", results[::2]), ("Perturbed", results[1::2]),
    ]):
        cur_ax = axes[ax_idx]
        labels = ["w/o Tube\nw/o Softmin", "w/o Tube\nw/ Softmin",
                  "w/ Tube\nw/o Softmin", "w/ Tube\nw/ Softmin"]
        data = []
        for r in group:
            hit_pos = [pe * 100 for pe, ht in zip(r.pos_errors, r.hit_types)
                       if ht in ("active", "passive")]
            data.append(hit_pos if hit_pos else [0])

        bp = cur_ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
        colors = ["#4472C4", "#5B9BD5", "#70AD47", "#FFC000"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

        cur_ax.set_title(title, fontsize=14, fontweight="bold")
        cur_ax.set_ylabel("Position Error (cm)" if ax_idx == 0 else "", fontsize=13)
        cur_ax.axhline(y=HIT_THRESHOLD * 100, color="red", linestyle="--",
                       alpha=0.7, linewidth=1.2, label=f"Contact threshold ({HIT_LABEL} cm)")
        cur_ax.legend(fontsize=10, loc="upper right")
        cur_ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_combined_table(results, save_path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    col_labels = ["Method", "Condition", "Valid", "Hit", "Active",
                  "Hit Rate", "Active\nRate", "Pos Err\n(cm)", "v_racket\n(m/s)"]
    table_data = []
    cell_colors = []
    for r in results:
        label_parts = r.label.split("\n")
        method = f"{label_parts[0]} {label_parts[1]}" if len(label_parts) > 1 else label_parts[0]
        cond = label_parts[2] if len(label_parts) > 2 else ""
        row = [
            method,
            cond,
            str(r.n_valid), str(r.n_hit), str(r.n_active),
            f"{r.hit_rate:.1f}%", f"{r.active_rate:.1f}%",
            f"{r.avg_pos_error:.1f}+/-{r.std_pos_error:.1f}",
            f"{r.avg_v_racket:.2f}+/-{r.std_v_racket:.2f}",
        ]
        table_data.append(row)
        if r.hit_rate >= 90:
            color = ["#C6EFCE"] * len(col_labels)
        elif r.hit_rate >= 50:
            color = ["#FFEB9C"] * len(col_labels)
        else:
            color = ["#FFC7CE"] * len(col_labels)
        cell_colors.append(color)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=["#D9E2F3"] * len(col_labels),
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#B4C6E7")
        cell.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot/plot_v8_results.py <results_dir>")
        return

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return

    # 2x2 消融实验：Tube (有/无) × Softmin (有/无)
    exp_defs = [
        ("v8_notube_no_perturb", "w/o Tube\nw/o Softmin\nNominal"),
        ("v8_notube_perturb", "w/o Tube\nw/o Softmin\nPerturbed"),
        ("v8_softmin_only_no_perturb", "w/o Tube\nw/ Softmin\nNominal"),
        ("v8_softmin_only_perturb", "w/o Tube\nw/ Softmin\nPerturbed"),
        ("v8_nosoftmin_no_perturb", "w/ Tube\nw/o Softmin\nNominal"),
        ("v8_nosoftmin_perturb", "w/ Tube\nw/o Softmin\nPerturbed"),
        ("v8_default_no_perturb", "w/ Tube\nw/ Softmin\nNominal"),
        ("v8_default_perturb", "w/ Tube\nw/ Softmin\nPerturbed"),
    ]

    results = []
    for csv_name, label in exp_defs:
        path = results_dir / f"{csv_name}.csv"
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        rows = load_csv(path)
        r = summarize(csv_name, label, rows)
        results.append(r)

    if len(results) < 6:
        print(f"[ERROR] Only loaded {len(results)}/8 experiments")
        return

    # 检测 Tube+Softmin 是否与 Softmin-only 命中率一致，合并用于柱状图/箱线图
    soft_only_nom = next((r for r in results if r.name == "v8_softmin_only_no_perturb"), None)
    full_nom = next((r for r in results if r.name == "v8_default_no_perturb"), None)
    soft_only_per = next((r for r in results if r.name == "v8_softmin_only_perturb"), None)
    full_per = next((r for r in results if r.name == "v8_default_perturb"), None)

    merge = False
    if soft_only_nom and full_nom and soft_only_per and full_per:
        if (abs(soft_only_nom.hit_rate - full_nom.hit_rate) < 0.01 and
                abs(soft_only_per.hit_rate - full_per.hit_rate) < 0.01):
            merge = True
            print("[INFO] Tube+Softmin 数据与 Softmin-only 命中率一致，图表合并展示")

    plot_results = [r for r in results if r.name not in ("v8_default_no_perturb", "v8_default_perturb")] if merge else results

    print(f"\nResults from: {results_dir}")
    print(f"{'Method':<30} {'Cond':<6} {'Valid':>5} {'Hit':>4} {'Act':>4} "
          f"{'Hit%':>6} {'Act%':>6} {'Pos(cm)':>10} {'v(m/s)':>10}")
    print("-" * 90)
    for r in results:
        cond = "Nom" if "no_perturb" in r.name else "Pert"
        print(f"{r.name:<30} {cond:<6} {r.n_valid:>5} {r.n_hit:>4} {r.n_active:>4} "
              f"{r.hit_rate:>5.1f}% {r.active_rate:>5.1f}% "
              f"{r.avg_pos_error:>7.1f}+/-{r.std_pos_error:<3.1f} "
              f"{r.avg_v_racket:>7.2f}+/-{r.std_v_racket:<3.2f}")

    # 柱状图/箱线图用合并后的数据（3方法），表格用完整数据（4方法）
    plot_hit_rate_3methods(plot_results, results_dir / "fig1_hit_rate.png")
    plot_pos_error_3methods(plot_results, results_dir / "fig3_pos_error.png")
    plot_combined_table(results, results_dir / "fig5_summary_table.png")
    print(f"\nDone! Plots saved in: {results_dir}")


if __name__ == "__main__":
    main()
