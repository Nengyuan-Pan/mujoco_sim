"""V10 消融实验绘图脚本。"""

import csv
import sys
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


@dataclass
class ExpResult:
    name: str
    label: str
    n_valid: int
    n_hit: int
    n_active: int
    hit_rate: float
    active_rate: float
    avg_pos_error: float
    std_pos_error: float
    avg_v_racket: float
    std_v_racket: float
    pos_errors: list
    v_rackets: list
    hit_types: list


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def summarize(name, label, rows):
    valid = [r for r in rows if r["hit_type"] != "error"]
    n_valid = len(valid)
    pos_errors = [float(r["pos_error"]) for r in valid]
    v_rackets = [float(r["v_racket"]) for r in valid]
    hit_types = [r["hit_type"] for r in valid]

    n_hit = sum(1 for t in hit_types if t in ("active", "passive"))
    n_active = sum(1 for t in hit_types if t == "active")

    return ExpResult(
        name=name, label=label, n_valid=n_valid,
        n_hit=n_hit, n_active=n_active,
        hit_rate=n_hit / n_valid * 100 if n_valid else 0,
        active_rate=n_active / n_valid * 100 if n_valid else 0,
        avg_pos_error=np.mean(pos_errors) * 100 if pos_errors else 0,
        std_pos_error=np.std(pos_errors) * 100 if pos_errors else 0,
        avg_v_racket=np.mean(v_rackets) if v_rackets else 0,
        std_v_racket=np.std(v_rackets) if v_rackets else 0,
        pos_errors=pos_errors, v_rackets=v_rackets, hit_types=hit_types,
    )


def plot_hit_rate(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["w/o Tube\nw/o Softmin", "w/ Tube\nw/o Softmin", "w/ Tube\nw/ Softmin"]
    x = np.arange(len(labels))
    width = 0.32

    nom = results[::2]
    per = results[1::2]

    bars1 = ax.bar(x - width / 2, [r.hit_rate for r in nom], width,
                   label="Nominal", color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, [r.hit_rate for r in per], width,
                   label="Perturbed (±300ms/±15cm)", color="#ED7D31",
                   edgecolor="black", linewidth=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_title("V10 (40cm offset, no backswing, no follow-through)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper left")
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_active_rate(results, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["w/o Tube\nw/o Softmin", "w/ Tube\nw/o Softmin", "w/ Tube\nw/ Softmin"]
    x = np.arange(len(labels))
    width = 0.32

    nom = results[::2]
    per = results[1::2]

    bars1 = ax.bar(x - width / 2, [r.active_rate for r in nom], width,
                   label="Nominal", color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, [r.active_rate for r in per], width,
                   label="Perturbed (±300ms/±15cm)", color="#ED7D31",
                   edgecolor="black", linewidth=0.8)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Active Hit Rate (%)", fontsize=13)
    ax.set_title("V10 Active Hit Rate (v_racket > 2 m/s at contact)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pos_error_boxplot(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["w/o Tube\nw/o Softmin", "w/ Tube\nw/o Softmin", "w/ Tube\nw/ Softmin"]
    colors = ["#4472C4", "#70AD47", "#FFC000"]

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
                            medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

        cur_ax.set_title(title, fontsize=14, fontweight="bold")
        cur_ax.set_ylabel("Position Error (cm)" if ax_idx == 0 else "", fontsize=13)
        cur_ax.axhline(y=HIT_THRESHOLD * 100, color="red", linestyle="--",
                       alpha=0.7, linewidth=1.2, label="Contact threshold (15.3cm)")
        cur_ax.legend(fontsize=10, loc="upper right")
        cur_ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_v_racket_boxplot(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["w/o Tube\nw/o Softmin", "w/ Tube\nw/o Softmin", "w/ Tube\nw/ Softmin"]
    colors = ["#4472C4", "#70AD47", "#FFC000"]

    for ax_idx, (title, group) in enumerate([
        ("Nominal", results[::2]), ("Perturbed (±300ms/±15cm)", results[1::2]),
    ]):
        cur_ax = axes[ax_idx]
        data = []
        for r in group:
            hit_v = [v for v, ht in zip(r.v_rackets, r.hit_types)
                     if ht in ("active", "passive")]
            data.append(hit_v if hit_v else [0])

        bp = cur_ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)

        cur_ax.set_title(title, fontsize=14, fontweight="bold")
        cur_ax.set_ylabel("Racket Speed at Hit (m/s)" if ax_idx == 0 else "", fontsize=13)
        cur_ax.axhline(y=2.0, color="red", linestyle="--",
                       alpha=0.7, linewidth=1.2, label="Active threshold (2.0 m/s)")
        cur_ax.legend(fontsize=10, loc="upper right")
        cur_ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary_table(results, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    col_labels = ["Method", "Condition", "Valid", "Hit", "Active",
                  "Hit Rate", "Active\nRate", "Pos Err\n(cm)", "v_racket\n(m/s)"]
    table_data = []
    cell_colors = []

    for r in results:
        parts = r.label.split("\n")
        method = f"{parts[0]} {parts[1]}" if len(parts) > 1 else parts[0]
        cond = parts[2] if len(parts) > 2 else ""
        row = [
            method, cond,
            str(r.n_valid), str(r.n_hit), str(r.n_active),
            f"{r.hit_rate:.1f}%", f"{r.active_rate:.1f}%",
            f"{r.avg_pos_error:.1f}±{r.std_pos_error:.1f}",
            f"{r.avg_v_racket:.2f}±{r.std_v_racket:.2f}",
        ]
        table_data.append(row)
        if r.hit_rate >= 80:
            color = ["#C6EFCE"] * len(col_labels)
        elif r.hit_rate >= 40:
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
    results_dir = Path(__file__).resolve().parent.parent / "results" / "v10_ablation"
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    exp_defs = [
        ("v10_notube_nominal", "w/o Tube\nw/o Softmin\nNominal"),
        ("v10_notube_perturb", "w/o Tube\nw/o Softmin\nPerturbed"),
        ("v10_nosoftmin_nominal", "w/ Tube\nw/o Softmin\nNominal"),
        ("v10_nosoftmin_perturb", "w/ Tube\nw/o Softmin\nPerturbed"),
        ("v10_full_nominal", "w/ Tube\nw/ Softmin\nNominal"),
        ("v10_full_perturb", "w/ Tube\nw/ Softmin\nPerturbed"),
    ]

    results = []
    for csv_name, label in exp_defs:
        path = results_dir / f"{csv_name}.csv"
        if not path.exists():
            print(f"[WARN] {path} not found")
            continue
        rows = load_csv(path)
        r = summarize(csv_name, label, rows)
        results.append(r)

    if len(results) < 6:
        print(f"[ERROR] Only {len(results)}/6 experiments loaded")
        return

    print(f"\nResults from: {results_dir}")
    print(f"{'Method':<30} {'Cond':<6} {'Valid':>5} {'Hit':>4} {'Act':>4} "
          f"{'Hit%':>6} {'Act%':>6} {'Pos(cm)':>12} {'v(m/s)':>12}")
    print("-" * 100)
    for r in results:
        cond = "Nom" if "Nominal" in r.label else "Pert"
        print(f"{r.name:<30} {cond:<6} {r.n_valid:>5} {r.n_hit:>4} {r.n_active:>4} "
              f"{r.hit_rate:>5.1f}% {r.active_rate:>5.1f}% "
              f"{r.avg_pos_error:>7.1f}±{r.std_pos_error:<4.1f} "
              f"{r.avg_v_racket:>7.2f}±{r.std_v_racket:<4.2f}")

    plot_hit_rate(results, results_dir / "fig1_hit_rate.png")
    plot_active_rate(results, results_dir / "fig2_active_rate.png")
    plot_pos_error_boxplot(results, results_dir / "fig3_pos_error.png")
    plot_v_racket_boxplot(results, results_dir / "fig4_v_racket.png")
    plot_summary_table(results, results_dir / "fig5_summary_table.png")
    print(f"\nDone! Plots saved in: {results_dir}")


if __name__ == "__main__":
    main()
