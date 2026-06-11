"""V9 消融实验绘图: 2^3 因子 (Tube × Softmin × Follow-through)。"""

import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "SimSun"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 11


@dataclass
class ExpResult:
    name: str
    label: str
    tube: bool
    softmin: bool
    follow: bool
    perturbed: bool
    n_valid: int
    n_hit: int
    n_active: int
    hit_rate: float
    active_rate: float
    avg_pos_error: float
    avg_v_racket: float
    pos_errors: list
    v_rackets: list
    hit_types: list


NAMES = [
    "none_nominal", "tube_only_nominal", "soft_only_nominal", "follow_only_nominal",
    "tube_soft_nominal", "tube_follow_nominal", "soft_follow_nominal", "full_nominal",
    "none_perturb", "tube_only_perturb", "soft_only_perturb", "follow_only_perturb",
    "tube_soft_perturb", "tube_follow_perturb", "soft_follow_perturb", "full_perturb",
]

SHORT_LABELS = {
    (False, False, False): "Baseline",
    (True,  False, False): "Tube",
    (False, True,  False): "Softmin",
    (False, False, True):  "Follow",
    (True,  True,  False): "Tube+Soft",
    (True,  False, True):  "Tube+Follow",
    (False, True,  True):  "Soft+Follow",
    (True,  True,  True):  "Full",
}

COMBO_ORDER = [
    (False, False, False),
    (True,  False, False),
    (False, True,  False),
    (False, False, True),
    (True,  True,  False),
    (True,  False, True),
    (False, True,  True),
    (True,  True,  True),
]


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(name, rows, tube, soft, follow, perturbed):
    valid = [r for r in rows if r["hit_type"] != "error"]
    n = len(valid)
    pe = [float(r["pos_error"]) for r in valid]
    vr = [float(r["v_racket"]) for r in valid]
    ht = [r["hit_type"] for r in valid]
    n_hit = sum(1 for t in ht if t in ("active", "passive"))
    n_act = sum(1 for t in ht if t == "active")
    hit_pe = [float(r["pos_error"]) for r in valid if r["hit_type"] in ("active", "passive")]
    hit_vr = [float(r["v_racket"]) for r in valid if r["hit_type"] in ("active", "passive")]
    return ExpResult(
        name=name, label=SHORT_LABELS[(tube, soft, follow)],
        tube=tube, softmin=soft, follow=follow, perturbed=perturbed,
        n_valid=n, n_hit=n_hit, n_active=n_act,
        hit_rate=n_hit / n * 100 if n else 0,
        active_rate=n_act / n * 100 if n else 0,
        avg_pos_error=np.mean(hit_pe) * 100 if hit_pe else 0,
        avg_v_racket=np.mean(hit_vr) if hit_vr else 0,
        pos_errors=[float(r["pos_error"]) for r in valid],
        v_rackets=vr, hit_types=ht,
    )


def plot_hit_rate_bar(all_r, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [SHORT_LABELS[c] for c in COMBO_ORDER]
    x = np.arange(len(labels))
    w = 0.35

    nom = [next(r for r in all_r if (r.tube, r.softmin, r.follow) == c and not r.perturbed) for c in COMBO_ORDER]
    per = [next(r for r in all_r if (r.tube, r.softmin, r.follow) == c and r.perturbed) for c in COMBO_ORDER]

    b1 = ax.bar(x - w/2, [r.hit_rate for r in nom], w, label="Nominal",
                color="#4472C4", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, [r.hit_rate for r in per], w, label="Random Perturb",
                color="#ED7D31", edgecolor="black", linewidth=0.6)

    for b in list(b1) + list(b2):
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Hit Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.axhline(90, color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("V9 Ablation: Hit Rate (Tube × Softmin × Follow-through)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_active_rate_bar(all_r, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    labels = [SHORT_LABELS[c] for c in COMBO_ORDER]
    x = np.arange(len(labels))
    w = 0.35

    nom = [next(r for r in all_r if (r.tube, r.softmin, r.follow) == c and not r.perturbed) for c in COMBO_ORDER]
    per = [next(r for r in all_r if (r.tube, r.softmin, r.follow) == c and r.perturbed) for c in COMBO_ORDER]

    b1 = ax.bar(x - w/2, [r.active_rate for r in nom], w, label="Nominal",
                color="#4472C4", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, [r.active_rate for r in per], w, label="Random Perturb",
                color="#ED7D31", edgecolor="black", linewidth=0.6)

    for b in list(b1) + list(b2):
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Active Hit Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("V9 Ablation: Active Hit Rate (v_racket > 2 m/s)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pos_error_box(all_r, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    labels = [SHORT_LABELS[c] for c in COMBO_ORDER]
    colors = ["#A5A5A5", "#4472C4", "#FFC000", "#70AD47",
              "#5B9BD5", "#ED7D31", "#BDD7EE", "#FF6B6B"]

    for ax_idx, (title, pert) in enumerate([
        ("Nominal", False), ("Random Perturb (t: +/-50-100ms / s: +/-3-8cm)", True)
    ]):
        ax = axes[ax_idx]
        data = []
        for c in COMBO_ORDER:
            r = next(r for r in all_r if (r.tube, r.softmin, r.follow) == c and r.perturbed == pert)
            hit_pe = [pe * 100 for pe, ht in zip(r.pos_errors, r.hit_types) if ht in ("active", "passive")]
            data.append(hit_pe if hit_pe else [0])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Position Error (cm)" if ax_idx == 0 else "", fontsize=12)
        ax.axhline(15.3, color="red", ls="--", alpha=0.6, lw=1.2, label="Contact threshold (15.3cm)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_heatmap(all_r, save_path):
    """2×2 热力图: Softmin (行) × Follow (列), 每个 cell 里 Nominal/Perturb 两个值。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (title, pert) in enumerate([("Nominal", False), ("Perturbed", True)]):
        ax = axes[ax_idx]
        mat = np.zeros((2, 2))
        for i, soft in enumerate([False, True]):
            for j, follow in enumerate([False, True]):
                r = next(r for r in all_r if r.softmin == soft and r.follow == follow
                         and not r.tube and r.perturbed == pert)
                mat[i, j] = r.hit_rate

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["w/o Follow", "w/ Follow"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["w/o Softmin", "w/ Softmin"])
        ax.set_xlabel("Follow-through")
        ax.set_ylabel("Softmin")
        ax.set_title(f"{title} (w/o Tube)", fontsize=13)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if mat[i, j] < 40 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Hit Rate (%)")

    fig.suptitle("V9 Ablation Heatmap: Softmin × Follow-through (without Tube)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_heatmap_tube(all_r, save_path):
    """2×2 热力图: Tube(行) × Softmin(列), Follow=off。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (title, pert) in enumerate([("Nominal", False), ("Perturbed", True)]):
        ax = axes[ax_idx]
        mat = np.zeros((2, 2))
        for i, tube in enumerate([False, True]):
            for j, soft in enumerate([False, True]):
                r = next(r for r in all_r if r.tube == tube and r.softmin == soft
                         and not r.follow and r.perturbed == pert)
                mat[i, j] = r.hit_rate

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["w/o Softmin", "w/ Softmin"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["w/o Tube", "w/ Tube"])
        ax.set_xlabel("Softmin")
        ax.set_ylabel("Tube")
        ax.set_title(f"{title} (w/o Follow)", fontsize=13)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if mat[i, j] < 40 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Hit Rate (%)")

    fig.suptitle("V9 Ablation Heatmap: Tube × Softmin (without Follow-through)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary_table(all_r, save_path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    col = ["Tube", "Soft", "Follow", "Perturb", "Valid", "Hit", "Active",
           "Hit%", "Active%", "Pos(cm)", "v(m/s)"]
    data = []
    colors = []
    for r in all_r:
        row = [
            "Y" if r.tube else "-", "Y" if r.softmin else "-",
            "Y" if r.follow else "-", "Y" if r.perturbed else "-",
            str(r.n_valid), str(r.n_hit), str(r.n_active),
            f"{r.hit_rate:.0f}%", f"{r.active_rate:.0f}%",
            f"{r.avg_pos_error:.1f}", f"{r.avg_v_racket:.2f}",
        ]
        data.append(row)
        if r.hit_rate >= 90:
            c = ["#C6EFCE"] * len(col)
        elif r.hit_rate >= 60:
            c = ["#FFEB9C"] * len(col)
        elif r.hit_rate >= 30:
            c = ["#FFF2CC"] * len(col)
        else:
            c = ["#FFC7CE"] * len(col)
        colors.append(c)

    table = ax.table(cellText=data, colLabels=col, cellColours=colors,
                     colColours=["#D9E2F3"] * len(col),
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#B4C6E7")
        cell.set_linewidth(0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "v9_ablation"

    combos = [
        (False, False, False),
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
        (True,  True,  False),
        (True,  False, True),
        (False, True,  True),
        (True,  True,  True),
    ]

    all_r = []
    for tube, soft, follow in combos:
        for pert in [False, True]:
            tag = "perturb" if pert else "nominal"
            name = f"{SHORT_LABELS[(tube,soft,follow)].lower().replace('+','_')}_{tag}"
            # find matching name from file
            for n in NAMES:
                parts = n.replace("_nominal", "").replace("_perturb", "").split("_")
                t_flag = "tube" in parts or n.startswith("tube_") or n.startswith("full_")
                s_flag = "soft" in parts or n.startswith("soft_") or n.startswith("full_")
                f_flag = "follow" in parts or n.startswith("follow_") or n.startswith("full_")
                # more robust: match by experiment name
            # use exact names from EXPERIMENTS list
            mapping = {
                (False, False, False, False): "none_nominal",
                (True,  False, False, False): "tube_only_nominal",
                (False, True,  False, False): "soft_only_nominal",
                (False, False, True,  False): "follow_only_nominal",
                (True,  True,  False, False): "tube_soft_nominal",
                (True,  False, True,  False): "tube_follow_nominal",
                (False, True,  True,  False): "soft_follow_nominal",
                (True,  True,  True,  False): "full_nominal",
                (False, False, False, True):  "none_perturb",
                (True,  False, False, True):  "tube_only_perturb",
                (False, True,  False, True):  "soft_only_perturb",
                (False, False, True,  True):  "follow_only_perturb",
                (True,  True,  False, True):  "tube_soft_perturb",
                (True,  False, True,  True):  "tube_follow_perturb",
                (False, True,  True,  True):  "soft_follow_perturb",
                (True,  True,  True,  True):  "full_perturb",
            }
            csv_name = mapping[(tube, soft, follow, pert)]
            csv_path = results_dir / f"{csv_name}.csv"
            if not csv_path.exists():
                print(f"[WARN] {csv_path} not found")
                continue
            rows = load_csv(csv_path)
            r = summarize(csv_name, rows, tube, soft, follow, pert)
            all_r.append(r)

    print(f"\nLoaded {len(all_r)} experiments from {results_dir}")
    print(f"{'Name':<25} {'T':>2} {'S':>2} {'F':>2} {'P':>2} "
          f"{'Hit%':>6} {'Act%':>6} {'Pos':>6} {'v':>6}")
    print("-" * 70)
    for r in all_r:
        print(f"{r.name:<25} {'Y' if r.tube else '-':>2} {'Y' if r.softmin else '-':>2} "
              f"{'Y' if r.follow else '-':>2} {'Y' if r.perturbed else '-':>2} "
              f"{r.hit_rate:>5.0f}% {r.active_rate:>5.0f}% "
              f"{r.avg_pos_error:>5.1f}cm {r.avg_v_racket:>5.2f}m/s")

    plot_hit_rate_bar(all_r, results_dir / "fig1_hit_rate.png")
    plot_active_rate_bar(all_r, results_dir / "fig2_active_rate.png")
    plot_pos_error_box(all_r, results_dir / "fig3_pos_error.png")
    plot_heatmap(all_r, results_dir / "fig4_heatmap_soft_follow.png")
    plot_heatmap_tube(all_r, results_dir / "fig5_heatmap_tube_soft.png")
    plot_summary_table(all_r, results_dir / "fig6_summary_table.png")
    print(f"\nDone! Plots in: {results_dir}")


if __name__ == "__main__":
    main()
