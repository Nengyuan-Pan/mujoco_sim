# Skill: 论文图表生成（figure_generation）

## 目的
基于实验数据生成 IEEE RAL 论文级别的出版质量图表。
调用时机：实验数据收集完成后、论文撰写过程中需要插入图表时。

## 输出目录
```
paper/figures/
├── fig1_system_overview.pdf       # 系统场景图
├── fig2_algorithm_flowchart.pdf   # 算法流程图
├── fig3_tube_corridor.pdf         # Tube 空间走廊示意
├── fig4_joint_trajectory.pdf      # 关节轨迹+力矩
├── fig5_hit_rate_vs_speed.pdf     # 命中率 vs 球速
├── fig6_tube_robustness.pdf       # Tube 鲁棒性对比
├── fig7_realtime_performance.pdf  # 实时性能
├── fig8_tube_diagnostic.pdf       # Tube 诊断六子图
└── table_data/                    # LaTeX 表格数据
    ├── table1_comparison.tex
    └── table2_ablation.tex
```

## IEEE 格式规范

### 图片尺寸
| 类型 | 宽度 | 高度 | 说明 |
|------|------|------|------|
| 单栏图 | 3.5 in (8.89 cm) | ≤ 3.0 in | 一列宽 |
| 双栏图 | 7.16 in (18.19 cm) | ≤ 5.0 in | 两列宽 |
| 子图 | 按比例 | — | n×m 网格 |

### 字体与样式
```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})
```

### 色盲友好调色板
```python
COLORS = {
    "tube_on": "#0072B2",       # 蓝
    "tube_off": "#D55E00",      # 橙红
    "algo_cap": "#009E73",      # 绿
    "strict_joint": "#56B4E9",  # 天蓝
    "tcp_dual": "#E69F00",      # 琥珀
    "far_stage": "#CC79A7",     # 粉
    "near_stage": "#0072B2",    # 蓝
    "ball": "#D55E00",          # 橙红
    "racket": "#0072B2",        # 蓝
    "hit_window": "#E69F00",    # 琥珀
}
```

### 输出格式
- 优先矢量格式（PDF）
- 备选 300dpi PNG
- 文件名与论文中 `\label{fig:xxx}` 一致

---

## 图表 1：系统场景图（Fig.1）

### 说明
展示 RM-65 双臂机械臂 + 球拍 + 网球飞行的实际场景。

### 数据来源
MuJoCo 渲染截图。

### 生成方法
```bash
# 方法1：MuJoCo viewer 截图
python scripts/rm65_mpc_tube_constraint.py --serve-box --ball-speed 9 --seed 0 --viewer
# 在 viewer 中按 Ctrl+S 保存截图

# 方法2：离屏渲染（无头环境）
python -c "
import mujoco
import numpy as np
from pathlib import Path
from src.sim.rm65_env import RM65Env

model_path = Path('src/robot/rm65_model.xml')
env = RM65Env(model_path, dt=0.005)
# 设置初始姿态后渲染
env.reset(np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45]))
renderer = mujoco.Renderer(env.model, height=800, width=1200)
renderer.update_scene(env.data, camera='front_view')
img = renderer.render()
from PIL import Image
Image.fromarray(img).save('paper/figures/fig1_system_overview.png')
"
```

---

## 图表 2：算法流程图（Fig.2）

### 说明
展示 MPC+iLQR+Tube 三层框架的控制流。

### 生成方法
使用 TikZ 或 draw.io 手绘，不通过代码生成。模板：

```latex
% paper/figures/fig2_algorithm_flowchart.tex
% 使用 TikZ 绘制
\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=0.8cm, >=stealth,
    block/.style={rectangle, draw, fill=blue!10, minimum width=2.5cm, minimum height=0.6cm},
    decision/.style={diamond, draw, fill=orange!10, minimum width=1.5cm}]

% MPC 外循环
\node[block] (observe) {Observe ball state};
\node[block, below of=observe] (predict) {Predict trajectory \& search hit window};
\node[block, below of=predict] (tube) {Build Tube corridor};
\node[block, below of=tube] (ilqr) {iLQR backward + forward pass};
\node[block, below of=ilqr] (execute) {Execute with safety filter};

% 循环箭头
\draw[->] (observe) -- (predict);
\draw[->] (predict) -- (tube);
\draw[->] (tube) -- (ilqr);
\draw[->] (ilqr) -- (execute);
\draw[->] (execute.east) -- ++(1.5,0) |- (observe.east)
    node[midway, right] {replan interval};

% Tube 代价注释
\node[right=1cm of tube, text width=2cm, font=\scriptsize] {
    Perp. deviation\\
    Velocity direction\\
    Normal alignment
};

% 安全滤波注释
\node[right=1cm of execute, text width=2cm, font=\scriptsize] {
    Joint limits\\
    TCP speed limit\\
    X-plane wall
};

\end{tikzpicture}
\caption{Overview of the MPC+iLQR+Tube framework.}
\label{fig:algorithm_flowchart}
\end{figure}
```

---

## 图表 3：Tube 空间走廊示意（Fig.3）

### 说明
3D 展示球轨迹 + 不确定性管道 + 候选击球窗口 + 球拍轨迹。

### 数据来源
单次成功运行的 `HittingTube` 数据 + `ball_pos_history` + `racket_pos_history`。

### 生成代码

```python
"""生成 Fig.3: Tube 空间走廊示意图。

数据来源: experiment_data/exp1_algorithm_capability/raw/ 中的 NPZ 文件
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "figure.dpi": 300,
})

COLORS = {
    "ball": "#D55E00",
    "racket": "#0072B2",
    "tube": "#E69F00",
    "window": "#009E73",
}


def generate_fig3(
    ball_positions: np.ndarray,
    racket_positions: np.ndarray,
    tube_candidates_p: np.ndarray,
    tube_candidates_weights: np.ndarray,
    uncertainty_radius: np.ndarray,
    output_path: str = "paper/figures/fig3_tube_corridor.pdf",
) -> None:
    """生成 Tube 空间走廊示意图。

    Args:
        ball_positions: 球轨迹 (N, 3)
        racket_positions: 球拍轨迹 (M, 3)
        tube_candidates_p: Tube 候选位置 (K, 3)
        tube_candidates_weights: 候选权重 (K,)
        uncertainty_radius: 不确定性半径 (K,)
        output_path: 输出路径
    """
    fig = plt.figure(figsize=(7.16, 3.5))
    ax = fig.add_subplot(111, projection="3d")

    # 球轨迹
    ax.plot(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2],
            color=COLORS["ball"], alpha=0.6, linewidth=1.5, label="Ball trajectory")

    # 球拍轨迹
    ax.plot(racket_positions[:, 0], racket_positions[:, 1], racket_positions[:, 2],
            color=COLORS["racket"], alpha=0.8, linewidth=1.5, label="Racket trajectory")

    # Tube 候选窗口（带不确定性半径的球体）
    sizes = tube_candidates_weights * 200
    ax.scatter(tube_candidates_p[:, 0], tube_candidates_p[:, 1], tube_candidates_p[:, 2],
               c=COLORS["window"], s=sizes, alpha=0.5, marker="o", label="Hit window")

    # 不确定性管道：在候选点画圆环
    for i in range(0, len(tube_candidates_p), max(1, len(tube_candidates_p) // 5)):
        r = uncertainty_radius[i]
        center = tube_candidates_p[i]
        theta = np.linspace(0, 2 * np.pi, 20)
        circle_x = center[0] + r * np.cos(theta)
        circle_y = center[1] + r * np.sin(theta)
        circle_z = np.full_like(theta, center[2])
        ax.plot(circle_x, circle_y, circle_z,
                color=COLORS["tube"], alpha=0.3, linewidth=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left", fontsize=7)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig.3 已保存到 {output_path}")
```

---

## 图表 4：关节轨迹 + 力矩（Fig.4）

### 说明
展示单次成功击球的 6 个关节角度和力矩随时间变化。

### 数据来源
`experiment_data/exp2_strict_joint/raw/` 中的 NPZ（含 X_history, U_history）。

### 生成代码

```python
"""生成 Fig.4: 关节轨迹 + 力矩。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
})

JOINT_NAMES = ["J0 Shoulder Pan", "J1 Shoulder Lift", "J2 Elbow",
               "J3 Wrist 1", "J4 Wrist 2", "J5 Wrist 3"]
JOINT_COLORS = ["#0072B2", "#D55E00", "#009E73",
                "#56B4E9", "#E69F00", "#CC79A7"]


def generate_fig4(
    X: np.ndarray,
    U: np.ndarray,
    dt: float = 0.005,
    hit_step: int | None = None,
    output_path: str = "paper/figures/fig4_joint_trajectory.pdf",
) -> None:
    """生成关节轨迹和力矩图。

    Args:
        X: 状态轨迹 (N+1, 12)
        U: 控制轨迹 (N, 6)
        dt: 时间步长
        hit_step: 击球步数（画竖线）
        output_path: 输出路径
    """
    NQ = 6
    t = np.arange(len(X)) * dt * 1000  # ms

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    # 上: 关节角度
    ax_q = axes[0]
    for j in range(NQ):
        ax_q.plot(t, X[:, j] * 180 / np.pi, color=JOINT_COLORS[j],
                  linewidth=0.8, label=JOINT_NAMES[j])
    ax_q.set_ylabel("Joint angle (deg)")
    ax_q.legend(loc="upper left", ncol=3, fontsize=6)
    ax_q.grid(True, alpha=0.3)

    # 下: 控制力矩
    ax_u = axes[1]
    for j in range(NQ):
        ax_u.plot(t[:-1], U[:, j], color=JOINT_COLORS[j],
                  linewidth=0.8, label=JOINT_NAMES[j])
    ax_u.set_ylabel("Torque (Nm)")
    ax_u.set_xlabel("Time (ms)")
    ax_u.legend(loc="upper left", ncol=3, fontsize=6)
    ax_u.grid(True, alpha=0.3)

    # 击球时刻竖线
    if hit_step is not None:
        for ax in axes:
            ax.axvline(x=hit_step * dt * 1000, color="red",
                       linestyle="--", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig.4 已保存到 {output_path}")
```

---

## 图表 5：命中率 vs 球速（Fig.5）

### 说明
三条曲线对比：算法能力 / 严格关节约束 / TCP双限制。

### 数据来源
`experiment_data/exp1/results.csv` + `exp2/results.csv` + `exp3/results.csv`

### 生成代码

```python
"""生成 Fig.5: 命中率 vs 球速（三条件对比）。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
})


def generate_fig5(
    exp1_csv: str = "experiment_data/exp1_algorithm_capability/results.csv",
    exp2_csv: str = "experiment_data/exp2_strict_joint/results.csv",
    exp3_csv: str = "experiment_data/exp3_tcp_joint_dual/results.csv",
    output_path: str = "paper/figures/fig5_hit_rate_vs_speed.pdf",
) -> None:
    """生成命中率 vs 球速对比图。"""
    df1 = pd.read_csv(exp1_csv)
    df2 = pd.read_csv(exp2_csv)
    df3 = pd.read_csv(exp3_csv)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # (a) 命中率
    ax = axes[0]
    for df, label, color, marker in [
        (df1[df1["use_tube"] == True], "Algorithm capability", "#009E73", "o"),
        (df2[df2["use_tube"] == True], "Strict joint (RM-65B)", "#56B4E9", "s"),
        (df3[df3["max_tcp_limit"] == 1.8], "Joint + TCP ≤ 1.8m/s", "#E69F00", "^"),
    ]:
        grouped = df.groupby("ball_speed")["hit"].agg(["mean", "sem", "count"])
        ax.errorbar(grouped.index, grouped["mean"] * 100, yerr=grouped["sem"] * 100,
                    color=color, marker=marker, capsize=2, label=label, linewidth=1.0)
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=6, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) Hit rate", fontsize=9)

    # (b) 位置误差
    ax = axes[1]
    for df, label, color, marker in [
        (df1[df1["use_tube"] == True], "Algorithm capability", "#009E73", "o"),
        (df2[df2["use_tube"] == True], "Strict joint (RM-65B)", "#56B4E9", "s"),
        (df3[df3["max_tcp_limit"] == 1.8], "Joint + TCP ≤ 1.8m/s", "#E69F00", "^"),
    ]:
        grouped = df.groupby("ball_speed")["pos_error"].agg(["mean", "sem"])
        ax.errorbar(grouped.index, grouped["mean"] * 1000, yerr=grouped["sem"] * 1000,
                    color=color, marker=marker, capsize=2, label=label, linewidth=1.0)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Position error (mm)")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("(b) Position error", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig.5 已保存到 {output_path}")
```

---

## 图表 6：Tube 鲁棒性对比（Fig.6）

### 说明
时间/空间扰动下，Tube on vs off 的命中率变化。

### 数据来源
`experiment_data/exp4_tube_robustness/results.csv`

### 生成代码

```python
"""生成 Fig.6: Tube 鲁棒性对比。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
})


def generate_fig6(
    csv_path: str = "experiment_data/exp4_tube_robustness/results.csv",
    output_path: str = "paper/figures/fig6_tube_robustness.pdf",
) -> None:
    """生成 Tube 鲁棒性对比图。"""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # (a) 时间扰动
    ax = axes[0]
    df_time = df[df["perturb_type"] == "time"]
    for use_tube, color, label in [(True, "#0072B2", "Tube ON"), (False, "#D55E00", "Tube OFF")]:
        sub = df_time[df_time["use_tube"] == use_tube]
        grouped = sub.groupby("perturb_value")["hit"].agg(["mean", "sem"])
        ax.errorbar(grouped.index, grouped["mean"] * 100, yerr=grouped["sem"] * 100,
                    color=color, marker="o", capsize=2, label=label)
    ax.set_xlabel("Time perturbation (ms)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) Time perturbation", fontsize=9)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.5)

    # (b) 空间偏移
    ax = axes[1]
    df_space = df[df["perturb_type"] == "space"]
    for use_tube, color, label in [(True, "#0072B2", "Tube ON"), (False, "#D55E00", "Tube OFF")]:
        sub = df_space[df_space["use_tube"] == use_tube]
        grouped = sub.groupby("perturb_value")["hit"].agg(["mean", "sem"])
        ax.errorbar(grouped.index * 1000, grouped["mean"] * 100, yerr=grouped["sem"] * 100,
                    color=color, marker="s", capsize=2, label=label)
    ax.set_xlabel("Spatial perturbation (mm)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("(b) Spatial perturbation", fontsize=9)
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig.6 已保存到 {output_path}")
```

---

## 图表 7：实时性能（Fig.7）

### 说明
重规划耗时的箱线图，分 far/near 阶段，叠加预算线。

### 数据来源
`experiment_data/exp5_realtime_performance/results.csv`

### 生成代码

```python
"""生成 Fig.7: 实时性能。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
})


def generate_fig7(
    csv_path: str = "experiment_data/exp5_realtime_performance/results.csv",
    replan_interval: int = 10,
    dt: float = 0.005,
    output_path: str = "paper/figures/fig7_realtime_performance.pdf",
) -> None:
    """生成实时性能图。"""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    far_budget_ms = replan_interval * dt * 1000
    near_budget_ms = 2 * replan_interval * dt * 1000

    # (a) 重规划耗时分布
    ax = axes[0]
    data_far = df["avg_far_replan_ms"].dropna().values
    data_near = df["avg_near_replan_ms"].dropna().values
    bp = ax.boxplot([data_far, data_near], labels=["Far\n(k > 50)", "Near\n(k ≤ 50)"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#56B4E9")
    bp["boxes"][1].set_facecolor("#0072B2")
    ax.axhline(y=far_budget_ms, color="#D55E00", linestyle="--",
               linewidth=0.8, label=f"Far budget ({far_budget_ms:.0f}ms)")
    ax.axhline(y=near_budget_ms, color="#009E73", linestyle="--",
               linewidth=0.8, label=f"Near budget ({near_budget_ms:.0f}ms)")
    ax.set_ylabel("Replan time (ms)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) Replan time distribution", fontsize=9)

    # (b) 每步耗时分布
    ax = axes[1]
    step_data = df["avg_step_ms"].dropna().values
    ax.hist(step_data, bins=30, color="#0072B2", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(x=dt * 1000, color="#D55E00", linestyle="--",
               linewidth=0.8, label=f"dt = {dt*1000:.0f}ms")
    ax.set_xlabel("Step time (ms)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("(b) Per-step time", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig.7 已保存到 {output_path}")
```

---

## 图表 8：Tube 诊断六子图（Fig.8）

### 说明
单次成功击球的详细诊断数据。

### 数据来源
复用 `rm65_mpc_tube_constraint.py` 中已有的 `plot_tube_results` 函数，调整样式。

### 注意
代码中已有 `plot_tube_results` 函数（`scripts/rm65_mpc_tube_constraint.py:1140-1259`），
直接调用即可。如需 IEEE 风格，调整 `figsize` 和 `dpi` 参数。

---

## LaTeX 表格生成

### Table I: 定量对比表

```python
"""生成 Table I: 不同约束条件下的性能对比。"""
import pandas as pd


def generate_table1(
    exp1_csv: str,
    exp2_csv: str,
    exp3_csv: str,
    output_path: str = "paper/figures/table_data/table1_comparison.tex",
) -> None:
    """生成定量对比 LaTeX 表格。"""
    df1 = pd.read_csv(exp1_csv)
    df2 = pd.read_csv(exp2_csv)
    df3 = pd.read_csv(exp3_csv)

    rows = []
    for label, df in [
        ("Algorithm capability\n(speed exemption)", df1[df1["use_tube"] == True]),
        ("Strict joint\n(RM-65B, qdot $\\leq$ 1.0x)", df2[df2["use_tube"] == True]),
        ("Joint + TCP\n($v_{tcp} \\leq$ 1.8 m/s)", df3[df3["max_tcp_limit"] == 1.8]),
    ]:
        for speed in sorted(df["ball_speed"].unique()):
            sub = df[df["ball_speed"] == speed]
            hit_rate = sub["hit"].mean() * 100
            pos_err = sub["pos_error"].mean() * 1000
            min_dist = sub["min_distance"].mean() * 1000
            n = len(sub)
            rows.append({
                "condition": label,
                "ball_speed": speed,
                "hit_rate": f"{hit_rate:.0f}\\%",
                "pos_error_mm": f"{pos_err:.1f}",
                "min_dist_mm": f"{min_dist:.1f}",
                "n": n,
            })

    df_out = pd.DataFrame(rows)

    latex = r"\begin{table}[t]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Performance comparison under different constraints.}" + "\n"
    latex += r"\label{tab:comparison}" + "\n"
    latex += r"\begin{tabular}{lcrrr}" + "\n"
    latex += r"\toprule" + "\n"
    latex += r"Condition & Ball speed & Hit rate & Pos. error & Min. dist. \\" + "\n"
    latex += r" & (m/s) & (\%) & (mm) & (mm) \\" + "\n"
    latex += r"\midrule" + "\n"

    prev_cond = ""
    for _, row in df_out.iterrows():
        cond = row["condition"] if row["condition"] != prev_cond else ""
        latex += f"  {cond} & {row['ball_speed']:.0f} & {row['hit_rate']} & {row['pos_error_mm']} & {row['min_dist_mm']} \\\\\\\\\n"
        prev_cond = row["condition"]

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Table I 已保存到 {output_path}")
```

### Table II: 消融实验表

```python
"""生成 Table II: 消融实验。"""


def generate_table2(
    exp6_csv: str,
    output_path: str = "paper/figures/table_data/table2_ablation.tex",
) -> None:
    """生成消融实验 LaTeX 表格。"""
    import pandas as pd
    df = pd.read_csv(exp6_csv)

    latex = r"\begin{table}[t]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Ablation study results (ball speed = 9 m/s, 20 seeds).}" + "\n"
    latex += r"\label{tab:ablation}" + "\n"
    latex += r"\begin{tabular}{lcrr}" + "\n"
    latex += r"\toprule" + "\n"
    latex += r"Configuration & Value & Hit rate & Pos. error \\" + "\n"
    latex += r" & & (\%) & (mm) \\" + "\n"
    latex += r"\midrule" + "\n"

    for var in df["ablation_var"].unique():
        sub = df[df["ablation_var"] == var]
        for val in sub["ablation_value"].unique():
            subsub = sub[sub["ablation_value"] == val]
            hit_rate = subsub["hit"].mean() * 100
            pos_err = subsub["pos_error"].mean() * 1000
            latex += f"  {var} & {val} & {hit_rate:.0f}\\% & {pos_err:.1f} \\\\\\\\\n"
        latex += r"\midrule" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Table II 已保存到 {output_path}")
```

---

## 图表生成顺序

按依赖关系排列的生成顺序：

1. **先运行实验** → 数据收集到 `experiment_data/`
2. **提取 CSV** → `python scripts/extract_experiment_results.py --exp-dir ...`
3. **生成图表**（可并行）：
   - Fig.1: MuJoCo 截图（需要 viewer 或离屏渲染）
   - Fig.2: TikZ 流程图（手动绘制）
   - Fig.3: Tube 走廊（需要 NPZ 原始数据）
   - Fig.4: 关节轨迹（需要 NPZ 原始数据）
   - Fig.5: 命中率对比（需要 exp1+exp2+exp3 CSV）
   - Fig.6: Tube 鲁棒性（需要 exp4 CSV）
   - Fig.7: 实时性能（需要 exp5 CSV）
   - Fig.8: Tube 诊断（复用已有代码）
   - Table I: 定量对比（需要 exp1+exp2+exp3 CSV）
   - Table II: 消融（需要 exp6 CSV）
