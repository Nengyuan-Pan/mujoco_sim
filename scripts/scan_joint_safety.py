"""关节安全范围扫描脚本 — 确定保守关节限制，确保右臂不越过身体中线。

两种模式:
  1. 单关节扫描（默认）: 每个关节独立遍历，记录臂体X曲线
  2. 蒙特卡洛全局扫描 (--monte-carlo): 随机采样 N 个 6D 构型，输出保守 joint_limits

用法:
  python scripts/scan_joint_safety.py                         # 单关节扫描
  python scripts/scan_joint_safety.py --monte-carlo           # 全局扫描 (100k点)
  python scripts/scan_joint_safety.py --monte-carlo --mc-samples 500000 --limit-x 0.0
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.mujoco_loader import load_mujoco_model
import mujoco

# 需要监控的臂体
CHECK_BODIES = ["r_link3", "r_link5", "r_racket_body"]
BODY_LABELS = {"r_link3": "elbow (r_link3)", "r_link5": "wrist (r_link5)", "r_racket_body": "racket (r_racket_body)"}

# 关节名称
JOINT_NAMES = {
    0: "r_joint1 (shoulder_pan)",
    1: "r_joint2 (shoulder_lift)",
    2: "r_joint3 (elbow)",
    3: "r_joint4 (wrist_1)",
    4: "r_joint5 (wrist_2)",
    5: "r_joint6 (wrist_3)",
}

# 初始右臂关节角度（击球准备姿势）
INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)


def scan_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_idx: int,
    deg_range: tuple[float, float],
    step_deg: float,
    init_q: np.ndarray,
    body_ids: dict[str, int],
) -> dict[str, np.ndarray]:
    """扫描单个关节：遍历角度范围，记录各 body 的 X 坐标。

    Args:
        model: MuJoCo 模型。
        data: MuJoCo 数据。
        joint_idx: 关节索引 (0-5)。
        deg_range: (start_deg, end_deg)，MuJoCo range 值。
        step_deg: 扫描步长 (度)。
        init_q: 初始关节角度 (rad)。
        body_ids: {body_name: body_id} 映射。

    Returns:
        {
            "angles_deg": 角度数组 (N,),
            "r_link3": X 坐标数组 (N,),
            "r_link5": X 坐标数组 (N,),
            "r_racket_body": X 坐标数组 (N,),
        }
    """
    start_deg, end_deg = deg_range
    n_steps = int(abs(end_deg - start_deg) / step_deg) + 1
    angles_deg = np.linspace(start_deg, end_deg, n_steps)
    angles_rad = np.deg2rad(angles_deg)

    result = {name: np.zeros(n_steps) for name in CHECK_BODIES}
    result["angles_deg"] = angles_deg

    q = init_q.copy()
    for i, a_rad in enumerate(angles_rad):
        q[joint_idx] = a_rad
        data.qpos[:6] = q
        data.qpos[6:12] = 0.0  # 左臂归零
        mujoco.mj_forward(model, data)
        for name in CHECK_BODIES:
            result[name][i] = data.xpos[body_ids[name], 0]

    return result


def find_safe_limit(
    angles_deg: np.ndarray,
    x_curves: dict[str, np.ndarray],
    limit_x: float,
) -> tuple[float, float]:
    """找角度范围的安全上下限。

    安全 = 所有 body 的 X ≤ limit_x。

    Returns:
        (deg_lo, deg_hi): 安全范围的中心位置。返回 None 表示无限制。
    """
    # 检查所有 body 在每步的 X 是否 ≤ limit_x
    max_x = np.maximum(np.maximum(x_curves["r_link3"], x_curves["r_link5"]), x_curves["r_racket_body"])
    safe = max_x <= limit_x

    # 找连续安全区间
    safe_start = None
    safe_end = None
    best_len = 0
    current_start = None

    for i in range(len(safe)):
        if safe[i]:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length > best_len:
                    best_len = length
                    safe_start = current_start
                    safe_end = i - 1
                current_start = None

    if current_start is not None:
        length = len(safe) - current_start
        if length > best_len:
            safe_start = current_start
            safe_end = len(safe) - 1

    if safe_start is None or safe_end is None:
        return float("nan"), float("nan")

    return angles_deg[safe_start], angles_deg[safe_end]


def print_report(results: dict[int, dict], limit_x: float, model: mujoco.MjModel) -> None:
    """打印安全范围报告。"""
    print(f"\n{'='*75}")
    print(f"  关节安全范围扫描报告 (limit_x = {limit_x}m)")
    print(f"  初始 pose: q = {np.round(INIT_Q, 2)} rad")
    print(f"{'='*75}")
    print(f"  {'关节':<28s} {'MuJoCo range':>18s} {'安全上限(deg)':>14s} {'推荐上限(deg)':>14s}")
    print(f"  {'-'*73}")

    for j in range(6):
        r = results[j]
        angles = r["angles_deg"]
        max_x = np.maximum(np.maximum(r["r_link3"], r["r_link5"]), r["r_racket_body"])

        # 找到 max_x 首次超过 limit_x 的角度
        first_violation = None
        for i in range(len(angles)):
            if max_x[i] > limit_x:
                first_violation = angles[i]
                break

        safe_lo, safe_hi = find_safe_limit(angles, r, limit_x)

        mj_range_lo = float("-inf")
        mj_range_hi = float("inf")
        for mj in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, mj)
            if name == f"r_joint{j+1}":
                mj_range_lo = float(model.jnt_range[mj, 0])
                mj_range_hi = float(model.jnt_range[mj, 1])
                break

        mj_range_str = f"[{np.rad2deg(mj_range_lo):.0f}, {np.rad2deg(mj_range_hi):.0f}]"
        safe_str = f"{safe_hi:.0f}" if np.isfinite(safe_hi) else "no limit"
        recommended = f"{safe_hi:.0f}" if np.isfinite(safe_hi) else "full range"
        if first_violation is not None:
            recommended = f"{min(first_violation - 5, safe_hi):.0f}" if np.isfinite(safe_hi) else "full"

        print(f"  {JOINT_NAMES[j]:<28s} {mj_range_str:>18s} {safe_str:>14s} {recommended:>14s}")
        print(f"    首次撞墙 @ {first_violation:.0f}°" if first_violation is not None else f"    全程安全")

    print(f"{'='*75}")
    print()
    print("  建议 joint_limits 配置（可直接复制到 rm65_mpc_tube_constraint.py）:")
    print()
    print("  SAFE_JOINT_LIMITS_DEG = {")
    for j in range(6):
        r = results[j]
        angles = r["angles_deg"]
        max_x = np.maximum(np.maximum(r["r_link3"], r["r_link5"]), r["r_racket_body"])

        violations = [(angles[i], max_x[i]) for i in range(len(angles)) if max_x[i] > limit_x]
        if violations:
            first_violation_deg = violations[0][0]
            print(f"      {j}: (None, {first_violation_deg - 5:.0f}),   # {JOINT_NAMES[j]}")
        else:
            # 检查是否有下限危险
            print(f"      {j}: (None, None),   # {JOINT_NAMES[j]} — 无碰撞风险")
    print("  }")
    print()


def plot_scan(results: dict[int, dict], limit_x: float, out_dir: str = "results") -> None:
    """绘制扫描热力图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Joint vs Arm Body X Position (safe when X <= {limit_x}m)", fontsize=13)

    for j in range(6):
        ax = axes[j // 3, j % 3]
        r = results[j]
        angles = r["angles_deg"]
        colors = {"r_link3": "blue", "r_link5": "green", "r_racket_body": "red"}
        labels = {"r_link3": "elbow", "r_link5": "wrist", "r_racket_body": "racket"}

        for name in CHECK_BODIES:
            ax.plot(angles, r[name], color=colors[name], label=labels[name], linewidth=1.5)

        ax.axhline(y=limit_x, color="black", linestyle="--", linewidth=1, label=f"limit X={limit_x}m")
        ax.set_xlabel("Joint Angle (deg)")
        ax.set_ylabel("Body X (m)")
        ax.set_title(JOINT_NAMES[j])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "joint_safety_scan.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def monte_carlo_scan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_samples: int,
    limit_x: float,
    body_ids: dict[str, int],
    rng: np.random.Generator,
) -> dict:
    """蒙特卡洛全局扫描：随机采样 N 个 6D 构型，找出保守安全范围。

    对每个采样点，计算臂体 X 坐标。若全部安全（X ≤ limit_x），
    记录该构型的关节角度。最终输出所有安全样本的各关节 min/max。

    Returns:
        {
            "safe_ratio": 安全样本占比,
            "joint_lo": 各关节最小值 (6,),
            "joint_hi": 各关节最大值 (6,),
            "safe_lo": 安全样本中各关节最小值 (6,),
            "safe_hi": 安全样本中各关节最大值 (6,),
            "worst_x": 所有样本中最大臂体X值,
        }
    """
    n_joints = 6
    safe_q = np.zeros((n_samples, n_joints))
    n_safe = 0
    worst_x = -float("inf")

    # 获取各关节的 MuJoCo range
    joint_ranges = np.zeros((n_joints, 2))
    for j in range(n_joints):
        for mj in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, mj)
            if name == f"r_joint{j+1}":
                joint_ranges[j, 0] = float(model.jnt_range[mj, 0])
                joint_ranges[j, 1] = float(model.jnt_range[mj, 1])
                break

    t_start = time.perf_counter()
    for i in range(n_samples):
        # 在 MuJoCo range 内均匀采样
        q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        data.qpos[:6] = q
        data.qpos[6:12] = 0.0
        mujoco.mj_forward(model, data)

        max_x = max(data.xpos[body_ids[n], 0] for n in CHECK_BODIES)
        if max_x > worst_x:
            worst_x = max_x

        if max_x <= limit_x:
            if n_safe < n_samples:
                safe_q[n_safe] = q
            n_safe += 1

        if (i + 1) % (n_samples // 10) == 0:
            print(f"  sampled {i+1}/{n_samples}, safe={n_safe} ({100*n_safe/(i+1):.1f}%)")

    t_elapsed = time.perf_counter() - t_start
    safe_q = safe_q[:n_safe]

    result = {
        "safe_ratio": n_safe / n_samples,
        "joint_lo": joint_ranges[:, 0],
        "joint_hi": joint_ranges[:, 1],
        "safe_lo": safe_q.min(axis=0) if n_safe > 0 else np.zeros(n_joints),
        "safe_hi": safe_q.max(axis=0) if n_safe > 0 else np.zeros(n_joints),
        "worst_x": worst_x,
        "n_safe": n_safe,
        "n_total": n_samples,
        "time_s": t_elapsed,
    }
    return result


def print_mc_report(result: dict, limit_x: float, safety_margin_deg: float = 8.0) -> None:
    """打印蒙特卡洛扫描报告 + 推荐 joint_limits 配置。"""
    safe_ratio = result["safe_ratio"]
    worst_x = result["worst_x"]
    margin_rad = np.deg2rad(safety_margin_deg)

    print(f"\n{'='*80}")
    print(f"  蒙特卡洛全局安全扫描报告 (limit_x={limit_x}m, margin={safety_margin_deg}°)")
    print(f"  样本: {result['n_total']}, 安全: {result['n_safe']} ({safe_ratio*100:.2f}%)")
    print(f"  耗时: {result['time_s']:.1f}s, 最坏X: {worst_x:.3f}m")
    print(f"{'='*80}")
    print(f"  {'关节':<28s} {'MuJoCo range':>20s} {'安全范围(margin后)':>26s}")
    print(f"  {'-'*78}")

    for j in range(6):
        mj_lo = np.rad2deg(result["joint_lo"][j])
        mj_hi = np.rad2deg(result["joint_hi"][j])
        safe_lo = np.rad2deg(result["safe_lo"][j])
        safe_hi = np.rad2deg(result["safe_hi"][j])
        # 加 margin
        rec_lo = safe_lo + safety_margin_deg
        rec_hi = safe_hi - safety_margin_deg

        mj_str = f"[{mj_lo:.0f}, {mj_hi:.0f}]"
        if rec_lo < rec_hi and safe_ratio > 0.001:
            safe_str = f"[{rec_lo:.0f}, {rec_hi:.0f}]"
        else:
            safe_str = "full range"
        print(f"  {JOINT_NAMES[j]:<28s} {mj_str:>20s} {safe_str:>26s}")

    print(f"{'='*80}")
    print()
    print("  # 可直接复制到 rm65_mpc_tube_constraint.py 的安全关节限制:")
    print()
    print("  CONSERVATIVE_JOINT_LIMITS_DEG = {")
    for j in range(6):
        safe_lo = np.rad2deg(result["safe_lo"][j])
        safe_hi = np.rad2deg(result["safe_hi"][j])
        rec_lo = safe_lo + safety_margin_deg
        rec_hi = safe_hi - safety_margin_deg
        if rec_lo < rec_hi and safe_ratio > 0.001:
            print(f"      {j}: ({rec_lo:.0f}, {rec_hi:.0f}),   # {JOINT_NAMES[j]}")
        else:
            print(f"      {j}: (None, None),   # {JOINT_NAMES[j]} — 无可行安全范围")
    print("  }")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="关节安全范围扫描")
    parser.add_argument("--monte-carlo", action="store_true", help="蒙特卡洛全局扫描模式")
    parser.add_argument("--mc-samples", type=int, default=100000, help="蒙特卡洛采样数")
    parser.add_argument("--mc-margin-deg", type=float, default=8.0, help="安全 margin (度)")
    parser.add_argument("--limit-x", type=float, default=0.0, help="X 墙上限 (m)")
    parser.add_argument("--step-deg", type=float, default=2.0, help="单关节扫描步长 (度)")
    parser.add_argument("--no-plot", action="store_true", help="不绘图")
    parser.add_argument("--out", type=str, default="results", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    model = load_mujoco_model(model_path)
    data = mujoco.MjData(model)

    # 获取 body IDs
    body_ids: dict[str, int] = {}
    for name in CHECK_BODIES:
        body_ids[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    if args.monte_carlo:
        # 蒙特卡洛全局扫描
        print(f"Monte Carlo scanning: {args.mc_samples} samples, limit_x={args.limit_x}m...")
        rng = np.random.default_rng(args.seed)
        result = monte_carlo_scan(model, data, args.mc_samples, args.limit_x, body_ids, rng)
        print_mc_report(result, args.limit_x, args.mc_margin_deg)
    else:
        # 单关节扫描（原有逻辑）
        results: dict[int, dict] = {}
        for j in range(6):
            mj_range = None
            for mj in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, mj)
                if name == f"r_joint{j+1}":
                    mj_range = model.jnt_range[mj]
                    break
            if mj_range is None:
                print(f"Warning: joint r_joint{j+1} not found")
                continue
            deg_range = (float(np.rad2deg(mj_range[0])), float(np.rad2deg(mj_range[1])))
            print(f"Scanning {JOINT_NAMES[j]}: {deg_range[0]:.0f}° to {deg_range[1]:.0f}°...")
            r = scan_joint(model, data, j, deg_range, args.step_deg, INIT_Q, body_ids)
            results[j] = r
        print_report(results, args.limit_x, model)
        if not args.no_plot:
            plot_scan(results, args.limit_x, args.out)


if __name__ == "__main__":
    main()
