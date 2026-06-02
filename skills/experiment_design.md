# Skill: 实验设计与数据管理（experiment_design）

## 目的
围绕 MPC+iLQR+Tube 网球击打框架，系统化设计、执行和管理论文实验数据。
调用时机：运行批量实验、收集论文数据、分析实验结果时。

## 核心脚本映射

本项目的核心算法实现在以下三个脚本中：

| 脚本 | 用途 | 关键特性 |
|------|------|---------|
| `scripts/rm65_mpc_tube_constraint.py` | 离线仿真主脚本 | MPC+iLQR+Tube+硬约束+X平面墙 |
| `scripts/rm65_mpc_tube_constraint_realtime.py` | 实时仿真脚本 | 异步重规划+buffer机制+实时节奏 |
| `scripts/run_tcp_limit_experiment.py` | TCP限速实验 | monkey-patch安全滤波器注入TCP检查 |

## 数据存储规范

### 根目录
```
experiment_data/
├── README.md
├── exp1_algorithm_capability/
│   ├── config.yaml          # 实验参数（自动生成）
│   ├── results.csv          # 汇总表
│   ├── raw/                 # 原始 .npy 数据
│   │   ├── seed0_ball_speed9.npz
│   │   └── ...
│   └── figures/             # 该实验的调试图
├── exp2_strict_joint/
├── exp3_tcp_joint_dual/
├── exp4_tube_robustness/
├── exp5_realtime_performance/
└── exp6_ablation/
```

### CSV 格式标准

所有 results.csv 必须包含以下通用列：

```csv
seed,ball_speed,hit,pos_error,min_distance,max_qdot_ratio,max_tcp_speed,mpc_steps,total_time_s,hit_time_error_ms,tube_ready_ms,ball_near_ms
0,9.0,True,0.035,0.035,2.75,3.9,149,2.1,5.2,45.0,25.0
```

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| seed | int | — | 随机种子 |
| ball_speed | float | m/s | 球到达击打点时的水平速度 |
| hit | bool | — | 命中判定（pos_error < 0.05 且 min_distance < 0.153） |
| pos_error | float | m | 末端位置误差 ‖p_ee - p_hit‖ |
| min_distance | float | m | 全程最小球拍-球距离 |
| max_qdot_ratio | float | × | 最大关节速度 / 额定限速 |
| max_tcp_speed | float | m/s | 最大 TCP 线速度 |
| mpc_steps | int | — | MPC 执行步数 |
| total_time_s | float | s | 总墙钟时间 |
| hit_time_error_ms | float | ms | 击打时间误差 |
| tube_ready_ms | float | ms | tube_ready 持续时间 |
| ball_near_ms | float | ms | ball_near 持续时间 |

实验特有列在每组实验的 config.yaml 中定义。

### NPZ 原始数据格式

每次运行的原始数据保存为 NPZ（可选，用于调试和生成额外图表）：

```python
np.savez_compressed(
    f"raw/seed{seed}_ballspeed{ball_speed}.npz",
    X_history=np.array(X_history),       # (N+1, 12)
    U_history=np.array(U_history),       # (N, 6)
    ball_pos_history=np.array(ball_pos), # (M, 3)
    distances=np.array(distances_history),
    normal_align=np.array(normal_align_history),
    step_times=np.array(step_times),
    replan_times=np.array(replan_times),
)
```

---

## 实验 1：算法能力上限（速度豁免模式）

### 目的
验证 iLQR+Tube 在理想高速执行器条件下的击打能力上限，不受真实关节速度限制。

### 脚本
`scripts/rm65_mpc_tube_constraint.py`

### 参数矩阵

```yaml
# exp1_algorithm_capability/config.yaml
experiment: exp1_algorithm_capability
description: "算法能力上限，速度豁免模式"
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_seeds: 20

# 变量1: 球速
ball_speeds: [9, 12, 15, 18, 20, 25, 30]

# 变量2: Tube 开关
tube_conditions:
  - use_tube: true
  - use_tube: false

# 固定参数
fixed_params:
  serve_box: true
  no_backswing: true
  no_plot: true
  fd: false
  horizon: 120
  iter: 10
  replan_interval: 10
  window_ms: 50.0
  sigma0: 0.02
  tube_cost_ratio: 0.3
  normal_weight: 500000.0

# 速度豁免：forward_pass_margin=3.0（默认），不限制关节速度
speed_exemption:
  forward_pass_margin: 3.0
  qdot_scale: 0.95  # 使用默认值，不做严格限制
  # 击球前100ms豁免关节速度检查（代码内置行为）

# 总运行数 = 7 ball_speeds × 2 tube_conditions × 20 seeds = 280 runs
```

### 运行命令模板

```bash
# 单次运行示例
python scripts/rm65_mpc_tube_constraint.py \
    --serve-box --ball-speed 12 --seed 0 \
    --use_tube true --no-backswing --no-plot \
    --horizon 120 --iter 10 --replan-interval 10

# 批量运行脚本（PowerShell）
$ballSpeeds = @(9, 12, 15, 18, 20, 25, 30)
$seeds = 0..19
$tubeModes = @("true", "false")

foreach ($speed in $ballSpeeds) {
    foreach ($tube in $tubeModes) {
        foreach ($seed in $seeds) {
            $tag = "speed${speed}_tube${tube}_seed${seed}"
            Write-Host "Running: $tag"
            python scripts/rm65_mpc_tube_constraint.py `
                --serve-box --ball-speed $speed --seed $seed `
                --use_tube $tube --no-backswing --no-plot `
                --horizon 120 --iter 10 --replan-interval 10 `
                2>&1 | Tee-Object -FilePath "experiment_data/exp1_algorithm_capability/raw/${tag}.log"
        }
    }
}
```

### 结果提取

从脚本标准输出中提取以下指标（用正则匹配）：

```
位置误差: (\d+\.\d+) m
速度误差: (\d+\.\d+) m/s
最小球拍-球距离: (\d+\.\d+) m
ball_near 步数: (\d+) = (\d+\.\d+) ms
tube_ready 步数: (\d+) = (\d+\.\d+) ms
max_qdot=(\d+\.\d+)x, max_tcp=(\d+\.\d+)m/s
```

### 额外 CSV 列
| 列名 | 说明 |
|------|------|
| use_tube | bool，是否启用 Tube |
| max_qdot_joint | int，最大速度超限的关节编号 |
| hit_type | str，"主动击球"/"被动接触"/"未触球" |

---

## 实验 2：真实关节约束可行性

### 目的
在 RM-65B 真实关节约束下（qdot ≤ 1.0×，全程无豁免），测试系统可行球速范围。

### 脚本
`scripts/rm65_mpc_tube_constraint.py`

### 参数矩阵

```yaml
# exp2_strict_joint/config.yaml
experiment: exp2_strict_joint
description: "真实机械臂严格关节约束"
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_seeds: 20

ball_speeds: [7, 8, 9, 10, 11, 12]

tube_conditions:
  - use_tube: true
  - use_tube: false

fixed_params:
  serve_box: true
  no_backswing: true
  no_plot: true
  fd: false
  horizon: 120
  iter: 10
  replan_interval: 10
  window_ms: 50.0
  tube_cost_ratio: 0.3

# 严格约束：通过 monkey-patch 或配置实现
strict_constraints:
  forward_pass_margin: 1.0    # 严格：不允许搜索超出约束
  qdot_scale: 1.0             # 严格：不缩放，使用完整限速
  forward_pass_q_tol_deg: 0.0 # 严格：无额外容忍
  # 击球前 100ms 豁免窗口仍然存在（与实验1一致）
  # 如需完全无豁免，需修改 strict_braking_check

# 总运行数 = 6 ball_speeds × 2 tube × 20 seeds = 240 runs
```

### 运行命令

需要 monkey-patch 严格参数。创建辅助脚本 `scripts/_run_exp2_strict.py`：

```python
"""实验2辅助脚本：注入严格关节约束参数。"""
import sys
from pathlib import Path

sys.argv = [
    "rm65_mpc_tube_constraint.py",
    "--serve-box",
    "--ball-speed", sys.argv[1],
    "--seed", sys.argv[2],
    "--use-tube", sys.argv[3],
    "--no-backswing", "--no-plot",
]

import scripts.rm65_mpc_tube_constraint as main_mod
from src.ilqt.robot_limits import RobotLimits

_orig = RobotLimits.from_config
@classmethod
def _strict(cls, config, dt, ctrlrange):
    config = dict(config)
    config["forward_pass_margin"] = 1.0
    config["qdot_scale"] = 1.0
    config["forward_pass_q_tol_deg"] = 0.0
    return _orig(config, dt, ctrlrange)
RobotLimits.from_config = _strict

main_mod.main()
```

```bash
python scripts/_run_exp2_strict.py 12 0 true
```

---

## 实验 3：TCP + 关节双约束

### 目的
在关节约束基础上增加 TCP 线速度硬限制，测试 TCP 约束对性能的影响。

### 脚本
`scripts/run_tcp_limit_experiment.py`

### 参数矩阵

```yaml
# exp3_tcp_joint_dual/config.yaml
experiment: exp3_tcp_joint_dual
description: "TCP线速度+关节速度双重硬约束"
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_seeds: 20

ball_speeds: [7, 8, 9, 10]

tcp_limits: [1.0, 1.5, 1.8, 2.0, 2.5]

fixed_params:
  no_backswing: true
  no_plot: true
  realtime: true

# run_tcp_limit_experiment.py 内部已 monkey-patch:
# - strict_braking_check 全程无豁免
# - qdot_scale=1.0, forward_pass_margin=1.0
# - TCP 速度硬限制注入到安全滤波器

# 总运行数 = 4 ball_speeds × 5 tcp_limits × 20 seeds = 400 runs
```

### 运行命令

```bash
python scripts/run_tcp_limit_experiment.py `
    --ball-speed 9 --seed 0 --max-tcp 1.8
```

### 额外 CSV 列
| 列名 | 说明 |
|------|------|
| max_tcp_limit | float，TCP 速度硬限制值 (m/s) |
| actual_max_tcp | float，实际最大 TCP 速度（应 ≈ max_tcp_limit） |

---

## 实验 4：Tube 鲁棒性（时间/空间扰动）

### 目的
测试 Tube 机制在球轨迹预测存在时间误差和空间偏移时的鲁棒性。

### 脚本
`scripts/rm65_mpc_tube_constraint.py`

### 参数矩阵

```yaml
# exp4_tube_robustness/config.yaml
experiment: exp4_tube_robustness
description: "Tube鲁棒性：时间预测扰动 + 击打点空间偏移"
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_seeds: 20

# 子实验 4a: 时间扰动
time_perturbs_ms: [-100, -50, -20, 0, 20, 50, 100]

# 子实验 4b: 空间偏移
space_perturbs_m: [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10]

fixed_params:
  serve_box: true
  ball_speed: 9           # 固定球速 9 m/s（RM-65B 可行范围内）
  use_tube: true
  no_backswing: true
  no_plot: true
  horizon: 120
  iter: 10
  replan_interval: 10

# 严格关节约束（与实验2一致）
strict_constraints:
  forward_pass_margin: 1.0
  qdot_scale: 1.0

# 运行数 = (7 时间 + 7 空间) × 20 seeds = 280 runs
```

### 运行命令

```bash
# 4a: 时间扰动
python scripts/rm65_mpc_tube_constraint.py `
    --serve-box --ball-speed 9 --seed 0 `
    --use_tube true --no-backswing --no-plot `
    --time-perturb-ms 50

# 4b: 空间偏移
python scripts/rm65_mpc_tube_constraint.py `
    --serve-box --ball-speed 9 --seed 0 `
    --use_tube true --no-backswing --no-plot `
    --space-perturb-m 0.06
```

### 额外 CSV 列
| 列名 | 说明 |
|------|------|
| perturb_type | str，"time" 或 "space" |
| perturb_value | float，扰动量（ms 或 m） |
| use_tube | bool |

### 对照组
同时运行 `--use_tube false` 的同等扰动，对比 Tube on/off 的鲁棒性差异。

---

## 实验 5：实时性能分析

### 目的
收集 MPC 重规划的时间开销数据，验证实时可行性。

### 脚本
`scripts/rm65_mpc_tube_constraint_realtime.py`

### 参数矩阵

```yaml
# exp5_realtime_performance/config.yaml
experiment: exp5_realtime_performance
description: "实时性能分析"
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
n_seeds: 20

ball_speeds: [9, 12]

fixed_params:
  serve_box: true
  use_tube: true
  no_backswing: true
  no_plot: true
  realtime: true

# 总运行数 = 2 ball_speeds × 20 seeds = 40 runs
```

### 需提取的实时性指标

从输出日志中提取：

```
# 重规划性能
首次规划(冷启动): XXXXms
稳态重规划(N次): avg=XXms, max=XXms
far阶段: avg=XXms, max=XXms / 预算=100ms
near阶段: avg=XXms, max=XXms / 预算=200ms

# 逐步执行
MPC 每步平均: X.Xms
非重规划步: avg=X.Xms
真实机器人预估: X.Xms

# MPC 实时比率
MPC 实时比率: X.XXx
```

### 额外 CSV 列
| 列名 | 说明 |
|------|------|
| first_replan_ms | float，首次规划耗时 (ms) |
| avg_steady_replan_ms | float，稳态重规划平均耗时 (ms) |
| max_steady_replan_ms | float，稳态重规划最大耗时 (ms) |
| avg_far_replan_ms | float，far 阶段平均耗时 (ms) |
| max_far_replan_ms | float，far 阶段最大耗时 (ms) |
| avg_near_replan_ms | float，near 阶段平均耗时 (ms) |
| max_near_replan_ms | float，near 阶段最大耗时 (ms) |
| far_over_budget | bool，far 阶段是否超预算 |
| near_over_budget | bool，near 阶段是否超预算 |
| avg_step_ms | float，每步平均耗时 (ms) |
| mpc_realtime_ratio | float，MPC 实时比率 |
| n_replans | int，重规划总次数 |
| buffer_exhaust_count | int，buffer 耗尽次数 |

---

## 实验 6：消融实验

### 目的
验证各组件对系统性能的独立贡献。

### 脚本
`scripts/rm65_mpc_tube_constraint.py`

### 6a: Tube 开关

```yaml
tube_conditions:
  - use_tube: true, window_ms: 50, tube_cost_ratio: 0.3
  - use_tube: false
ball_speeds: [9, 12]
seeds: 0..19
# 固定: serve_box, no_backswing, strict_joint
```

### 6b: Tube 窗口大小

```yaml
window_ms_values: [20, 30, 50, 80, 100]
ball_speeds: [9]
tube_cost_ratio: 0.3
seeds: 0..19
```

### 6c: Tube 代价占比

```yaml
tube_cost_ratio_values: [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
ball_speeds: [9]
window_ms: 50
seeds: 0..19
```

### 6d: 后摆策略

```yaml
backswing_conditions:
  - backswing: 0.6, bs_ratio: 0.35  # 启用后摆
  - no_backswing: true                # 禁用后摆
ball_speeds: [9, 12]
seeds: 0..19
```

### 6e: R 退火

```yaml
r_decay_conditions:
  - r_decay: 0.40      # 启用 R 退火
  - no_r_decay: true    # 禁用 R 退火
ball_speeds: [9, 12]
seeds: 0..19
```

### 额外 CSV 列
| 列名 | 说明 |
|------|------|
| ablation_var | str，消融变量名（tube/window/ratio/backswing/r_decay） |
| ablation_value | str，变量取值 |

---

## 结果提取脚本模板

创建 `scripts/extract_experiment_results.py` 用于自动从日志提取数据到 CSV：

```python
"""从实验日志批量提取结果到 CSV。

用法:
    python scripts/extract_experiment_results.py --exp-dir experiment_data/exp1_algorithm_capability
"""
import re
import csv
import argparse
from pathlib import Path


PATTERNS = {
    "pos_error": r"位置误差:\s+([\d.]+)\s+m",
    "vel_error": r"速度误差:\s+([\d.]+)\s+m/s",
    "min_distance": r"最小球拍-球距离:\s+([\d.]+)\s+m",
    "ball_near_ms": r"ball_near 步数:\s+\d+\s+=\s+([\d.]+)\s+ms",
    "tube_ready_ms": r"tube_ready 步数:\s+\d+\s+=\s+([\d.]+)\s+ms",
    "max_qdot_tcp": r"max_qdot=([\d.]+)x,\s+max_tcp=([\d.]+)m/s",
    "hit_type": r"击球类型:\s+(.+)",
}


def parse_log(log_path: Path) -> dict:
    """从单个日志文件提取指标。"""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    result = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            if key == "max_qdot_tcp":
                result["max_qdot_ratio"] = float(m.group(1))
                result["max_tcp_speed"] = float(m.group(2))
            else:
                result[key] = m.group(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    log_files = sorted(exp_dir.glob("raw/*.log"))
    if not log_files:
        print(f"未找到日志文件: {exp_dir / 'raw/*.log'}")
        return

    rows = []
    for lf in log_files:
        row = parse_log(lf)
        row["log_file"] = lf.name
        rows.append(row)

    csv_path = exp_dir / "results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已提取 {len(rows)} 条结果到 {csv_path}")
    else:
        print("未提取到任何结果")


if __name__ == "__main__":
    main()
```

---

## 数据完整性检查清单

每次实验运行后检查：

- [ ] 每个 (ball_speed, seed, condition) 组合都有对应的日志文件
- [ ] 日志中包含完整的评估输出（"========" 分隔符之间的内容）
- [ ] pos_error 为有效数值（非 NaN、非 Inf）
- [ ] max_qdot_ratio 在合理范围内（实验1: 0~5.0，实验2/3: 0~1.2）
- [ ] 无异常退出（日志末尾有 "========" 结束标记）
- [ ] CSV 行数 = 预期运行数（n_speeds × n_conditions × n_seeds）

---

## 已有数据复用

以下已有实验结果可直接整理后复用：

| 已有文件 | 对应实验 | 说明 |
|---------|---------|------|
| `docs/experiment_report.md` | exp1 + exp2 + exp3 | 已有完整数据表，可直接转录为 CSV |
| `docs/realtime_performance_analysis.md` | exp5 | 已有时间分析，可提取数值 |
| `docs/rm65_tennis_report.md` | 方法描述 | 可复用系统架构描述 |

从已有文档提取数据的步骤：
1. 将 `experiment_report.md` 中的表格手动转录到对应 exp 目录的 CSV
2. 补充缺失的 seed 级别数据（目前只有汇总，需要逐 seed 数据）
3. 对比已有数据与重新运行的结果一致性
