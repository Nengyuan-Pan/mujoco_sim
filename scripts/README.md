# scripts/ 目录结构

## 组织原则

- **根目录保留 10 个被其他脚本 import 或 subprocess 调用的核心仿真脚本**
- **其余按功能分类入 6 个子目录**

## 目录说明

| 目录 | 数量 | 内容 | 运行方式 |
|------|------|------|---------|
| `scripts/` (根) | 10 | 核心仿真脚本（被 cross-import / subprocess 调用） | `python scripts/xxx.py --args` |
| `sim/` | 17 | 独立仿真脚本（MPC/iLQR/Training/变体） | `python scripts/sim/xxx.py --args` |
| `exp/` | 43 | 实验基础设施（包装·批量·运行器） | `python scripts/exp/xxx.py --args` |
| `extract/` | 6 | 结果提取：日志 → CSV | `python scripts/extract/xxx.py` |
| `plot/` | 12 | 论文图表生成 | `python scripts/plot/xxx.py` |
| `tools/` | 10 | 独立工具（查看器·扫描·诊断·可视化） | `python scripts/tools/xxx.py` |
| `test/` | 9 | 快速验证脚本 | `python scripts/test/xxx.py` |

## 详细清单

### 根目录（10 个）

被 `exp/` 下脚本通过 Python `import` 或 `subprocess` 引用，不可移动：

| 文件 | 用途 | 引用方式 |
|------|------|---------|
| `rm65_mpc_tube_constraint.py` | 离线 MPC+iLQR+Tube+硬约束 | exp1 豁免包装 import |
| `rm65_mpc_tube_constraint_realtime.py` | 实时 v1 | TCP 限速实验 import |
| `rm65_mpc_tube_constraint_realtime_v2.py` | 实时 v2 | V3 实验、paper 实验 subprocess |
| `rm65_mpc_tube.py` | 原始 Tube-based 击打 | scan_ball_params import |
| `rm65_mpc_ilqr_5_5.py` | MPC+iLQR 带后摆 | realtime_batch import |
| `rm65_evaluate.py` | 评估脚本 | realtime_batch import |
| `rm65_mpc_v6.py` | V6：满秩 Q_v + 来球反方向 + softmin + PD 随挥 | 10 个 run_exp 脚本 subprocess |
| `rm65_mpc_v7.py` | V7：V6 + 击球点终端 + TCP/关节硬约束 | 4 个 run_exp 脚本 subprocess |
| `rm65_mpc_v8.py` | V8：解耦 Tube 走廊 + Softmin 终端，支持 `--no-tube`/`--no-softmin` | run_20hits_video import + run_v8_exp subprocess |
| `rm65_mpc_v9.py` | V9：V8 + 更长随挥距离(0.20m) + 更多随挥步数(80步) | 最新迭代 |

### sim/ — 独立仿真脚本（17 个）

| 文件 | 用途 |
|------|------|
| `rm65_mpc_tube_constraint_realtime_v4.py` | 实时 v4（V5 前身） |
| `rm65_mpc_tube_constraint_realtime_v5.py` | ★ 当前活跃：主动击球+随挥+Tube 安全滤波 |
| `rm65_mpc_ilqt.py` | 简化 MPC+iLQR（无 Tube） |
| `rm65_mpc_ilqr_5_7_python.py` | 纯 Python iLQR benchmark |
| `rm65_mpc_fast.py` | 快速模式 |
| `rm65_mpc_fast_workspace.py` | 快速模式 + workspace 约束 |
| `rm65_constrained_fast.py` | 约束快速模式 |
| `rm65_joint_limit.py` | 关节限速版本 |
| `rm65_batch_viz.py` | 批量击球 + 回放 + 视频 |
| `rm65_realtime_batch.py` | 批量评估（20 球汇总统计） |
| `rm65_realtime_play.py` | 实时连续击球 |
| `train_mpc.py` | MPC Rolling Planner 训练 |
| `train_ilqt.py` | 单次 iLQR 优化 + 可视化 |
| `rm65_mpc_v8_softmin_only.py` | V8 变体：仅 softmin（默认 flags 不同） |
| `rm65_mpc_v8_tuned.py` | V8 变体：调参版 |
| `rm65_mpc_v8_tuned_softmin_only.py` | V8 变体：调参 + 仅 softmin |
| `rm65_mpc_v9_softmin_only.py` | V9 变体：仅 softmin |

### exp/ — 实验基础设施（43 个）

#### 包装脚本（5 个）
| 文件 | 用途 |
|------|------|
| `_run_exp1_exempt.py` | 速度豁免 monkey-patch（bounce 模式） |
| `_run_exp1_v3_exempt.py` | 速度豁免 + no-bounce |
| `_run_strict_experiment.py` | 严格约束 monkey-patch |
| `_run_exp2_v3_strict.py` | 严格约束包装（exp2_v3，离线） |
| `_run_exp7_noise.py` | 噪声实验包装 |

#### TCP 限速实验（3 个）
| 文件 | 用途 |
|------|------|
| `run_tcp_limit_experiment.py` | TCP+关节双约束 v1 |
| `run_tcp_limit_experiment_v2.py` | TCP+关节双约束 v2 |
| `run_tcp_limit_experiment_v3.py` | TCP+关节双约束 v3（无豁免） |

#### 批量运行器（5 个）
| 文件 | 用途 |
|------|------|
| `run_exp1_batch.py` | Exp1 bounce 模式扫参 |
| `run_exp1_v3_batch.py` | Exp1 V3 并行扫参（540 runs, 4 workers） |
| `run_exp2_v2_batch.py` | Exp2 低球速扫参（7-8 m/s） |
| `run_exp2_v3_batch.py` | Exp2 V3 严格约束并行扫参（8-18 m/s, 4 workers） |
| `run_exp7_batch.py` | Exp7 噪声实验批量 |

#### 早期实验运行器（12 个）
| 文件 | 用途 |
|------|------|
| `run_experiments.py` | 实验 A（豁免）+ B（严格）对比 |
| `run_tube_robustness.py` / `run_no_tube_robustness.py` | Tube 鲁棒性对比 |
| `run_robustness_batch_v2.py` | V2 鲁棒性并行批量 |
| `run_perturb_stats.py` | 扰动统计 |
| `run_bidirectional_perturb.py` | 双向扰动 |
| `run_exp_ablation_A.py` | 消融：corridor vs softmin |
| `run_expA_random.py` / `run_expB_alpha_sweep.py` | 随机扰动 / alpha 衰减扫参 |
| `run_v3_exp1_rerun.py` / `run_v3_exp2_time.py` / `run_v3_exp2_coupled.py` | V3 专项重跑 |
| `run_v3_experiments.py` | V3 实验批量 |

#### V6+ 实验运行器（15 个）
| 文件 | 用途 |
|------|------|
| `run_exp_ablation_v6.py` | V6 消融：Softmin × 随挥 4 条件 |
| `run_exp_corridor_ablation.py` | 走廊代价消融：v6 vs v7 |
| `run_exp_direction_shift.py` | 方向 × 偏移 2×2 消融 |
| `run_exp_fast_near.py` | fast_lin + near_iters 调参 |
| `run_exp_near_iters.py` | Near_iters 调参实验 |
| `run_exp_offset_sweep.py` | 偏移量扫描：1-5 cm |
| `run_exp_paper.py` | 论文实验：v2/v6/v7 对比 |
| `run_exp_progressive_v2.py` | V2 特性逐步移植消融 |
| `run_exp_qv_sweep.py` | V7 Q_v/Q_p 调参 |
| `run_exp_speed_v6.py` | V6 球速实验：5-11 m/s |
| `run_exp_target_speed.py` | 终端速度消融：1.8/3.0/5.0 m/s |
| `run_exp_v5_robustness.py` | V2+V5 鲁棒性测试 |
| `run_exp_v5v6_compare.py` | V5 vs V6 公平对比 |
| `run_exp_v6_v2feat_perturb.py` | V6 + V2 全特性 + 扰动 |
| `run_v8_exp.py` | V8 批量消融实验 |

#### 其他（3 个）
| 文件 | 用途 |
|------|------|
| `run_tcp_batch.py` | TCP 限速批量 |
| `run_20hits_video.py` | 20 次连续击球 + 视频 |
| `sweep_margins.py` | 关节裕度扫参 |

### extract/ — 结果提取（6 个）

| 文件 | 数据源 |
|------|--------|
| `extract_exp1_results.py` | exp1_algorithm_capability |
| `extract_exp1_v3_results.py` | exp1_v3_algorithm_capability |
| `extract_exp2_results.py` | exp2_strict_joint（实时，含 UTF-16LE 处理） |
| `extract_exp2_v2_results.py` | exp2_strict_joint_v2（离线格式） |
| `extract_exp2_v3_results.py` | exp2_v3_strict_joint（离线格式） |
| `extract_exp7_results.py` | exp7_noise（噪声实验） |

### plot/ — 图表生成（12 个）

| 文件 | 图表 |
|------|------|
| `plot_v3_results.py` | V3 热力图 + 柱状图 |
| `plot_v3_exp2_time.py` | V3 Exp2 时间扰动 |
| `plot_v3_exp2_coupled.py` | V3 Exp2 耦合扰动 |
| `plot_exp_random.py` | 随机扰动对比 |
| `plot_exp_ablation_A.py` | 消融对比 |
| `plot_perturb_results.py` | 扰动结果（含 subprocess 重跑） |
| `plot_ablation_v6.py` | V6 消融 4 条件对比图 |
| `plot_exp_ablation_v6.py` | V6 消融柱状图 |
| `plot_exp_speed_v6.py` | V6 球速实验图表 |
| `plot_qv_sweep.py` | Q_v/Q_p 调参图表（V6） |
| `plot_v7_qv_sweep.py` | Q_v/Q_p 调参图表（V7） |
| `plot_v8_results.py` | V8 消融汇总图 |

### tools/ — 独立工具（10 个）

| 文件 | 用途 |
|------|------|
| `rm65_joint_viewer.py` | 关节调节查看器（position 执行器） |
| `run_rm65.py` | 基础 MuJoCo 查看器 |
| `eval_sim.py` | 仿真评估 |
| `scan_joint_safety.py` | 关节安全范围扫描 |
| `scan_ball_params.py` | 球参数网格扫描 |
| `visualize_robot_parts.py` | 机器人部位可视化 |
| `render_20hits_video.py` | 离屏渲染 MP4 视频 |
| `compare_v3_v6_ablation.py` | V3 vs V6 消融逐 seed 对比分析 |
| `compare_v3_v6_detail.py` | V3 vs V6 详细指标对比 |
| `diagnose_v6_robustness.py` | V6 鲁棒性 per-seed 诊断 |

### test/ — 快速验证（9 个）

全部通过 subprocess 调用 `rm65_mpc_tube.py`，用于快速验证单次运行结果。文件名自描述用途（`test_perturb.py`, `test_compare.py` 等）。

## 运行注意事项

1. **所有脚本从项目根目录运行**：`python scripts/xxx.py ...` 或 `python scripts/sim/xxx.py ...`
2. **`exp/` 下包装脚本不可直接 import**：它们在模块顶层读取 `sys.argv`，只能通过 subprocess 调用
3. **extract 脚本路径已相对化**：从任何位置运行均可正确找到 `experiment_data/`
4. **两个预存语法问题**（非移动引起）：`exp/run_robustness_batch_v2.py` f-string 转义、`exp/run_tcp_limit_experiment_v2.py` BOM 字符
