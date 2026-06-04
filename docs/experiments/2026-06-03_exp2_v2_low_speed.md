# 实验记录：exp2 低球速扫参（离线，7-8 m/s）

**日期**: 2026-06-03
**实验编号**: exp2_strict_joint_v2
**数据目录**: `experiment_data/exp2_strict_joint_v2/`

## 目的

降低球速到 5-8 m/s，验证严格约束下的命中率基线。同时测试 serve_box 最低生成球速。

## 参数

| 变量 | 值 |
|------|-----|
| 球速范围 | 5, 6, 7, 8 m/s |
| Tube | on/off |
| Seeds | 0-9（10 个） |
| 约束 | 严格（forward_pass_margin=1.0, qdot_scale=1.0） |
| 脚本 | `rm65_mpc_tube_constraint.py`（离线） |
| 发球模式 | serve_box bounce |
| 总运行数 | 80 |

## 结果

| 球速 | Tube ON | Tube OFF | 误差 ON/OFF | 备注 |
|------|---------|----------|-------------|------|
| 5 m/s | — | — | — | serve_box 最低 8m/s，全部生成失败 |
| 6 m/s | — | — | — | 同上 |
| 7 m/s | 8/10 | 8/10 | 28.7/28.7mm | |
| 8 m/s | 8/10 | 8/10 | 39.6/39.0mm | |

## 结论

### 数据观察（Agent 生成）

1. 严格约束下 7-8 m/s 命中率均为 80%（Tube ON 8/10，Tube OFF 8/10），位置误差 29-40mm。
2. 5-6 m/s 全部生成失败（serve_box bounce 模式最低球速限制 8 m/s），40 runs 中 20 runs 失败。
3. Tube ON/OFF 误差几乎相同（28.7mm vs 28.7mm @ 7m/s），低速无扰动下无差异。

### 分析与决策（人工填写）

1. 严格约束下 7-8 m/s 命中率 80%，位置误差 29-40mm。
2. serve_box 弹跳模式最低球速 8 m/s，低于此值全部 RuntimeError。
3. Tube ON/OFF 在低速无扰动时无差异 — 需扰动实验验证 Tube 价值。
