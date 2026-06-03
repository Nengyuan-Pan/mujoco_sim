# 实验记录：exp2 严格关节约束（实时 v5，3 seeds 试探）

**日期**: 2026-06-02
**实验编号**: exp2_strict_joint
**数据目录**: `experiment_data/exp2_strict_joint/`

## 目的

验证 RM-65B 真实关节约束下（qdot≤1.0×, TCP≤1.8m/s），球速 9 m/s 时实时脚本的命中能力。

## 参数

| 变量 | 值 |
|------|-----|
| 球速范围 | 9 m/s |
| Tube | on/off |
| Seeds | 0-2（3 个） |
| 约束 | 严格（forward_pass_margin=1.0, qdot_scale=1.0） |
| 脚本 | `rm65_mpc_tube_constraint_realtime_v5.py` |
| 发球模式 | serve_box bounce |
| 总运行数 | 6 |

## 结果

| 球速 | Tube ON | Tube OFF | 说明 |
|------|---------|----------|------|
| 9 m/s | 0/3 | 0/3 | 全部 miss |

## 结论

1. 实时脚本在 9 m/s 下严格约束命中率为 0%，TCP 1.8 限制频繁触发紧急制动（22-26 次/run）
2. 需降低球速测试（转向 exp2_v2, 7-8 m/s）或切换到离线脚本
