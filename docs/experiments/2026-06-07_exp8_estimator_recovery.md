# exp8: BallEstimator KF 滤波恢复实验

**日期**: 2026-06-07
**版本**: v1
**状态**: 运行中

## 1. 实验目的

建立"噪声等级 → KF 恢复率"的完整映射，验证 6D 卡尔曼滤波器（BallEstimator）能否恢复因观测噪声损失的 MPC 命中率。

## 2. 背景

exp7 噪声×Tube 消融实验（5000 runs）证明：
- off（无噪声）: Tube ON=72.2%, OFF=81.4%
- lo（σ_p=2cm, σ_v=0.2m/s）: ~1.4%
- mid/hi/anis: 0-0.6%
- Tube 空间走廊对噪声鲁棒性无帮助

pipeline 集成测试（7 tests, 63/63 pass）证明 BallEstimator 平均衰减 80% 位置噪声、73% 速度噪声。

**核心问题**: 滤波精度 ≠ MPC 命中率。KF 能恢复多少实际命中率？

## 3. 实验设计

### 3.1 参数空间

| 维度 | 水平 | 数量 |
|------|------|------|
| 球速 | [6,7,8,9,10,11,12,13,14,15] m/s | 10 |
| 种子 | range(50) | 50 |
| Tube | {true, false} | 2 |
| 噪声 | {off, lo, mid, hi, anis} | 5 |
| Estimator | {kf, nokf} | 2 |
| **总计** | | **10,000** |

### 3.2 对照设计

- **nokf 组**: monkey-patch 噪声注入（与 exp7 完全一致），重新跑
- **kf 组**: 噪声注入 + BallEstimator KF 滤波
- **off+kf**: 阴性对照（零噪声 + KF，验证 KF 不损害性能）

### 3.3 KF 参数映射（R 矩阵精确匹配）

| noise_mode | 实际噪声 | estimator R |
|------------|---------|-------------|
| off | σ=0 | σ_p=0.001, σ_v=0.01（默认 process noise） |
| lo | σ_p=0.02, σ_v=0.2 | σ_p=0.02, σ_v=0.2 |
| mid | σ_p=0.05, σ_v=0.5 | σ_p=0.05, σ_v=0.5 |
| hi | σ_p=0.10, σ_v=1.0 | σ_p=0.10, σ_v=1.0 |
| anis | pos=(.03,.10,.03), vel=(.3,1.0,.3) | 同左 |

### 3.4 Monkey-patch 列表

| Patch | nokf | kf |
|-------|------|-----|
| `step()` 重置 `_kf_consumed` | - | ✓ |
| `get_ball_state()` 加噪声 | ✓ | ✓（噪声→KF） |
| `get_ball_pos/vel()` 加噪声 | ✓ | 返回 KF 缓存 |
| `find_hitting_point_physics` 缓存回退 | ✓ | ✓ |
| `RobotLimits.from_config` 速度豁免 | ✓ | ✓ |

### 3.5 单步推进 flag

`get_ball_state()` 在 MPC 循环中每物理步被调用 5-7 次。用 `_kf_consumed` 布尔 flag 控制每物理步仅推进 estimator 一次，`env.step()` 后重置 flag。

## 4. 数据流（kf 组）

```
MuJoCo 真值 → add_observation_noise() → BallEstimator.update(noisy) → 滤波输出
                                                   ↑
                                          _kf_consumed flag 控制
get_ball_pos/vel → estimator.state 缓存（不推进）
env.step() → _kf_consumed = False
```

## 5. 分析指标

- **绝对恢复**: `hit_rate(kf) - hit_rate(nokf)`（百分点）
- **相对恢复**: `(hit_rate(kf) - hit_rate(nokf)) / (hit_rate(off) - hit_rate(nokf))`（补回损失的比例）
- 按 `noise × tube × estimator` 三维汇总

## 6. 实现文件

| 文件 | 用途 |
|------|------|
| `scripts/exp/_run_exp7_kf.py` | KF wrapper 脚本 |
| `scripts/exp/run_exp8_batch.py` | 批量运行器 |
| `scripts/extract/extract_exp8_results.py` | 结果提取 + 恢复率分析 |
| `experiment_data/exp8_estimator_recovery/config.yaml` | 完整自包含配置 |
| `experiment_data/exp8_estimator_recovery/raw/` | 日志目录 |

## 7. 运行配置

- 主脚本: `rm65_mpc_tube_constraint.py`（与 exp7 一致）
- `--no-backswing --no-plot --serve-box`
- 并行: 4 workers, 180s 超时
- 预估: ~5-6h
- 数据目录: `experiment_data/exp8_estimator_recovery/`

---

## 8. 数据观察（Agent 自动填写）

> 待实验完成后用 experiment-log skill 提取

## 9. 分析决策（人工填写）

> 待数据观察完成后填写
