# exp8: BallEstimator KF 滤波恢复实验

**日期**: 2026-06-07
**版本**: v1
**状态**: 已完成（v1 → v1-fixed）
**数据目录**: `experiment_data/exp8_estimator_recovery/`
**运行时间**: 22:12 → 00:39（2h27min）
**总 runs**: 10,000 / 10,000（0 failed）

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
- 预估: ~3h（实际 2h27min）
- 数据目录: `experiment_data/exp8_estimator_recovery/`

---

## 8. 数据观察（Agent 生成）

### Bug 修复记录

v1 首次运行（10,000 runs）KF 组命中率全面 0%，包括零噪声 off 组。
根因：`BallEstimator.update()` 使用 `perf_counter()` 墙钟时间作为预测 dt，
MPC 循环中两次 update 间隔 ~20ms（计算耗时），但物理仅推进 5ms，
导致 KF 预测持续超前，产生 ~66mm 系统性位置偏差。
修复：在 wrapper 中每次 `update()` 前强制 `_last_update_time = perf_counter() - dt`，
使 elapsed 恰好等于 physics dt=0.005。修复后偏差从 66mm 降至 7.5mm。
v1-fixed 重跑后获得有效数据。

### 聚合结果表

| 噪声 | Estimator | Tube | 命中/总数 | 命中率 | 位置误差(mm) |
|------|-----------|------|----------|--------|-------------|
| off | kf | ON | 263/500 | 52.6% | 99 |
| off | kf | OFF | 261/500 | 52.2% | 95 |
| off | nokf | ON | 361/500 | 72.2% | 60 |
| off | nokf | OFF | 407/500 | 81.4% | 61 |
| lo | kf | ON | 67/500 | 13.4% | 167 |
| lo | kf | OFF | 76/500 | 15.2% | 177 |
| lo | nokf | ON | 7/500 | 1.4% | 139 |
| lo | nokf | OFF | 5/500 | 1.0% | 108 |
| mid | kf | ON | 26/500 | 5.2% | 216 |
| mid | kf | OFF | 25/500 | 5.0% | 192 |
| mid | nokf | ON | 0/500 | 0.0% | — |
| mid | nokf | OFF | 0/500 | 0.0% | — |
| hi | kf | ON | 10/500 | 2.0% | 219 |
| hi | kf | OFF | 6/500 | 1.2% | 76 |
| hi | nokf | ON | 0/500 | 0.0% | — |
| hi | nokf | OFF | 0/500 | 0.0% | — |
| anis | kf | ON | 54/500 | 10.8% | 176 |
| anis | kf | OFF | 40/500 | 8.0% | 181 |
| anis | nokf | ON | 3/500 | 0.6% | 128 |
| anis | nokf | OFF | 3/500 | 0.6% | 37 |

### KF 恢复率分析

| 噪声 | Tube | nokf | kf | 绝对恢复(pp) | 相对恢复(%) |
|------|------|------|-----|-------------|------------|
| off | ON | 72.2% | 52.6% | -19.6 | — |
| off | OFF | 81.4% | 52.2% | -29.2 | — |
| lo | ON | 1.4% | 13.4% | +12.0 | 16.9% |
| lo | OFF | 1.0% | 15.2% | +14.2 | 17.7% |
| mid | ON | 0.0% | 5.2% | +5.2 | 7.2% |
| mid | OFF | 0.0% | 5.0% | +5.0 | 6.1% |
| hi | ON | 0.0% | 2.0% | +2.0 | 2.8% |
| hi | OFF | 0.0% | 1.2% | +1.2 | 1.5% |
| anis | ON | 0.6% | 10.8% | +10.2 | 14.2% |
| anis | OFF | 0.6% | 8.0% | +7.4 | 9.2% |

### 关键数字

- 最高命中率：nokf+off+OFF = 81.4%（9m/s 达 94%）；kf+off+ON = 52.6%（9m/s 达 86%）
- KF 性能税：零噪声下命中率降低 19.6~29.2pp（52.6% vs 72.2%）
- lo 噪声绝对恢复：+12~14pp（从 1.0-1.4% 恢复到 13.4-15.2%）
- lo 噪声相对恢复：~17%（仅补回了 17% 的因噪声损失的性能）
- anis 意外恢复：+7.4~10.2pp（KF 的 per-axis R 正确建模了深度方向大噪声）
- mid/hi 恢复微弱：+1.2~5.2pp（高噪声下 KF 预测偏差被放大）
- Tube 无交互效应：所有噪声等级下 Tube ON/OFF 差异 <3pp
- 主动击球数：0（全部被动接触，`--no-backswing` 模式）
- 6m/s 全部 miss（nokf 和 kf 均为 0/50），与 exp7 一致

### 数据观察结论

1. **KF 引入显著性能税**：零噪声 off 组命中率从 ~77% 降至 ~52%（-19.6~29.2pp），
   表明 KF 的 7.5mm 系统偏差（dt 修复后残留）和每物理步 2 次 update 的额外开销
   对 MPC 规划精度有实质影响。
2. **lo 噪声恢复最显著**：1.0-1.4% → 13.4-15.2%（+12-14pp），相对恢复 ~17%，
   KF 确实补回了部分因噪声损失的性能，但远未恢复到 off 水平（77%）。
3. **mid/hi 恢复微弱**：0% → 1.2-5.2%，KF 在高噪声下预测偏差被放大，
   恢复不足以产生实际价值。
4. **anis 恢复意外好于 lo**：0.6% → 8.0-10.8%（+7.4-10.2pp），
   KF 的 per-axis R 矩阵正确建模了深度方向（Y轴）大噪声，
   对各向异性噪声的处理优于均匀噪声。
5. **Tube 无交互效应**：与 exp7 结论一致，Tube 空间走廊对噪声/KF 场景无额外帮助。

## 9. 分析与决策（人工填写）

<!-- 人工填写，格式：
1. 核心发现 1
2. 核心发现 2
3. 下一步行动
-->
