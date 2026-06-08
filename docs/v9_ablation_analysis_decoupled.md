# V9 消融实验分析（解耦版）

> 日期: 2026-06-07
> 脚本: `scripts/rm65_mpc_v9.py` (新增 `--ablation` 参数)
> 扰动: 随机时间 ±50~100ms + 空间 ±3~8cm
> 50 seeds × 4 workers, 16 组 = 800 runs, 总耗时 9min

## 1. 实验设计

### 2^3 因子设计

| 因子 | 实现方式 | 水平 |
|------|---------|------|
| Tube 走廊 | `--ablation` 包含 tube_only/full | ON/OFF |
| Softmin 终端 | `--ablation` 包含 softmin_only/full | ON/OFF |
| Follow-through | `--no-follow-through` 标志 | ON/OFF |

8 种组合 × 2 条件（Nominal / Perturbed）= **16 组实验**。

### 关键代码改动

1. **修复 `--no-follow-through` 死代码**: `follow_through_steps` 从硬编码 160 改为读取 config_dict
2. **新增 `--ablation` 参数**: 替代旧的 `--no-tube`/`--no-softmin`，支持 4 种模式
3. **`TubeOnlyCost` 类**: 继承 `TubeHittingCostWrapper`，强制 `_use_softmin=False`
4. **`SoftminOnlyCost` 类**: 继承 `TubeHittingCostWrapper`，清空走廊步骤集合
5. **候选点搜索解耦**: `search_hit_window()` 不再依赖 `use_tube` 条件
6. **修复 `total_horizon` 污染 bug**: 分离 `mpc_horizon` 和 `total_horizon`，修复随挥步数干扰 `remaining_horizon` 计算的问题

## 2. 完整结果

| Tube | Softmin | Follow | 扰动 | 命中率 | 主动率 | 位置误差 | 拍速 |
|------|---------|--------|------|--------|--------|----------|------|
| - | - | - | - | 86% | 86% | 5.4cm | 1.30 |
| Y | - | - | - | 88% | 88% | 5.4cm | 1.35 |
| - | Y | - | - | **94%** | **94%** | 4.5cm | 1.19 |
| - | - | Y | - | 86% | 86% | 5.4cm | 1.30 |
| Y | Y | - | - | **94%** | **94%** | 4.5cm | 1.19 |
| Y | - | Y | - | 88% | 88% | 5.4cm | 1.35 |
| - | Y | Y | - | **94%** | **94%** | 4.5cm | 1.19 |
| Y | Y | Y | - | **94%** | **94%** | 4.5cm | 1.19 |
| - | - | - | Y | 62% | 62% | 7.5cm | 0.93 |
| Y | - | - | Y | 64% | 60% | 7.3cm | 0.80 |
| - | Y | - | Y | **88%** | **66%** | 7.6cm | 0.63 |
| - | - | Y | Y | 62% | 58% | 7.5cm | 0.93 |
| Y | Y | - | Y | **88%** | **66%** | 7.7cm | 0.63 |
| Y | - | Y | Y | 64% | 56% | 7.3cm | 0.80 |
| - | Y | Y | Y | **88%** | **66%** | 7.6cm | 0.63 |
| Y | Y | Y | Y | **88%** | **66%** | 7.7cm | 0.63 |

## 3. 关键发现

### 3.1 Softmin 是核心鲁棒机制

**Nominal 条件下**：
- Softmin ON（无论 Tube/Follow 状态）: 命中率 **94%**
- Softmin OFF: 命中率 **86-88%**
- 提升: **+6~8pp**

**扰动条件下**：
- Softmin ON: 命中率 **88%**（vs baseline 62%，**+26pp**）
- Softmin OFF: 命中率 **62-64%**

Softmin 是唯一有显著独立贡献的机制。

### 3.2 Follow-through 不影响命中率（修复后）

| 对比 | Nominal | Perturbed |
|------|---------|-----------|
| Baseline → +Follow | 86% → 86% (0) | 62% → 62% (0) |
| Softmin → Soft+Follow | 94% → 94% (0) | 88% → 88% (0) |
| Tube+Soft → Full | 94% → 94% (0) | 88% → 88% (0) |

修复 `total_horizon` 污染 bug 后，Follow-through 对命中率**零影响**。
这符合预期：随挥是击球后的 PD 控制器，不影响 MPC 规划质量。

**Bug 根因**：`total_horizon += follow_through_steps` 使 `remaining_horizon` 偏大，导致 MPC 在重规划时搜索到更远的击球点，从而产生不同的轨迹。例如同一 seed 下：
- 无随挥: `REPLAN step=220 k_hit=29` → 球到达时命中
- 有随挥: `REPLAN step=220 k_hit=37` → 球未到达，MPC 已结束

### 3.3 Tube 走廊无独立贡献

| 对比 | Nominal | Perturbed |
|------|---------|-----------|
| Baseline → +Tube | 86% → 88% (+2) | 62% → 64% (+2) |
| Softmin → Tube+Softmin | 94% → 94% (0) | 88% → 88% (0) |

Tube 走廊仅贡献 +2pp，统计不显著。走廊的 hinge loss 权重太低（Q_p_tube=500 vs 终端 Q_p=200000，比例 0.25%）。

### 3.4 各机制效应总结

| 效应 | Nominal | Perturbed |
|------|---------|-----------|
| 主效应: Softmin | +6~8pp | **+26pp** |
| 主效应: Tube | +2pp | +2pp |
| 主效应: Follow | 0pp | 0pp |
| 交互: Softmin×Tube | 0pp | 0pp |
| 交互: Softmin×Follow | 0pp | 0pp |
| 交互: Tube×Follow | 0pp | 0pp |
| 三阶交互 | 0pp | 0pp |

**结论**：三个机制之间无交互效应。Softmin 独立贡献全部鲁棒性增益。

## 4. 修复的 Bug 清单

| Bug | 影响 | 修复 |
|-----|------|------|
| `--no-follow-through` 死代码 | 随挥无法关闭 | 从 config_dict 读取 `follow_through_steps` |
| `total_horizon += follow_through_steps` | `remaining_horizon` 被随挥步数污染，导致 MPC 规划偏移 | 引入 `mpc_horizon`，`remaining_horizon` 使用 `mpc_horizon - step` |
| 随挥触发当轮的 double-step | 同一迭代执行两次 `env.step` | 添加 `step > follow_through_start` 条件 |
| softmin 诊断引用 `base_cost_fn._softmin_alpha_cache` | 属性不存在于 base_cost_fn | 改为 `cost_fn._last_softmin_alphas` |
| `tube_cfg.use_softmin_terminal` 不随 ablation 模式变化 | `--ablation tube_only` 时 softmin 未真正关闭 | 改为 `ablation_mode in ("full", "softmin_only")` |

## 5. 后续建议

1. **增强 Tube 走廊**: 增大 Q_p_tube 权重（当前 500 vs 终端 200000），或改为硬约束
2. **测试更大扰动范围**: 当前 ±50~100ms/±3~8cm 是温和扰动，应测试极端情况
3. **探索 Softmin 的最优参数**: beta 值、候选窗口宽度对鲁棒性的影响
4. **评估 Follow-through 的运动学收益**: 虽不影响命中率，但可能影响击球后的安全减速
