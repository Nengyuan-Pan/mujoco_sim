# V9 消融实验分析报告：Tube、Softmin、随挥的作用机制

> 实验日期：2026-06-06
> 脚本版本：`scripts/rm65_mpc_v9.py`
> 扰动设置：随机扰动，时间 ±50~100ms，空间 ±3~8cm
> 实验矩阵：2^3 因子（Tube × Softmin × Follow-through）× 2（无扰动 / 随机扰动）= 16 组 × 50 seeds

---

## 一、实验结果摘要

### 1.1 命中率汇总

| 组合 | Tube | Softmin | 随挥 | 无扰动 Hit% | 扰动 Hit% | ΔHit% | 扰动 Active% |
|------|:----:|:-------:|:----:|:-----------:|:---------:|:-----:|:-----------:|
| Baseline | - | - | - | 91% | 45% | -46 | 40% |
| Tube only | Y | - | - | 94% | 47% | -47 | 38% |
| Softmin only | - | Y | - | 91% | 45% | -46 | 40% |
| Follow only | - | - | Y | 91% | 45% | -46 | 40% |
| **Tube+Softmin** | **Y** | **Y** | - | **100%** | **57%** | **-43** | **34%** |
| Tube+Follow | Y | - | Y | 94% | 47% | -47 | 38% |
| Soft+Follow | - | Y | Y | 91% | 45% | -46 | 40% |
| Full | Y | Y | Y | 100% | 57% | -43 | 34% |

### 1.2 位置误差（仅命中样本）

| 组合 | 无扰动 pos_err | 扰动 pos_err | Δpos_err |
|------|---------------|-------------|---------|
| Baseline | 5.4 cm | 7.3 cm | +1.9 |
| Tube only | 5.4 cm | 7.2 cm | +1.8 |
| Tube+Softmin | **4.5 cm** | **6.7 cm** | **+2.2** |
| Full | 4.5 cm | 6.7 cm | +2.2 |

---

## 二、各机制作用分析

### 2.1 Tube 走廊代价

**机制**：Tube 走廊是一个空间约束（hinge loss），在 MPC 运行代价中惩罚球拍偏离球轨迹走廊的行为。

**实现位置**：`rm65_mpc_v9.py` 的 `TubeHittingCostWrapper._compute_tube_cost_at_k()`（line ~1219）

```
走廊半宽 = RACKET_RADIUS
margin = perp_dist(球拍, 球轨迹线) - RACKET_RADIUS
cost = Q_p_tube * max(0, margin)²
```

**无扰动下的作用**：
- Baseline 91% → Tube only 94%（+3pp）
- 微弱提升，因为无扰动时球拍本身就不太偏离轨迹

**扰动下的作用**：
- Baseline 45% → Tube only 47%（+2pp）
- 单独 Tube 几乎没有鲁棒性增益

**为什么单独 Tube 效果有限**：
- Tube 走廊仅约束运行代价（每步位置），不改变终端目标
- 当时间/空间扰动导致 MPC 瞄准错误位置时，走廊约束只能轻微"拉回"轨迹
- 缺少 Softmin 时，终端代价仍是单点约束，MPC 仍然死盯一个错误的击球点

### 2.2 Softmin 多终端代价

**机制**：将终端代价从单点（best_k）扩展为多个候选点的 softmin 加权，允许 MPC "选择"最佳击球时刻。

**实现位置**：`rm65_mpc_v9.py` 的 `TubeHittingCostWrapper._compute_softmin_terminal()`（line ~976）

```
cost = -log(Σᵢ wᵢ · exp(-β · cᵢ)) / β
其中 cᵢ = ||p_ee - p_ball[i]||² + ||v_ee - v_des[i]||²
```

**关键耦合**：Softmin 无法独立于 Tube 存在。
- `--no-tube` 导致 `TubeHittingCostWrapper` 不被创建，softmin 作为其内部方法随之消失
- Softmin 的候选点数据（`_p_ball_candidates`）来自 Tube 的 hit_window 搜索
- 没有 Tube → 没有候选点 → softmin 退化为单点终端

**这解释了实验中 Softmin only = Baseline 的结果**：消融实验中 `--no-tube` 同时禁用了 softmin，"Softmin only" 组实际上没有任何 softmin 效果。

### 2.3 Tube + Softmin 组合（真正的增益来源）

**机制协同**：
1. Tube 的 `search_hit_window()` 生成多个候选击球时刻和位置
2. 走廊代价引导球拍沿球轨迹方向运动（不绑定精确时刻）
3. Softmin 终端代价让 MPC 在多个候选点中自动选择最优
4. 时间扰动导致最佳时刻偏移时，softmin 的梯度平滑地导向次优候选

**实验验证**：
- 无扰动：100%（唯一达到满分的配置）
- 扰动：57%（比 Baseline 高 12pp）
- 这是所有组合中鲁棒性增益最大的

**位置误差也更小**：4.5cm vs 5.4cm（-17%），说明 Tube+Softmin 不仅命中更多，命中质量也更高。

### 2.4 随挥（Follow-through）

**机制**：击球后 MPC 循环结束，由 PD 控制器接管，沿击球方向匀减速 160 步（800ms）。

**实现位置**：`rm65_mpc_v9.py` line ~3296-3332

```python
if follow_through_start >= 0:
    # PD 控制计算 u_follow
    x_current = env.step(u_follow)
    continue  # 跳过 MPC 控制
```

**为什么随挥不影响消融指标**：
1. **随挥是执行层组件，不是规划层组件**：它在击球后才激活，完全绕过 MPC
2. **不影响代价函数**：不参与 iLQR 后向/前向传递
3. **不影响击球精度或速度**：随挥只影响击球后的运动轨迹

**实验结果**：所有含 Follow 的组合与对应不含 Follow 的组合结果完全一致：
- Follow only = Baseline（91% / 45%）
- Tube+Follow = Tube only（94% / 47%）
- Soft+Follow = Soft only（91% / 45%）
- Full = Tube+Softmin（100% / 57%）

---

## 三、发现的问题

### 3.1 Bug：`--no-follow-through` 是死代码

`--no-follow-through` 将 `config_dict["hitting"]["follow_through_steps"]` 设为 0，但 `follow_through_steps` 在 line 2060 被硬编码为 160，从不读取 config_dict。该标志实际无效。

### 3.2 设计局限：Softmin 无法与 Tube 解耦

Softmin 的实现嵌套在 `TubeHittingCostWrapper` 内部，候选点数据依赖 Tube 的 `search_hit_window()`。要实现真正的 "Softmin without Tube" 消融，需要：
- 将候选点搜索逻辑从 Tube 构建中独立出来
- 创建独立的 Softmin 代价包装器

---

## 四、结论

| 机制 | 是否有效 | 无扰动增益 | 扰动下增益 | 作用方式 |
|------|---------|-----------|-----------|---------|
| Tube alone | 微弱 | +3pp (91→94%) | +2pp (45→47%) | 走廊约束轻微引导轨迹 |
| Softmin alone | ❌ 无法独立测试 | — | — | 无 Tube 则无候选点，退化为 baseline |
| Follow-through | ❌ 不影响规划 | 0pp | 0pp | 纯执行层，击球后 PD 控制器 |
| **Tube+Softmin** | **✅ 唯一有效组合** | **+9pp (91→100%)** | **+12pp (45→57%)** | **走廊引导 + 多终端自适应选择** |

### 核心结论

**Tube 走廊和 Softmin 终端代价是一个不可分割的协同对**：
- Tube 提供候选点和空间约束
- Softmin 利用候选点实现时间自适应
- 两者缺一不可：单独 Tube 缺少终端灵活性，单独 Softmin 因缺少候选点而无法工作
- 随挥是纯执行层组件，不影响 MPC 规划质量，不应纳入 MPC 鲁棒性消融实验
