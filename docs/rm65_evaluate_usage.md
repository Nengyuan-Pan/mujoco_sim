# RM-65 评估脚本 (rm65_evaluate.py) 使用说明

> 脚本文件：`scripts/rm65_evaluate.py`
> 基于：`scripts/rm65_mpc_ilqr_5_5.py`（共享全部 MPC+iLQR 核心逻辑）
> 差异：新增反弹球模式、多维度评估报告、matplotlib 图表生成

---

## 一、与基础脚本的关系

`rm65_evaluate.py` 是 `rm65_mpc_ilqr_5_5.py` 的超集。两个脚本共享完全相同的：
- 所有辅助函数（15个）
- MPC 主循环逻辑
- iLQR 求解器调用方式
- 可视化函数 `visualize_rm65_result()`
- 后摆 / R 退火 / 关节5 固定逻辑

### 差异点汇总

| 方面 | 基础版 | 评估版 |
|------|--------|--------|
| 球生成 | 仅直球模式 | 直球 + **反弹球** (`--bounce`) |
| 评估报告 | 简单 print | 格式化文本 + **matplotlib 图表** |
| 关节分析 | 无 | 限位裕度 / 速度峰值 / 力矩峰值 |
| CPU 性能统计 | 基础 | 详细（每步/重规划耗时分布） |
| 命令行参数 | ~12 项 | ~17 项（+bounce/no-plot 系列） |

---

## 二、新增模块级常量

```python
# RM-65 关节名称
RM65_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

# 关节限位（度），来自 rm65_model.xml range 属性
RM65_JOINT_LIMIT_DEG = [[-178,178], [-130,130], [-135,135], [-178,178], [-128,128], [-360,360]]

# 额定关节速度 (rad/s)
RM65_JOINT_VEL_LIMIT = [3.14, 3.14, 3.14, 3.93, 3.93, 3.93]

# 力矩限位 (Nm)，来自 rm65_model.xml ctrlrange
RM65_TORQUE_LIMIT = [60.0, 60.0, 30.0, 10.0, 10.0, 10.0]
```

---

## 三、新增功能详解

### 3.1 反弹球模式 (`--bounce`)

球从约 20m 远（-Y方向）发出，飞行中触地弹跳一次（e=0.75），弹起后飞向击打区。

**算法**（`generate_bounce_ball()` 在 `src/tennis/ball.py`）：

```
给定:
  p_hit     ← 在 target_center ± 0.2m 内采样
  hit_time  ← total_horizon * dt * U(0.65, 0.80)
  t_bounce  ← hit_time * U(0.40, 0.60)    (弹跳时刻)

反推：
  1. p_bounce  ← p_hit - v_after · t_after (弹跳点在 Z=0)
  2. vz_after  ← (p_hit_z - 0.5·g·t_after²) / t_after
  3. v_after   ← [vx_after, vy_after, vz_after]  (满足 speed ∈ [16,22] m/s)
  4. v_before  ← [vx_after, vy_after, -vz_after / 0.75]  (Z 反转除恢复系数)
  5. v0        ← v_before - g · t_bounce
  6. p0        ← p_bounce - v0 · t_bounce - 0.5 · g · t_bounce²
```

返回的 `(p0, v0, p_hit)` 与解析弹跳模型 `ball_trajectory_with_bounce()` 完全一致。

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bounce` | 关 | 启用 |
| `--bounce-distance` | 20.0 | 发球距离 (m) |
| `--bounce-speed-min` | 16.0 | 到达击打区最低速度 (m/s) |
| `--bounce-speed-max` | 22.0 | 到达击打区最高速度 (m/s) |

**自动适配**：反弹球启用时 `total_horizon` 从 200 自动增至 240 或更长。

### 3.2 轨迹评估 (`evaluate_trajectory()`)

对完整轨迹 `X[0..N+1, 12]` 和 `U[0..N, 6]` 执行 9 类分析：

| 维度 | 计算方法 | 返回字段 |
|------|---------|---------|
| 球拍位置/速度 | `env.get_ee_pos()` / `get_ee_vel()` 逐帧 | `ee_pos`, `ee_vel`, `ee_speed` |
| 关节限位裕度 | `min(q-lo, hi-q)` 每关节每时刻 | `joint_margins`, `min_margin` |
| 关节速度峰值 | `max(abs(qdot))` 每关节 | `peak_qdot`, `peak_qdot_ratio` |
| 关节加速度峰值 | `(qdot[i+1]-qdot[i])/dt` 中心差分 | `peak_qacc`, `peak_qacc_ratio` |
| 力矩峰值 | `max(abs(U))` 每关节 | `peak_torque`, `peak_torque_ratio` |
| 球拍速度峰值 | `max(ee_speed)` + 时刻 | `peak_ee_speed_time` |
| 击打时刻球拍速度 | `ee_speed[hit_step]` | `ee_speed_at_hit`, `ee_vel_at_hit` |
| 位置误差序列 | `‖ee_pos - p_hit‖` 每帧 | `pos_err_traj` |
| 球轨迹 | `ball_pos_arr[:N+1]` | `ball_pos` |

返回字典供 `plot_evaluation()` 和 `print_evaluation_report()` 消费。

### 3.3 文本评估报告 (`print_evaluation_report()`)

```
============================================================
  RM-65 Tennis Hitting Evaluation Report
============================================================

--- Hit Accuracy ---
  Position error: X.XX cm
  Velocity error: X.XX m/s

--- Ball Speed ---
  Before hit: XX.XX m/s
  After hit:  XX.XX m/s
  Change:     ±X.XX m/s
  Rebound ratio: X.XXx

--- Racket Center Speed ---
  At hit time: X.XX m/s
  Direction:   [x, y, z]
  Peak speed:  X.XX m/s (t=X.XXXs)

--- Joint Limit Analysis ---
  J1 (shoulder_pan  ): limit=[-178, 178]deg, actual=[  X,  Y]deg, margin=  Zdeg  OK/WARN/EXCEED!
  ...

--- Joint Velocity Analysis ---
  J1 (shoulder_pan  ): peak=  XXX.X deg/s, limit=XXX.X deg/s, ratio=XX.X%  OK/WARN/EXCEED!
  ...

--- Joint Acceleration Analysis ---
  J1 (shoulder_pan  ): peak=  XXX.X deg/s², limit=600.0 deg/s², ratio=XX.X%  OK/WARN/EXCEED!
  ...

--- Torque Analysis ---
  J1 (shoulder_pan  ): peak=  XX.XX Nm, limit=XX.X Nm, ratio=XX.X%  OK/WARN/CLIPPED!
  ...
============================================================
```

状态判定：
- **OK**：在额定范围内（裕度>5°，速度<90%限速，力矩<90%限值）
- **WARN**：接近限额（裕度 0~5°，速度 90~100%，力矩 90~100%）
- **EXCEED!**：超限（裕度<0 或 >100%）
- **CLIPPED!**：力矩被裁剪到100%限值

### 3.4 matplotlib 评估图表 (`plot_evaluation()`)

生成 6 个子图的 PNG 报告，保存到 `results/rm65_evaluation.png`：

| 子图 | 内容 | 说明 |
|------|------|------|
| 1 | 球拍速度 vs 时间 | 标出击打时刻、峰值时刻 |
| 2 | 位置误差 vs 时间 | 末端-击打点距离随时间收敛 |
| 3 | 关节角度 vs 时间 | 6个关节角度 + 限位虚线 |
| 4 | 关节速度 vs 时间 | 6个关节速度 + 额定限速虚线 |
| 5 | 关节力矩 vs 时间 | 6个关节力矩 + 限位虚线 |
| 6 | 关节限位裕度 vs 时间 | 距上限/下限的最小角度余量 |

可通过 `--no-plot` 跳过生成。

### 3.5 CPU 性能统计

MPC 结束后输出：

```
MPC 完成: 总耗时=X.XXs, 平均每步=XX.Xms, 平均重规划=XXX.Xms, 最慢步=XXX.Xms, 实时倍率=X.XXx
```

| 指标 | 说明 |
|------|------|
| 总耗时 | 从 MPC 开始到结束的 wall-clock 时间 |
| 平均每步 | 含重规划平摊后每步计算时间 |
| 平均重规划 | 仅 iLQR 重规划调用的耗时均值 |
| 最慢步 | 首次规划步耗时（通常含冷启动） |
| 实时倍率 | `sim_time / wall_time`，<1 表示慢于实时 |

---

## 四、命令行参数完整列表

```
python scripts/rm65_evaluate.py [options]

MPC 核心参数:
  --viewer              回放可视化
  --seed SEED           随机种子
  --fd                  使用有限差分线性化
  --horizon N           iLQR 短地平线步数 (默认30)
  --iter N              每次重规划迭代数 (默认8)
  --fix-joint5          固定第6关节
  --no-plot             跳过评估图表生成

击打参数:
  --backswing FLOAT     后摆幅度 rad (默认0.6)
  --bs-ratio FLOAT      后摆时间占比 (默认0.35)
  --no-backswing        禁用后摆
  --hit-shift FLOAT     随挥偏移 m (默认0.01)
  --r-decay FLOAT       R退火占比 (默认0.30)
  --no-r-decay          禁用R退火
  --normal-weight FLOAT 拍面法向量权重 (默认500000)
  --normal-flip         翻转法向量方向

球参数:
  --ball-speed FLOAT    直球到达击打点速度 m/s
  --bounce              启用反弹球模式
  --bounce-distance FLOAT 发球距离 m (默认20)
  --bounce-speed-min FLOAT 击打区最低速度 m/s (默认16)
  --bounce-speed-max FLOAT 击打区最高速度 m/s (默认22)
```

---

## 五、典型运行示例

```bash
# 标准评估（直球，生成报告+图表）
python scripts/rm65_evaluate.py --seed 42 --normal-flip

# 反弹球评估（球从20m飞来，地面弹跳）
python scripts/rm65_evaluate.py --bounce --normal-flip --viewer

# 快速评估（不绘图）
python scripts/rm65_evaluate.py --seed 42 --normal-flip --no-plot

# 反弹球 + 指定速度范围
python scripts/rm65_evaluate.py --bounce --bounce-speed-min 18 --bounce-speed-max 22

# 多种子批量测试
for s in 42 0 7 99 123; do
    python scripts/rm65_evaluate.py --seed $s --normal-flip --no-plot
done
```

---

## 六、输出文件

| 文件 | 说明 |
|------|------|
| `results/rm65_evaluation.png` | matplotlib 评估图表 |
| stdout | 详细文本评估报告 + 各步日志 |

---

## 七、灯光配置

在 `visualize_rm65_result()` 中设置：

| 灯光 | 类型 | 位置 | 色温 | 作用 |
|------|------|------|------|------|
| Light 0 | 天光 | [0,0,8] | 冷白 1.45 | 模拟天空整体照明 |
| Light 1 | 主光 | [2,-2,3] | 暖白 1.15 | 45°侧光 |
| Light 2 | 补光 | [-1.5,-1,2.5] | 冷白 0.85 | 填充阴影 |
| Light 3 | 背光 | [0,2,2] | 中性 0.5 | 轮廓光 |
| Light 4 | 球灯 | 跟随 ball body | 暖黄 0.7 | 球始终可见 |

XML 模型文件中定义前 4 个全局灯，球灯是 body-local light。脚本中可动态调参。
