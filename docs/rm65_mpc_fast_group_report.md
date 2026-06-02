# RM-65 网球机器人 MPC+iLQR 快速版方法框架

> 面向组会汇报的进展说明文档  
> 主要代码文件：`scripts/rm65_mpc_fast.py`  
> 相关核心模块：`src/ilqt/cost.py`、`src/ilqt/solver.py`、`src/cpp/solver_cpp.py`、`src/tennis/hitting.py`、`src/tennis/ball.py`

## 1. 当前工作目标

当前阶段的目标是：在 MuJoCo 中让 RM-65 双臂机器人右臂挥拍击打来球，并在有限计算时间内给出可执行的关节力矩轨迹。

这个问题的难点主要有三类：

1. 网球来球速度快，击打时间窗口很短，需要同时对准位置、速度和拍面姿态。
2. 机械臂是强非线性系统，直接优化长时域轨迹计算量较大。
3. 击球是接触事件，规划阶段要避免碰撞干扰，执行阶段又必须打开碰撞完成击球。

因此当前实现采用的是：

```text
来球轨迹预测 + 击打点搜索
→ 后摆 warm-start
→ MPC 滚动重规划
→ 每次重规划内部用少量 iLQR 迭代优化
→ 接近击打时开启球拍碰撞并评估击球效果
```

一句话概括：  
**先用物理预测找到球会在哪里被打，再用带后摆初值的 MPC+iLQR 不断修正机械臂挥拍轨迹。**

## 2. 整体 Pipeline

`scripts/rm65_mpc_fast.py` 的主流程可以分为八步：

```text
加载配置和命令行参数
  ↓
初始化 RM-65 MuJoCo 环境
  ↓
生成一条可击打的来球轨迹
  ↓
用 MuJoCo 预测球轨迹并搜索最佳击打点
  ↓
计算期望击打速度、拍面法向量和随挥目标
  ↓
生成后摆 warm-start 或雅可比转置初始控制
  ↓
进入 MPC 主循环：反复重规划、执行控制、更新球和机械臂状态
  ↓
击打后继续仿真、统计误差、可选 viewer 回放
```

### 2.1 配置与初始状态

脚本会加载 `configs/default.yaml`，如果存在 `configs/mpc.yaml` 也会合并进去。随后快速版在 `main()` 中设置了一组专用参数。

当前源码中的主要参数如下：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `dt` | 0.005 s | MuJoCo 仿真步长 |
| `total_horizon` | 200 | MPC 总仿真步数，约 1 秒 |
| `replan_interval` | 20 | 每 20 步重规划一次，约 100 ms |
| `horizon_plan` | `min(k_hit, 40)` | 每次 iLQR 的短horizon，最多 40 步 |
| `max_iter_per_plan` | 5 | 常规重规划 iLQR 迭代次数 |
| `first_plan_iters` | 12 | 首次规划迭代次数，保留线搜索 |
| `near_plan_iters` | 5 | 接近击打时迭代次数 |
| `workspace_radius` | 0.90 m | 以右肩为中心的可达球空间半径 |
| `hit_shift` | 0.01 m | iLQR 终端目标沿击打方向前移，形成随挥 |
| `normal_weight` | 500000 | 拍面法向量终端代价权重 |

命令行可覆盖部分行为，例如：

```bash
python scripts/rm65_mpc_fast.py --viewer --seed 42
python scripts/rm65_mpc_fast.py --fd
python scripts/rm65_mpc_fast.py --no-backswing
python scripts/rm65_mpc_fast.py --fix-joint5
python scripts/rm65_mpc_fast.py --replan-interval 10
```

右臂初始关节角当前为：

```text
q0 = [-1.5, 1.57, -0.236, 0.404, 0.446, 2.45]
```

左臂使用镜像姿态，主要用于完整双臂模型展示；当前优化和控制针对右臂 6 自由度。

### 2.2 来球轨迹生成

来球由 `generate_ball_to_target_box()` 生成。脚本先定义一个目标立方体：

```text
target_center = [-0.82765693, -0.47411682, 0.86947444]
target_offset = 0.2
```

然后在这个立方体内采样期望击打点，并反推球的初始位置和速度，使球在给定时间附近进入工作空间。

当前来球设置为从 `-Y` 方向飞来：

```text
ball_direction = "y"
ball_start_y_range = (-5.5, -4.5)
ball_start_z_range = (1.4, 1.8)
```

如果命令行传入 `--ball-speed`，则会按指定到达速度反推起始状态；否则随机生成一条可达来球。

### 2.3 击打点搜索

击打点由 `find_hitting_point_physics()` 搜索。它不是用纯解析抛物线，而是调用环境的 `predict_ball_trajectory()`，用 MuJoCo 物理前向仿真球轨迹。

每个候选点要满足：

```text
dist = ||p_ball - p_shoulder|| < workspace_radius
p_ball.z > 0.3
-0.60 < p_ball.z - p_shoulder.z < 0.55
```

候选点评分为：

```text
score = dist + height_penalty - front_bonus
```

其中：

```text
height_penalty = max(0, p_ball.z - p_shoulder.z - 0.2)^2 * 5.0
front_bonus    = max(0, p_ball.x - p_shoulder.x) * 0.3
```

这表示算法偏好：

1. 离肩部更近的球；
2. 不要太高的球；
3. 更靠机器人前方的球，因为前方更容易挥拍。

输出包括：

```text
k_hit       击打还剩多少仿真步
p_hit       目标击打位置
v_ball_hit  击打时刻球速
dist        击打点到肩关节距离
```

### 2.4 期望击打速度与拍面方向

期望球拍速度由 `compute_desired_hit_velocity()` 计算：

```text
v_hit_desired = normalize(hit_direction) * racket_speed
```

其中 `hit_direction` 和 `racket_speed` 来自配置文件。

拍面法向量目标设为来球速度反方向：

```text
n_des = -v_ball_hit / ||v_ball_hit||
```

这样做的直观意义是：**球拍面朝向来球方向，避免用拍框或斜面擦到球。**

如果发现模型中球拍法向定义相反，可以通过 `--normal-flip` 翻转目标方向。

### 2.5 随挥目标

脚本没有直接把 iLQR 终端位置设成球的位置，而是做了一个很小的随挥偏移：

```text
d_hat = normalize(hit_direction)
p_follow = p_hit + hit_shift * d_hat
```

默认 `hit_shift = 0.01 m`。  
这样优化目标略微穿过球的位置，而不是停在球的位置，有利于形成真实的挥拍动作。

需要注意：评估误差时仍然用真实击打点 `p_hit`，随挥点主要用于引导轨迹和击球速度。

## 3. Warm-start 设计

iLQR 是局部优化方法，对初始控制序列比较敏感。当前脚本默认使用后摆 warm-start，而不是零力矩或常数力矩。

### 3.1 后摆 warm-start

函数入口为 `generate_backswing_warm_start()`。

核心步骤：

1. 对随挥目标 `p_follow` 求 IK，得到击打姿态 `q_hit`。
2. 根据拍面目标法向量 `n_des` 微调腕关节，使球拍面朝向来球。
3. 用位置雅可比伪逆计算击打时刻关节速度：

```text
qdot_hit = J_p^+ v_hit_desired
```

4. 对关节 1 生成“后摆→前挥”的五次多项式轨迹。
5. 其余关节从当前角度线性插值到击打姿态。
6. 用关节空间 PD 控制跟踪这条期望轨迹，得到初始力矩序列 `U_warm`。

后摆轨迹的关键是关节 1：

```text
q1(0)        = 当前角度
q1'(0)       = 当前速度
q1(alpha*T) = 当前角度 + backswing_offset
q1'(alpha*T)= 0
q1(T)        = 击打角度
q1'(T)       = 击打速度
```

用五次多项式满足以上 6 个边界条件，可以得到一条连续、平滑、先后摆再前挥的轨迹。

### 3.2 自适应后摆幅度

脚本会根据初始球拍到击打点的距离调整后摆幅度：

```text
dist_to_ball: 0.8 m → 1.5 m
backswing:    0.4 rad → 1.0 rad
```

直观上，球越远，需要的摆幅越大，后摆也越明显。

### 3.3 上次轨迹复用

MPC 每次重规划时，不一定重新生成后摆。当前策略是：

```text
如果 U_prev 长度足够：
    对上次剩余控制序列线性重采样，作为新的 warm-start
否则：
    重新生成后摆 warm-start
```

这可以减少轨迹抖动，也可以节省部分计算。

## 4. 代价函数设定

代价函数由 `src/ilqt/cost.py` 中的 `HittingCost` 实现。整体形式为：

```text
J(X, U) = sum_{k=0}^{N-1} l_k(x_k, u_k) + l_N(x_N)
```

其中：

```text
x = [q, qdot] ∈ R^12
u = tau ∈ R^6
```

### 4.1 终端代价

终端代价负责让球拍在击打时刻满足三个目标：

1. 位置到达击打点或随挥点；
2. 末端速度接近期望击打速度；
3. 拍面法向量朝向来球。

数学形式可以写成：

```text
l_N(x_N)
= 0.5 * (p_ee - p_target)^T Q_p (p_ee - p_target)
 + 0.5 * (v_ee - v_hit)^T Q_v (v_ee - v_hit)
 + 0.5 * Q_n * ||n_ee - n_des||^2
```

当前脚本中：

```text
p_target = p_follow = p_hit + hit_shift * normalize(hit_direction)
v_hit    = normalize(hit_direction) * racket_speed
n_des    = -normalize(v_ball_hit)
```

基础权重来自 `configs/default.yaml`，脚本中又乘了 2：

```text
Q_p = diag([50000, 50000, 50000]) * 2
Q_v = diag([200, 200, 200]) * 2
R   = 0.0001
```

### 4.2 拍面法向量代价

拍面法向量代价默认权重很高：

```text
Q_n = 500000
```

它的作用不是让球拍到达某个姿态角，而是直接约束球拍面法向量：

```text
minimize ||n_ee - n_des||^2
```

导数使用旋转雅可比构造：

```text
dn/dq ≈ skew(-n) * J_omega
```

这样比对姿态角做有限差分更稳定，也更符合“击球时拍面朝向”的任务需求。

### 4.3 运行代价

运行代价主要约束控制力矩：

```text
l_k(x_k, u_k) = 0.5 * u_k^T R_k u_k
```

`HittingCost` 还支持两个可选运行项：

```text
运行位置代价：0.5 * (p_ee - p_running)^T Q_run (p_ee - p_running)
关节轨迹代价：0.5 * sum_j Q_joint_j * (q_j - q_des_j)^2
```

但在 `rm65_mpc_fast.py` 当前主流程中：

```text
Q_p_running = 0.0
Q_joint = None
```

也就是说，后摆轨迹主要通过 warm-start 引导，当前没有额外加入关节轨迹跟踪代价。

### 4.4 控制代价退火

快速版中一个重要设置是 `R_schedule`。  
前段保持正常控制惩罚，最后一段逐渐降低控制惩罚：

```text
前 (1 - r_decay_ratio) 段：R_k = R
后 r_decay_ratio 段：R_k 从 R 衰减到 0
```

默认：

```text
r_decay_ratio = 0.30
```

也就是最后 30% 时间逐步放开力矩惩罚。

关节 1 是主要挥拍关节，额外使用更快的指数衰减：

```text
R_joint1 = R * (1 - s)^10
```

这样在临近击打时，关节 1 可以更积极地加速，形成更大的挥拍速度。

### 4.5 权重调度

MPC 重规划时会根据当前球拍到目标击打点的位置误差调整权重：

```text
如果 pos_err_now > 0.10 m:
    Q_p_scale = 5.0
    Q_v_scale = 3.0
否则:
    Q_v_scale 从 50.0 过渡到 3.0
```

当前位置误差较大时，优化重点仍然是尽快到达击打点；  
当位置误差较小时，速度权重增大，使球拍在击打瞬间有更合适的速度方向和大小。

当前 `Q_p_scale_far` 和 `Q_p_scale_near` 都是 5.0，因此位置权重基本恒定；主要变化来自速度权重 `Q_v_scale`。

## 5. MPC 方法说明

MPC，即模型预测控制，核心思想是：

```text
每次只优化未来一小段轨迹
实际只执行最前面的几步
下一次根据最新状态重新优化
```

在本项目中，MPC 解决的是“来球和机械臂状态都在不断变化”的问题。即使第一次预测不完美，后续也可以滚动修正。

### 5.1 主循环结构

主循环每个仿真步执行：

```text
读取当前球状态
判断是否需要重规划
如果需要：
    重新搜索击打点
    更新代价函数目标和权重
    生成或复用 warm-start
    运行少量 iLQR 迭代
    保存本轮要执行的控制 buffer
取出 buffer 中的一步控制力矩
执行 env.step_full(u_cmd)
记录轨迹和误差
接近击打时打开碰撞检测
```

控制 buffer 的作用是避免每一步都重规划：

```text
U_buffer = U_mpc[:replan_interval]
每个仿真步执行 U_buffer 中的一步
每 20 步或 buffer 用完后重新规划
```

### 5.2 重规划触发条件

当前触发条件是：

```text
need_replan = (step % replan_interval == 0)
           or (step == 0)
           or (buffer_idx >= len(U_buffer))
```

默认 `replan_interval = 20`，相当于每 100 ms 重规划一次。  
这比原先每 10 步重规划更快，代价是闭环修正频率降低。

### 5.3 重规划时重新估计击打点

每次重规划都会基于当前球位置和速度再次调用 `find_hitting_point_physics()`。

为了避免球拍碰撞或数值扰动导致 `k_hit` 突然异常变小，脚本加入了保护逻辑：

```text
如果新的 k_hit 小于 max(10, 上次 k_hit 的 1/4)，且上次 k_hit > 30：
    使用 k_hit_new = max(1, k_hit_old - replan_interval)
```

这相当于给击打时间加了一个简单的物理一致性约束：剩余步数应该大致随时间减少，而不是突然跳到很小。

### 5.4 远距与近距分支

重规划后分两类情况：

```text
如果 k_hit_new > far_threshold:
    使用雅可比转置控制器做粗跟踪
否则:
    进入 iLQR 优化
```

其中 `far_threshold = k_hit_total`。  
在常见情况下，随着时间推进 `k_hit_new` 会小于初始击打步数，所以主要会进入 iLQR 分支。远距分支更多是防止重新预测后击打时刻被推迟。

### 5.5 iLQR 短地平线

进入 iLQR 分支后：

```text
horizon_full = k_hit_new
horizon_plan = min(horizon_full, 40)
```

这里有一个重要工程折中：

1. 后摆 warm-start 仍按完整剩余击打时间生成，保证动作节奏正确。
2. 真正交给 iLQR 优化的地平线最多 40 步，保证计算速度。

这也是快速版的核心：**用完整时序生成初值，用短地平线做局部优化。**

### 5.6 快速版迭代策略

当前源码中的迭代策略为：

```text
首次规划：first_plan_iters = 8，并启用线搜索
常规规划：max_iter_per_plan = 5，跳过线搜索
近距规划：near_plan_iters = 5，跳过线搜索
```

首次规划保留线搜索，是为了在初始轨迹还不稳定时保证收敛。  
后续 MPC 依赖不断重规划纠错，因此用固定步长前向传递，省去多次线搜索的计算开销。

### 5.7 控制执行与碰撞窗口

规划阶段一般关闭球拍碰撞，避免轨迹优化时球被提前碰飞。  
执行阶段当 `k_hit_new <= 10` 时打开碰撞：

```text
enable_collision = (k_hit_new <= 10)
```

检测到 `ball` 和 `racket` 接触后，脚本会计算一次手动弹性反弹：

```text
v_ball_new = v_ball_pre - (1 + e) * dot(v_ball_pre - v_ee, n_hat) * n_hat
e = 0.8
```

这样做是因为 MuJoCo 默认接触响应在高速小球场景下可能偏弱；手动反弹可以更清楚地展示击打效果。

## 6. iLQR 方法说明

iLQR 是 MPC 每次重规划内部调用的局部最优控制算法。

它处理的问题是：

```text
给定当前状态 x0 和初始控制序列 U_init
寻找一条更优的控制序列 U，使总代价 J(X, U) 降低
```

系统动力学写成：

```text
x_{k+1} = f(x_k, u_k)
```

在名义轨迹附近线性化：

```text
δx_{k+1} ≈ A_k δx_k + B_k δu_k
```

### 6.1 iLQR 单次迭代

一次 iLQR 迭代可以分成四步：

```text
1. Rollout
   用当前 U 在 MuJoCo 中前向仿真，得到名义轨迹 X

2. Linearization
   沿 X, U 计算每一步的 A_k, B_k

3. Backward pass
   从终端时刻向前递推值函数，求反馈增益 K_k 和前馈修正 k_k

4. Forward pass
   用新控制律重新 rollout，得到更新后的 X_new, U_new
```

### 6.2 动力学线性化

脚本默认优先使用解析线性化：

```python
use_analytical = not args.fd
```

如果 `--fd` 被指定，则使用有限差分线性化。

解析线性化基于机械臂动力学：

```text
M(q) qddot + h(q, qdot) = tau
```

连续状态方程：

```text
d/dt [q]     = qdot
d/dt [qdot] = M^{-1}(tau - h)
```

再离散化为：

```text
A = I + A_c * dt
B = B_c * dt
```

当 `src.cpp.iLQR_Core` 可用时，`src/cpp/solver_cpp.py` 会自动使用 C++ 加速解析线性化；否则回退到 Python 实现。

### 6.3 后向传递

后向传递从终端代价导数开始：

```text
V_x  = l_x_N
V_xx = l_xx_N
```

每一步构造局部 Q 函数导数：

```text
Q_x  = l_x  + A^T V_x
Q_u  = l_u  + B^T V_x
Q_xx = l_xx + A^T V_xx A
Q_ux = l_ux + B^T V_xx A
Q_uu = l_uu + B^T V_xx B
```

为了保证数值稳定，对 `Q_uu` 加正则：

```text
Q_uu_reg = Q_uu + mu * I
```

然后求得控制律：

```text
K_k = -Q_uu_reg^{-1} Q_ux
k_k = -Q_uu_reg^{-1} Q_u
```

其中：

```text
K_k 是状态反馈增益
k_k 是前馈控制修正
```

### 6.4 前向传递

前向传递使用：

```text
u_new_k = u_k + alpha * k_k + K_k * (x_new_k - x_k)
```

然后对控制力矩做 MuJoCo actuator 范围裁剪：

```text
u_new_k = clip(u_new_k, ctrl_lo, ctrl_hi)
```

在完整 iLQR 中，`alpha` 会通过线搜索从多个候选中选最能降低代价的步长。  
在快速 MPC 模式中，除首次规划外，默认使用固定：

```text
alpha = 0.5
```

并直接接受有限的新轨迹。这是为了换取实时性。

## 7. 快速版的主要工程取舍

相比更完整的 MPC+iLQR 版本，`rm65_mpc_fast.py` 的优化重点是减少单次运行时间。

| 设计 | 作用 | 代价 |
|---|---|---|
| 重规划间隔从 10 步增大到 20 步 | 减少 iLQR 调用次数 | 闭环修正频率下降 |
| iLQR horizon 封顶 40 步 | 限制线性化和后向传递规模 | 远期动作主要依赖 warm-start |
| 常规规划跳过线搜索 | 显著减少前向传递次数 | 单次更新不保证严格降代价 |
| 复用上一轮控制序列 | 提高连续性，减少初始化成本 | 初值可能继承上一轮偏差 |
| R 末段退火 | 临近击打时允许更激进挥拍 | 力矩更大，对模型稳定性要求更高 |
| 规划期关闭碰撞，击打窗口开启碰撞 | 避免球提前被碰飞 | 接触时刻依赖窗口设置 |

这个版本的核心思想可以概括为：

```text
用 warm-start 保证动作大方向，
用短时域 iLQR 做局部修正，
用 MPC 滚动更新弥补预测误差，
用工程限幅和碰撞窗口保证仿真可控。
```

## 8. 当前可以汇报的阶段性进展

当前代码已经形成了完整的击球闭环：

1. 能随机生成进入工作空间的来球。
2. 能用 MuJoCo 物理预测搜索击打点。
3. 能根据击打点生成后摆 warm-start。
4. 能在 MPC 循环中持续更新击打目标和代价权重。
5. 能调用 iLQR 优化短时域关节力矩序列。
6. 能在接近击打时打开球拍碰撞，并统计击打误差、击打后球速、实时比率。
7. 可通过 `--viewer` 进行真实速度回放。

汇报时可以把当前系统定位为：

```text
已经不是单纯离线轨迹优化，
而是一个具备来球预测、击打点选择、滚动优化和接触执行的闭环仿真框架。
```

## 9. 当前实现中的注意点

有几个实现细节在组会中如果被问到，可以这样解释：

1. `--horizon` 当前只覆盖了 `fixed_horizon` 变量，但主循环实际使用 `horizon_plan = min(k_hit_new, 40)`，因此快速版真正生效的是 40 步封顶策略。
2. `schedule_mpc_weights()` 函数保留在文件中，但当前主循环实际采用的是基于实时位置误差 `pos_err_now` 的权重调度。
3. `Q_joint` 当前设为 `None`，所以后摆轨迹主要通过 warm-start 影响 iLQR，不作为强运行代价约束。
4. `mpc_cfg = config.get("mpc", {})` 已读取配置，但快速版的关键 MPC 参数主要在脚本内显式设置，便于快速实验。
5. 文件开头注释中部分迭代次数描述可能是早期参数，汇报和复现实验时应以 `main()` 中当前源码参数为准。

## 10. 后续改进方向

下一步可以从三条线继续推进：

1. **算法稳定性**：把 `fixed_horizon` 和配置文件真正接入快速版主循环，统一参数来源；对 `Q_p/Q_v/R` 做系统消融实验。
2. **击球物理真实性**：进一步调 MuJoCo 接触参数，减少手动弹性反弹的比例，使击球效果更多来自物理引擎。
3. **闭环能力**：加入视觉或状态估计噪声，测试 MPC 对来球预测误差的鲁棒性。

如果只用一页 PPT 总结，可以写成：

```text
本阶段实现了 RM-65 网球击打的快速 MPC+iLQR 框架：
通过 MuJoCo 预测来球并选取可达击打点，
用后摆 warm-start 给 iLQR 提供挥拍初值，
在 MPC 中每 20 步滚动重规划，
每次只优化最多 40 步短时域轨迹，
并通过终端位置、速度、拍面法向量和控制退火代价共同形成击打动作。
当前系统已能完成从来球生成、击打点搜索、轨迹优化到碰撞回放的完整仿真闭环。
```
