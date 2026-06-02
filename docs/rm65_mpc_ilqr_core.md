# RM-65 MPC+iLQR 核心机制详解

> 以 `scripts/rm65_mpc_ilqr_5_5.py` 为基础

本文档聚焦三个核心模块：

1. **代价函数设计**：引导 iLQR 优化的目标；
2. **iLQR 求解器使用方式**：求解器初始化、调用模式、收敛策略；
3. **MPC 主循环**：逐步执行流程、重规划逻辑、远近距离分支。

---

# 一、代价函数设计（`HittingCost`）

源码位置：

```text
src/ilqt/cost.py
```

---

## 1.1 数学形式

总代价由运行代价和终端代价组成：

```math
J(X,U)=\sum_{k=0}^{N-1} l_k(x_k,u_k)+l_N(x_N)
```

其中：

* $X={x_0,x_1,\dots,x_N}$ 为状态轨迹；
* $U={u_0,u_1,\dots,u_{N-1}}$ 为控制序列。

---

## 1.1.1 终端代价 $l_N(x_N)$：击打精度

终端代价约束：

* 球拍位置；
* 球拍速度；
* 拍面法向量。

```math
\begin{aligned}
l_N(x_N)
=&\frac{1}{2}
\left(\mathbf{p}_{ee}-\mathbf{p}_{hit}\right)^{\mathsf T}
\mathbf{Q}_p
\left(\mathbf{p}_{ee}-\mathbf{p}_{hit}\right)
\\
&+\frac{1}{2}
\left(\mathbf{v}_{ee}-\mathbf{v}_{hit}^{des}\right)^{\mathsf T}
\mathbf{Q}_v
\left(\mathbf{v}_{ee}-\mathbf{v}_{hit}^{des}\right)
\\
&+\frac{1}{2}Q_n
\left\|
\mathbf{n}-\mathbf{n}_{des}
\right\|_2^2
\end{aligned}
```

---

### 变量说明

| 符号                       | 维度 | 含义       |
| ------------------------ | -: | -------- |
| $\mathbf{p}_{ee}$        |  3 | 球拍中心世界坐标 |
| $\mathbf{p}_{hit}$       |  3 | 目标击打位置   |
| $\mathbf{v}_{ee}$        |  3 | 球拍中心速度   |
| $\mathbf{v}_{hit}^{des}$ |  3 | 期望击打速度   |
| $\mathbf{n}$             |  3 | 拍面法向量    |
| $\mathbf{n}_{des}$       |  3 | 目标法向量    |
| $Q_n$                    | 标量 | 法向量权重    |

---

### 期望拍面法向量

```math
\mathbf{n}_{des}
=
-\frac{\mathbf{v}_{ball}^{hit}}
{\left\|\mathbf{v}_{ball}^{hit}\right\|_2}
```

即：

* 拍面朝向来球方向；
* 保证稳定碰撞。

---

### Gauss-Newton 雅可比近似

```math
\mathbf{J}_h
\approx
\begin{bmatrix}
\mathbf{J}_p & \mathbf{0}
\\
\mathbf{0} & \mathbf{J}_p
\end{bmatrix}
\in \mathbb{R}^{6\times 12}
```

```math
\mathbf{J}_n
=
\operatorname{skew}(-\mathbf{n})
\mathbf{J}_{\omega}
```

其中：

* $\mathbf{J}_p$：位置雅可比；
* $\mathbf{J}_{\omega}$：角速度雅可比。

---

## 1.1.2 运行代价 $l_k(x_k,u_k)$：控制平滑性

运行代价用于：

* 控制输入平滑；
* 运行过程追球；
* 后摆关节轨迹跟踪。

```math
\begin{aligned}
l_k(x_k,u_k)
=&\frac{1}{2}
\mathbf{u}_k^{\mathsf T}
\mathbf{R}_k
\mathbf{u}_k
\\
&+\frac{1}{2}
\left(
\mathbf{p}_{ee}-\mathbf{p}_{ball}
\right)^{\mathsf T}
\mathbf{Q}_{p}^{run}
\left(
\mathbf{p}_{ee}-\mathbf{p}_{ball}
\right)
\\
&+\frac{1}{2}
\sum_{j=1}^{6}
Q_{joint,j}
\left(
q_j-q_{des,j}
\right)^2
\end{aligned}
```

---

## 1.1.3 默认权重配置

```math
\mathbf{Q}_p
=
2.0\times
\operatorname{diag}
(50000,50000,50000)
=
\operatorname{diag}
(100000,100000,100000)
```

```math
\mathbf{Q}_v
=
2.0\times
\operatorname{diag}
(200,200,200)
=
\operatorname{diag}
(400,400,400)
```

```math
R=10^{-4}
```

---

# 1.2 $\mathbf{Q}_p/\mathbf{Q}_v$ 权重调度

根据末端位置误差：

```python
if pos_err > 0.10:
    Q_p_scale = 5.0
    Q_v_scale = 3.0

elif pos_err < 0.05:
    Q_p_scale = 5.0
    Q_v_scale = 50.0

else:
    # 线性插值
```

---

### 设计意图

远距离：

* 优先靠近击打点；
* 速度不重要。

近距离：

* 速度精度最重要；
* 强制调准击打速度。

---

# 1.3 $\mathbf{R}$ 退火（控制代价衰减）

时间步：

```math
k=0,1,\dots,N-1
```

退火起点：

```math
k_{decay}=0.7N
```

---

### 前 70% 时间

```math
\mathbf{R}_k=\mathbf{R}_{base}
```

---

### 后 30% 时间

定义归一化进度：

```math
s=
\frac{k-k_{decay}}
{N-k_{decay}}
,\qquad
s\in[0,1]
```

其余关节：

```math
R_{other,k}
=
R_{base}(1-s)
```

关节 1：

```math
R_{joint1,k}
=
R_{base}(1-s)^{10}
```

---

### 设计意图

* 前期控制平滑；
* 终端允许大扭矩挥拍；
* shoulder_pan 单独快速释放。

---

# 1.4 随挥偏移（hit_shift）

挥拍方向：

```math
\hat{\mathbf{d}}
=
\frac{
\begin{bmatrix}
0\\
-1\\
0.3
\end{bmatrix}
}
{
\left\|
\begin{bmatrix}
0\\
-1\\
0.3
\end{bmatrix}
\right\|_2
}
```

随挥目标：

```math
\mathbf{p}_{follow}
=
\mathbf{p}_{hit}
+
0.01\hat{\mathbf{d}}
```

---

### 设计意图

目标略超过击打点：

* 鼓励随挥；
* 避免拍面停在球位。

---

# 1.5 拍面法向量代价

```math
\mathbf{n}_{des}
=
-\frac{\mathbf{v}_{ball}^{hit}}
{\left\|\mathbf{v}_{ball}^{hit}\right\|_2}
```

若启用：

```text
--normal-flip
```

则：

```math
\mathbf{n}_{des}^{flip}
=
-\mathbf{n}_{des}
```

---

# 二、iLQR 求解器使用方式

源码位置：

```text
src/ilqt/solver.py
```

---

# 2.1 初始化

```python
ilqt_cfg = dict(config["ilqt"])
solver = ILQTSolver(ilqt_cfg, use_analytical=True)
```

---

## 超参数

| 参数        |    值 |
| --------- | ---: |
| `mu_init` | 0.01 |
| `mu_min`  | 1e-6 |
| `mu_max`  | 1e10 |
| `delta_0` |  1.6 |
| `lin_eps` | 1e-6 |

---

# 2.2 MPC 中的调用方式

```python
X_mpc, U_mpc, iter_costs = solver.solve_few_iters(
    env,
    cost_fn,
    x_current,
    U_warm,
    max_iter=iters_plan,
    skip_linesearch=False,
)
```

---

# 2.3 单次 iLQR-MPC 迭代

---

## 1）前向 rollout

```math
X=
\operatorname{rollout}
(x_{current},U)
```

---

## 2）线性化动力学

```math
\delta x_{k+1}
=
A_k\delta x_k
+
B_k\delta u_k
```

---

## 3）计算代价导数

```math
l_x,\quad
l_u,\quad
l_{xx},\quad
l_{ux},\quad
l_{uu}
```

---

## 4）终端导数

```math
l_{x,N},\quad
l_{xx,N}
```

---

## 5）后向递推

```math
\delta u_k
=
k_k
+
K_k\delta x_k
```

---

## 6）前向更新

```math
u_k^{new}
=
u_k
+
\alpha k_k
+
K_k(x_k^{new}-x_k)
```

---

# 2.4 动力学线性化

连续时间：

```math
\delta \dot{x}
=
A_c\delta x
+
B_c\delta u
```

其中：

```math
A_c
=
\begin{bmatrix}
0 & I
\\
-M^{-1}H_q &
-M^{-1}H_{\dot q}
\end{bmatrix}
```

离散化：

```math
A
=
I+A_c\Delta t
```

```math
B
=
B_c\Delta t
```

---

# 2.5 rollout 期间关闭碰撞

```python
env.set_arm_collision(False)

for k in range(N):
    X[k+1] = env.step_from_state(X[k], U[k])

env.set_arm_collision(True)
```

---

# 2.6 输出利用

```python
U_buffer = U_mpc[:replan_interval]
U_prev   = U_mpc[replan_interval:]
```

形成：

```text
滚动时域 MPC
```

---

# 三、MPC 主循环

---

# 3.1 初始化

| 变量                | 含义       |
| ----------------- | -------- |
| `total_horizon`   | 总仿真步数    |
| `replan_interval` | 重规划间隔    |
| `U_buffer`        | 控制缓存     |
| `U_prev`          | 上轮剩余控制   |
| `k_hit_new`       | 当前剩余击打步数 |

---

# 3.2 每步流程

```text
1. 获取球状态
2. 判断是否重规划
3. 搜索击打点
4. 更新击打目标
5. 判断远距/近距
6. 远距：Jacobian transpose
7. 近距：iLQR
8. 执行一步控制
9. 碰撞检测
10. 更新状态
```

---

# 3.3 远距模式

条件：

```math
k_{hit}^{new}>k_{far}
```

使用：

```python
compute_jacobian_init_control()
```

不调用 iLQR。

---

# 3.4 近距模式

条件：

```math
k_{hit}^{new}\leq k_{far}
```

启动完整 iLQR。

---

## 位置误差

```math
e_p
=
\left\|
\mathbf{p}_{ee}
-
\mathbf{p}_{hit}^{new}
\right\|_2
```

---

## 更新目标

```python
cost_fn.update_target(
    p_follow,
    v_hit_desired,
    n_des,
)
```

---

## 迭代次数

| 条件             | 次数 |
| -------------- | -: |
| 首次规划           | 50 |
| near_threshold | 15 |
| 其他             |  8 |

---

# 3.5 碰撞处理

碰撞窗口：

```math
k_{hit}^{new}\leq10
```

---

## 相对法向速度

```math
v_{rel,n}
=
(
\mathbf{v}_{ball}^{pre}
-
\mathbf{v}_{ee}
)^{\mathsf T}
\mathbf{n}_{racket}
```

---

## 弹性碰撞

```math
\mathbf{v}_{ball}^{new}
=
\mathbf{v}_{ball}^{pre}
-
(1+e)
v_{rel,n}
\mathbf{n}_{racket}
```

其中：

```math
e=0.8
```

因此：

```math
\mathbf{v}_{ball}^{new}
=
\mathbf{v}_{ball}^{pre}
-
1.8
v_{rel,n}
\mathbf{n}_{racket}
```

---

# 四、Warm-start

---

# 4.1 Jacobian transpose

```math
u_k
=
J_p^{\mathsf T}
(p_{hit}-p_{ee})
\cdot
g\min(d,0.5)
-
2.0\dot q
```

---

# 4.2 后摆 warm-start

---

## IK 求解

```math
q_{hit}
=
IK(p_{hit})
```

---

## 击打关节速度

```math
\dot q_{hit}
=
J_p^{+}
v_{hit}^{des}
```

---

## 线性插值

```math
q_j(k)
=
(1-\lambda_k)q_{j,cur}
+
\lambda_k q_{j,hit}
```

---

# 4.3 控制序列重采样

```math
u_{new,k}
=
(1-\beta)u_{old,i}
+
\beta u_{old,i+1}
```

---

# 五、运行命令

---

## 标准运行

```bash
python scripts/rm65_mpc_ilqr_5_5.py \
    --seed 42 \
    --normal-flip \
    --viewer
```

---

## 调参

```bash
python scripts/rm65_mpc_ilqr_5_5.py \
    --seed 42 \
    --normal-flip \
    --backswing 0.8 \
    --bs-ratio 0.4 \
    --r-decay 0.35 \
    --hit-shift 0.02
```

---

## 无后摆

```bash
python scripts/rm65_mpc_ilqr_5_5.py \
    --seed 42 \
    --no-backswing \
    --viewer
```

---

## 禁用 R 退火

```bash
python scripts/rm65_mpc_ilqr_5_5.py \
    --seed 42 \
    --no-r-decay
```

---

## 有限差分线性化

```bash
python scripts/rm65_mpc_ilqr_5_5.py \
    --seed 42 \
    --fd
```
