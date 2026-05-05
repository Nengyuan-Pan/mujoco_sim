# Skill: 仿真运行（sim_run）

## 目的
在 MuJoCo 仿真中运行和验证网球击打场景。
调用时机：启动 iLQT 训练、评估轨迹、调试仿真问题时。

## 前提条件
- MuJoCo 已安装：`pip install mujoco`
- 机器人模型存在：`src/robot/model.xml`
- 配置文件存在：`configs/default.yaml`
- 依赖已安装：`pip install -r requirements.txt`

## 工作流程

### 步骤 1：运行 iLQT 优化
```bash
python scripts/train_ilqt.py --config configs/default.yaml
```

**执行流程：**
1. 加载 MuJoCo 模型和配置参数
2. 预测网球抛物线轨迹，计算期望击打点（位置 + 速度）
3. 用初始轨迹（零控制或直线插值）启动 iLQT
4. 迭代执行后向传递 + 前向传递（含线搜索）
5. 收敛后保存结果到 `results/<timestamp>/`

**关键输出：**
- `trajectory.npy` — 最优状态轨迹，形状 `(N+1, 12)`
- `controls.npy` — 最优控制序列，形状 `(N, 6)`
- `cost_history.npy` — 每次迭代总代价，形状 `(num_iters,)`
- `config_used.yaml` — 本次运行使用的配置副本

### 步骤 2：MuJoCo 仿真评估
```bash
python scripts/eval_sim.py --result-dir results/<timestamp>
```

**执行流程：**
1. 加载优化轨迹和控制序列
2. 在 MuJoCo 环境中逐步施加控制力矩
3. 对比实际仿真轨迹与优化轨迹的偏差
4. 打开 MuJoCo 查看器进行可视化回放
5. 计算并输出击打精度指标

**关键指标：**
- 末端位置误差：`||p_ee_final - p_hit||`
- 末端速度误差：`||v_ee_final - v_hit||`
- 轨迹跟踪误差：`mean(||x_sim - x_plan||)`
- 关节限位违反次数

### 步骤 3：绘制结果图表
```bash
python scripts/plot_results.py --result-dir results/<timestamp>
```

**输出图表：**
- 关节角度/角速度随时间变化
- 控制力矩随时间变化
- 代价函数收敛曲线
- 末端执行器 3D 轨迹
- 球拍-球接触时刻截图

## MuJoCo 环境封装设计

```python
class MujocoEnv:
    """MuJoCo 仿真环境封装类。"""

    def __init__(self, model_path: Path, config: dict) -> None:
        """加载模型并创建仿真数据。"""

    def reset(self, q0: np.ndarray | None = None) -> np.ndarray:
        """重置仿真状态，返回初始状态 x0。"""

    def step(self, u: np.ndarray) -> np.ndarray:
        """施加控制力矩 u，前进一步，返回新状态。"""

    def get_state(self) -> np.ndarray:
        """获取当前状态 x = [q, qdot]。"""

    def get_ee_pose(self) -> np.ndarray:
        """获取末端执行器（球拍中心）位置，形状 (3,)。"""

    def get_ee_velocity(self) -> np.ndarray:
        """获取末端执行器线速度，形状 (3,)。"""

    def get_jacobian(self) -> np.ndarray:
        """获取末端执行器雅可比矩阵，形状 (6, 6)。"""

    def render(self) -> None:
        """更新 MuJoCo 查看器画面。"""
```

## 跨平台注意事项

### Windows
- MuJoCo 查看器原生支持
- 使用 `mujoco.viewer.launch_passive()` 启动查看器
- 无需额外配置

### Ubuntu / Linux
- 同样使用 `mujoco.viewer.launch_passive()`
- 无头服务器（无显示器）需设置环境变量：
  ```bash
  export MUJOCO_GL=osmesa   # 或 egl
  ```
- **禁止使用** `glx` 或平台特定的渲染调用

### 通用
- 文件路径使用 `pathlib.Path`，不拼接字符串
- 不使用 `os.system()` 调用外部程序
- 不依赖平台特定的 shell 命令

## 常见问题与解决

| 问题                      | 原因                       | 解决方法                                |
|---------------------------|----------------------------|-----------------------------------------|
| iLQT 代价不下降           | 步长过大或正则化不足       | 减小 `alpha`，增大 `mu`（正则化参数）   |
| iLQT 代价发散             | 初始轨迹远离可行域         | 用更合理的初始轨迹，增大初始正则化      |
| 末端未到达击打点          | 代价权重不合理或时间不够   | 增大终端代价权重，增大时间步数 N        |
| MuJoCo 查看器打不开       | 无头环境未配置             | 设置 `MUJOCO_GL=osmesa`                 |
| 关节超限                  | 模型中未设关节限位         | 在 `model.xml` 中添加 `<limit>` 标签    |
| NaN 出现在轨迹中          | 数值不稳定                 | 增大正则化，检查初始状态是否合理        |
| 球拍-球未接触             | 击打点计算偏差             | 检查球轨迹预测，确认坐标系一致性        |

## 验证检查清单

### iLQT 优化阶段
- [ ] 代价函数单调下降（允许小幅波动）
- [ ] 最终终端代价满足精度要求
- [ ] 无 NaN 或 Inf 出现
- [ ] 正则化参数收敛到合理值

### 仿真评估阶段
- [ ] MuJoCo 回放轨迹与优化轨迹偏差小
- [ ] 末端执行器在击打时刻到达击打点附近
- [ ] 所有关节在限位范围内
- [ ] 控制力矩在合理范围内（无过大尖峰）
- [ ] 球拍-球在期望时刻接触

### 可视化阶段
- [ ] 关节轨迹平滑无跳变
- [ ] 末端执行器轨迹到达击打点
- [ ] 代价收敛曲线正常
- [ ] 图表保存到 `results/` 目录
