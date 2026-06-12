"""异步重规划器：后台线程执行 iLQR，主线程永不阻塞。

架构：
  主线程（执行层，5ms/步）        后台线程（规划层）
  ──────────────────────        ──────────────
  读取球状态                      
  检查新规划就绪？                  
  ├─ 是: 切换 U_active            
  └─ 否: 继续旧 buffer            
  need_replan?                    
  ├─ 是: 提交请求 ──────→        拷贝球状态到 env_plan
  │   (非阻塞)                   执行完整规划流程
  │   继续执行 U_active          完成后写 U_pending
  │   ←─────── 通知就绪 ────     设 new_plan_ready
  u_cmd = U_active[idx++]
  safety_filter + env.step

关键设计：
  - 双缓冲：U_active（主线程消费）/ U_pending（后台写入）
  - 独立 MjData：后台线程使用独立 env_plan，避免竞态
  - 非阻塞提交：submit() 立即返回，后台线程异步执行
  - Buffer 耗尽保护：U_active 用完后 fallback 到 JT 控制
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.sim.rm65_env import RM65Env

logger = logging.getLogger(__name__)


@dataclass
class PlanRequest:
    """规划请求：主线程 → 后台线程。"""
    x_current: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    step: int
    k_hit_current: int
    U_prev: np.ndarray
    p_hit_current: np.ndarray
    v_hit_desired: np.ndarray
    n_des_current: np.ndarray
    is_first_plan: bool = False


@dataclass
class PlanResult:
    """规划结果：后台线程 → 主线程。"""
    U_buffer: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    U_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    U_mpc_full: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    k_hit_new: int = 0
    p_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_des_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solver_ok: bool = True
    iters_plan: int = 0
    horizon_plan: int = 0
    fast_lin: bool = False
    fp_limits_was_none: bool = False
    plan_duration_ms: float = 0.0
    ball_unreachable: bool = False
    request_step: int = -1
    hitting_tube_data: object = None


class AsyncReplanner:
    """异步重规划器：后台线程执行 iLQR 规划，主线程永不阻塞。

    使用双缓冲机制：
    - U_active：主线程正在消费的控制序列
    - U_pending：后台线程规划完成后写入的新序列
    主线程在合适的时机原子切换到新序列。

    Args:
        env: 主线程的 MuJoCo 环境（仅读取 model_path/dt）。
        replan_fn: 规划函数，签名为 (request, env_plan) -> PlanResult。
        config: 配置字典。
    """

    def __init__(
        self,
        env: RM65Env,
        replan_fn: "callable",
        config: dict | None = None,
        state: object | None = None,
        model_path: "Path | None" = None,
    ) -> None:
        self._replan_fn = replan_fn
        self._config = config or {}
        self._state = state
        self._env = env

        # 创建独立 MjData 的规划环境（共享 MjModel 只读，独立 MjData）
        self._model_path = model_path or self._find_model_path(env)
        self._dt = env.dt
        self.env_plan: RM65Env | None = None  # 延迟初始化（在后台线程中）

        # 线程同步
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # 请求/结果
        self._request: PlanRequest | None = None
        self._result: PlanResult | None = None
        self._has_new_plan = False
        self._is_planning = False

        # 统计
        self.submit_count = 0
        self.complete_count = 0
        self.buffer_exhaustion_count = 0
        self.replan_durations_ms: list[float] = []
        self.replan_k_hit_history: list[int] = []

    @staticmethod
    def _find_model_path(env: RM65Env) -> Path:
        """从 env 实例推导 model 文件路径。"""
        # MuJoCo 3.x: model.filedir 给出 XML 目录
        filedir = env.model.filedir
        # 查找目录下的 XML 文件
        import glob
        xml_files = glob.glob(str(Path(filedir) / "*.xml"))
        if xml_files:
            return Path(xml_files[0])
        # 回退：使用已知路径
        return Path("src/robot/rm65_model.xml")

    def start(self) -> None:
        """启动后台规划线程。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._planner_loop, daemon=True)
        self._thread.start()
        logger.info("AsyncReplanner: 后台规划线程已启动")

    def stop(self) -> None:
        """停止后台规划线程。"""
        self._stop_event.set()
        self._request_event.set()  # 唤醒等待中的线程
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("AsyncReplanner: 后台规划线程已停止")

    def submit(self, request: PlanRequest) -> bool:
        """非阻塞提交规划请求。

        如果后台线程正在规划，则跳过（避免堆积请求）。

        Returns:
            True 表示请求已提交，False 表示后台忙（已跳过）。
        """
        with self._lock:
            if self._is_planning:
                logger.debug(f"步 {request.step}: 后台正在规划，跳过本次请求")
                return False
            self._request = request
            self._is_planning = True
            self.submit_count += 1
        self._request_event.set()
        return True

    def has_new_plan(self) -> bool:
        """检查后台规划是否完成且有新结果。"""
        return self._has_new_plan

    def apply_new_plan(self) -> PlanResult | None:
        """取出规划结果并重置标志。

        Returns:
            PlanResult 或 None（如果没有新结果）。
        """
        if not self._has_new_plan:
            return None
        with self._lock:
            result = self._result
            self._result = None
            self._has_new_plan = False
        if result is not None:
            self.complete_count += 1
            self.replan_durations_ms.append(result.plan_duration_ms)
            self.replan_k_hit_history.append(result.k_hit_new)
        return result

    def is_planning(self) -> bool:
        """后台线程是否正在规划。"""
        with self._lock:
            return self._is_planning

    def _ensure_env_plan(self) -> RM65Env:
        """在后台线程中延迟创建规划环境（避免主线程资源竞争）。"""
        if self.env_plan is None:
            self.env_plan = RM65Env(self._model_path, dt=self._dt)
            if hasattr(self._env, 'init_q_left'):
                self.env_plan.init_q_left = self._env.init_q_left.copy()
            self._env.clone_actuator_config(self.env_plan)
            # 初始化 env_plan 的 qpos 到默认值并调用 mj_forward
            import mujoco
            mujoco.mj_forward(self.env_plan.model, self.env_plan.data)
            logger.info("AsyncReplanner: 规划环境已创建（独立 MjData）")
        return self.env_plan

    def _planner_loop(self) -> None:
        """后台线程主循环：等待请求 → 执行规划 → 通知主线程。"""
        self._ensure_env_plan()

        while not self._stop_event.is_set():
            # 阻塞等待规划请求
            self._request_event.wait(timeout=1.0)
            self._request_event.clear()

            if self._stop_event.is_set():
                break

            # 取出请求
            with self._lock:
                request = self._request
                self._request = None

            if request is None:
                continue

            t_start = time.perf_counter()

            try:
                result = self._replan_fn(request, self.env_plan, self._state, self._config)
            except Exception as e:
                logger.error(f"AsyncReplanner: 规划异常: {e}", exc_info=True)
                result = PlanResult(solver_ok=False, ball_unreachable=False)
            finally:
                t_dur = (time.perf_counter() - t_start) * 1000
                result.plan_duration_ms = t_dur

            # 写入结果，通知主线程
            with self._lock:
                self._result = result
                self._has_new_plan = True
                self._is_planning = False

            logger.info(
                f"ASYNC_PLAN done: step={request.step} k_hit={result.k_hit_new} "
                f"iters={result.iters_plan} horizon={result.horizon_plan} "
                f"t={t_dur:.0f}ms fast_lin={result.fast_lin} ok={result.solver_ok}"
            )
