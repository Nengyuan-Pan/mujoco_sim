"""MuJoCo 模型安全加载工具。

MuJoCo 底层 C 函数 mj_loadXML 在 Windows 上不支持路径中包含非 ASCII 字符（如中文）。
本模块提供 load_mujoco_model() 函数，自动检测并处理该问题：
- Windows + 非 ASCII 路径 → 复制到临时 ASCII 目录后加载
- 其他情况 → 直接加载

同时处理 XML 文件中引用的相对路径资源（mesh、texture 等）。
"""

import atexit
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)

# 模块级缓存：同一进程只复制一次
_cached_tmp_dir: Path | None = None
_cached_src_hash: str | None = None


def _is_ascii_path(path: Path) -> bool:
    """检查路径是否仅包含 ASCII 字符。"""
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _compute_dir_signature(project_root: Path) -> str:
    """计算 src/robot/ 和 assets/ 目录的签名，用于判断缓存是否有效。"""
    import hashlib

    h = hashlib.md5()
    for pattern in ("src/robot/*.xml", "assets/rm_65/urdf/meshes/*.STL"):
        for f in sorted(project_root.glob(pattern)):
            h.update(f.name.encode())
            h.update(str(f.stat().st_mtime).encode())
    return h.hexdigest()[:12]


def _setup_temp_copy(project_root: Path) -> Path:
    """将 src/robot/ 和 assets/ 复制到临时 ASCII 目录，返回临时目录路径。

    使用模块级缓存避免重复复制。
    """
    global _cached_tmp_dir, _cached_src_hash

    sig = _compute_dir_signature(project_root)
    if _cached_tmp_dir is not None and _cached_src_hash == sig:
        logger.debug("复用缓存的临时模型目录: %s", _cached_tmp_dir)
        return _cached_tmp_dir

    if _cached_tmp_dir is not None:
        shutil.rmtree(_cached_tmp_dir, ignore_errors=True)

    tmp = Path(tempfile.mkdtemp(prefix="mj_model_"))
    logger.info("复制模型文件到临时目录: %s", tmp)

    robot_dst = tmp / "src" / "robot"
    robot_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root / "src" / "robot", robot_dst)

    assets_src = project_root / "assets"
    if assets_src.exists():
        shutil.copytree(assets_src, tmp / "assets")

    _cached_tmp_dir = tmp
    _cached_src_hash = sig

    atexit.register(_cleanup_temp)
    return tmp


def _cleanup_temp() -> None:
    """进程退出时清理临时目录。"""
    global _cached_tmp_dir
    if _cached_tmp_dir is not None:
        shutil.rmtree(_cached_tmp_dir, ignore_errors=True)
        _cached_tmp_dir = None


def load_mujoco_model(model_path: Path) -> mujoco.MjModel:
    """安全加载 MuJoCo 模型，自动处理 Windows 非 ASCII 路径问题。

    当在 Windows 上且路径包含非 ASCII 字符时，将 src/robot/ 和 assets/
    复制到临时 ASCII 目录后加载模型。其他情况直接使用原始路径。

    Args:
        model_path: MuJoCo XML 模型文件路径。

    Returns:
        加载完成的 MjModel 对象。
    """
    path_str = str(model_path)

    # 非 Windows 或纯 ASCII 路径：直接加载
    if sys.platform != "win32" or _is_ascii_path(model_path):
        return mujoco.MjModel.from_xml_path(path_str)

    # Windows + 非 ASCII 路径：需要复制到临时目录
    # 推断项目根目录（model_path 通常为 .../mujoco_sim/src/robot/xxx.xml）
    # 支持也放在 scripts/ 下直接调用的场景
    project_root = _infer_project_root(model_path)
    tmp = _setup_temp_copy(project_root)

    # 根据原始路径中的相对位置计算临时路径
    try:
        rel = model_path.relative_to(project_root)
        tmp_xml = tmp / rel
    except ValueError:
        # model_path 不在 project_root 下（不太可能），尝试直接用文件名
        tmp_xml = tmp / "src" / "robot" / model_path.name

    if not tmp_xml.exists():
        raise FileNotFoundError(
            f"临时目录中未找到模型文件: {tmp_xml} "
            f"(原始路径: {model_path})"
        )

    logger.info("从临时路径加载模型: %s (原路径: %s)", tmp_xml, model_path)
    return mujoco.MjModel.from_xml_path(str(tmp_xml))


def _infer_project_root(model_path: Path) -> Path:
    """从模型文件路径推断项目根目录。

    假设目录结构为:
        mujoco_sim/src/robot/xxx.xml  → mujoco_sim/
        mujoco_sim/assets/...         → mujoco_sim/

    Args:
        model_path: 模型文件路径。

    Returns:
        推断的项目根目录路径。
    """
    resolved = model_path.resolve()

    # 向上查找包含 src/robot/ 的目录
    for parent in resolved.parents:
        if (parent / "src" / "robot").is_dir() and (parent / "assets").is_dir():
            return parent

    # 回退：假设模型文件在 src/robot/ 下，往上两级
    return model_path.parent.parent.parent
