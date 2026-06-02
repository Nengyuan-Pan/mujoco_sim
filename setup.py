"""pybind11 构建脚本：编译 C++ iLQR 加速模块。

用法：
    pip install -e .           # 开发模式安装
    python setup.py build_ext --inplace  # 仅编译，输出到当前目录

依赖：
    pip install pybind11 numpy
    需要 C++17 编译器（MSVC 2019+ / GCC 9+ / Clang 10+）
    MuJoCo C 库（已随 pip install mujoco 安装）

Windows 链接策略：
    MuJoCo pip 包只提供 mujoco.dll，没有 .lib 导入库。
    使用 /DELAYLOAD 在运行时加载 DLL，避免编译时需要 .lib 文件。
    需要将 mujoco.dll 所在目录加入 PATH 或在 Python 中 os.add_dll_directory()。
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np


def find_mujoco():
    """查找 MuJoCo 安装路径（返回 include_dir, package_dir）。"""
    import mujoco
    mujoco_dir = Path(mujoco.__file__).parent
    return str(mujoco_dir / "include"), str(mujoco_dir)


def find_pybind11():
    """查找 pybind11 头文件路径。"""
    import pybind11
    return str(Path(pybind11.__file__).parent / "include")


def _generate_mujoco_import_lib(mujoco_dir: str) -> str:
    """从 mujoco.dll 生成 mujoco.lib 导入库（仅 Windows 需要）。

    使用 MSVC 的 dumpbin + lib 工具从 DLL 导出表生成 .lib。
    如果已存在同名 .lib 则直接返回路径。
    """
    dll_path = Path(mujoco_dir) / "mujoco.dll"
    lib_path = Path(mujoco_dir) / "mujoco.lib"
    if lib_path.exists():
        return str(lib_path)

    # 尝试使用 MSVC 工具链生成 .lib
    import subprocess
    import tempfile

    # 查找 MSVC 工具路径
    vs_where = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not vs_where.exists():
        print("[setup] vswhere.exe 未找到，尝试直接链接 DLL")
        return str(dll_path)

    try:
        vs_path = subprocess.check_output(
            [str(vs_where), "-latest", "-property", "installationPath"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print("[setup] vswhere 查询失败，尝试直接链接 DLL")
        return str(dll_path)

    # 查找 lib.exe
    msvc_dir = Path(vs_path) / "VC" / "Tools" / "MSVC"
    if not msvc_dir.exists():
        return str(dll_path)

    msvc_versions = sorted(msvc_dir.iterdir(), reverse=True)
    if not msvc_versions:
        return str(dll_path)

    lib_exe = msvc_versions[0] / "bin" / "Hostx64" / "x64" / "lib.exe"
    if not lib_exe.exists():
        lib_exe = msvc_versions[0] / "bin" / "Hostx86" / "x64" / "lib.exe"
    if not lib_exe.exists():
        return str(dll_path)

    # 使用 dumpbin 获取导出函数，生成 .def 文件，再用 lib.exe 生成 .lib
    dumpbin = lib_exe.parent / "dumpbin.exe"
    if not dumpbin.exists():
        print("[setup] dumpbin.exe 未找到，尝试直接链接 DLL")
        return str(dll_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".def", delete=False, encoding="ascii") as f:
        f.write("LIBRARY mujoco\nEXPORTS\n")
        try:
            exports = subprocess.check_output(
                [str(dumpbin), "/EXPORTS", str(dll_path)],
                text=True,
            )
            for line in exports.splitlines():
                line = line.strip()
                # dumpbin 输出中导出函数行的格式：ordinal hint RVA name
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    func_name = parts[-1]
                    f.write(f"    {func_name}\n")
        except subprocess.CalledProcessError:
            pass
        def_path = f.name

    try:
        subprocess.check_call(
            [str(lib_exe), f"/def:{def_path}", f"/out:{lib_path}", "/machine:x64"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[setup] 已生成导入库: {lib_path}")
    except subprocess.CalledProcessError as e:
        print(f"[setup] 生成 .lib 失败: {e}")
    finally:
        os.unlink(def_path)

    if lib_path.exists():
        return str(lib_path)
    return str(dll_path)


class BuildExt(build_ext):
    """自定义 build_ext：处理 MuJoCo DLL 链接。"""

    def build_extensions(self):
        mujoco_inc, mujoco_dir = find_mujoco()
        is_win32 = sys.platform == "win32"

        for ext in self.extensions:
            ext.include_dirs.append(mujoco_inc)
            ext.library_dirs.append(mujoco_dir)
            ext.include_dirs.append(np.get_include())

            if is_win32:
                # Windows: 生成或查找 mujoco.lib
                lib_path = _generate_mujoco_import_lib(mujoco_dir)
                if lib_path.endswith(".lib"):
                    # 有 .lib，正常链接
                    ext.libraries = ["mujoco"]
                else:
                    # 没有 .lib，使用 delay-load 方式
                    ext.libraries = []
                    ext.extra_link_args.append(f"/DELAYLOAD:mujoco.dll")
                    ext.extra_link_args.append("delayimp.lib")
        super().build_extensions()


# 编译配置
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc", "/utf-8"]
elif sys.platform == "linux":
    extra_compile_args = ["-std=c++17", "-O3", "-march=native", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif sys.platform == "darwin":
    extra_compile_args = ["-std=c++17", "-O3"]
    extra_link_args = []

# 要编译的源文件
cpp_sources = [
    "src/cpp/core_ext.cpp",
]

setup(
    name="iLQR_Core",
    version="1.0.0",
    description="C++ accelerated iLQR hot-path (linearize + forward pass)",
    ext_modules=[
        Extension(
            "src.cpp.iLQR_Core",
            sources=cpp_sources,
            include_dirs=[
                "src/cpp",
                find_pybind11(),
            ],
            libraries=["mujoco"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++",
        ),
    ],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
