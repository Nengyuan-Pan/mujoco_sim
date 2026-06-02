"""pybind11 构建脚本：编译 C++ iLQR 加速模块。

用法：
    pip install -e .           # 开发模式安装
    python setup.py build_ext --inplace  # 仅编译，输出到当前目录

依赖：
    pip install pybind11 numpy
    需要 C++17 编译器（MSVC 2019+ / GCC 9+ / Clang 10+）
    MuJoCo C 库（已随 pip install mujoco 安装）
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

# 查找 MuJoCo 头文件和库
def find_mujoco():
    """Find MuJoCo install path (returns include_dir, package_dir)."""
    import mujoco
    mujoco_dir = Path(mujoco.__file__).parent
    return str(mujoco_dir / "include"), str(mujoco_dir)

# 查找 pybind11 头文件
def find_pybind11():
    import pybind11
    return str(Path(pybind11.__file__).parent / "include")

class BuildExt(build_ext):
    """Custom build to handle MuJoCo linking."""
    def build_extensions(self):
        mujoco_inc, mujoco_dir = find_mujoco()
        for ext in self.extensions:
            ext.include_dirs.append(mujoco_inc)
            ext.library_dirs.append(mujoco_dir)
            ext.include_dirs.append(np.get_include())
        super().build_extensions()

# 编译配置
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc", "/utf-8"]
    extra_link_args = ["/DLL"]
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
