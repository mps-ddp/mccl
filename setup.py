"""
MCCL build script.

Builds the C++/Obj-C++ extension that provides the ProcessGroupMCCL backend.
Must be built on macOS with Apple Silicon and Xcode command-line tools.
"""
import os
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _torch_include_dirs():
    import torch
    return torch.utils.cpp_extension.include_paths()


def _torch_library_dirs():
    import torch
    return torch.utils.cpp_extension.library_paths()


class MCCLBuildExt(build_ext):
    """Custom build_ext that handles mixed .cpp/.mm compilation on macOS."""

    def build_extensions(self):
        if platform.system() != "Darwin":
            raise RuntimeError(
                "MCCL can only be built on macOS. "
                "This machine reports platform: " + platform.system()
            )

        arch = platform.machine()
        if arch not in ("arm64", "aarch64"):
            raise RuntimeError(
                f"MCCL requires Apple Silicon (arm64). Detected: {arch}"
            )

        sdk_path = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()

        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            ext.extra_link_args = ext.extra_link_args or []

            cpp_flags = [
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-fvisibility=hidden",
                "-DMCCL_BUILD",
                "-march=armv8.5-a+crc",
                "-isysroot", sdk_path,
            ]
            objcpp_flags = cpp_flags + ["-fobjc-arc"]

            ext._cpp_flags = cpp_flags
            ext._objcpp_flags = objcpp_flags

            ext.extra_link_args += [
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Accelerate",
                "-isysroot", sdk_path,
            ]

        super().build_extensions()

    def build_extension(self, ext):
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

        sources_cpp = [s for s in ext.sources if s.endswith(".cpp")]
        sources_mm = [s for s in ext.sources if s.endswith(".mm")]

        objects = []

        for src in sources_cpp:
            flags = ext._cpp_flags
            if src.endswith("Registration.cpp"):
                flags = [f for f in flags if f != "-fvisibility=hidden"]
            obj = self._compile_single(src, flags, ext)
            objects.append(obj)

        for src in sources_mm:
            obj = self._compile_single(src, ext._objcpp_flags, ext)
            objects.append(obj)

        ext.extra_link_args += [
            f"-L{torch_lib}",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            f"-Wl,-rpath,{torch_lib}",
        ]

        self._link_shared_object(objects, ext)

    def _compile_single(self, src, flags, ext):
        import torch

        obj = src + ".o"
        obj_path = os.path.join(self.build_temp, obj)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)

        include_flags = []
        for d in _torch_include_dirs() + (ext.include_dirs or []):
            include_flags += ["-I", d]

        cmd = ["clang++"] + flags + include_flags + ["-c", src, "-o", obj_path]

        self.announce(f"Compiling {src}", level=2)
        subprocess.check_call(cmd)
        return obj_path

    def _link_shared_object(self, objects, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        cmd = (
            ["clang++", "-shared", "-o", ext_path]
            + objects
            + (ext.extra_link_args or [])
        )

        self.announce(f"Linking {ext_path}", level=2)
        subprocess.check_call(cmd)

        self._compile_metallib(os.path.dirname(ext_path))

    @staticmethod
    def _detect_metal_std():
        """Pick the highest Metal language standard the host toolchain supports."""
        mac_ver = platform.mac_ver()[0]  # e.g. "15.1.1" or "14.7.2"
        if mac_ver:
            major = int(mac_ver.split(".")[0])
            if major >= 15:
                return "metal3.1"
        return "metal3.0"

    def _compile_metallib(self, output_dir):
        shader_src = "csrc/metal/shaders.metal"
        if not os.path.exists(shader_src):
            self.announce("shaders.metal not found, skipping metallib", level=2)
            return

        air_path = os.path.join(self.build_temp, "mccl_shaders.air")
        lib_path = os.path.join(output_dir, "mccl_shaders.metallib")
        os.makedirs(os.path.dirname(air_path), exist_ok=True)

        sdk_path = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True
        ).strip()

        metal_std = self._detect_metal_std()
        self.announce(
            f"Compiling {shader_src} -> .air (std={metal_std})", level=2
        )
        subprocess.check_call([
            "xcrun", "metal",
            "-c", shader_src,
            "-o", air_path,
            f"-std={metal_std}",
            "-isysroot", sdk_path,
        ])

        self.announce(f"Linking .air -> {lib_path}", level=2)
        subprocess.check_call([
            "xcrun", "metallib",
            air_path,
            "-o", lib_path,
        ])

        self.announce(f"Precompiled metallib: {lib_path}", level=2)


CPP_SOURCES = [
    "csrc/backend/ProcessGroupMCCL.cpp",
    "csrc/backend/WorkMCCL.cpp",
    "csrc/backend/Registration.cpp",
    "csrc/transport/TcpTransport.cpp",
    "csrc/transport/Connection.cpp",
    "csrc/runtime/ProgressEngine.cpp",
    "csrc/runtime/Rendezvous.cpp",
    "csrc/runtime/Watchdog.cpp",
    "csrc/runtime/HealthMonitor.cpp",
    "csrc/runtime/Metrics.cpp",
    "csrc/runtime/MemoryPool.cpp",
    "csrc/compression/Compression.cpp",
    "csrc/compression/FP16Compression.cpp",
    "csrc/compression/TopKCompression.cpp",
    "csrc/transport/rdma/RdmaTransport.cpp",
    "csrc/transport/rdma/IbvWrapper.cpp",
    "csrc/transport/rdma/RdmaConnection.cpp",
    "csrc/transport/rdma/SharedBuffer.cpp",
]

MM_SOURCES = [
    "csrc/metal/MPSInterop.mm",
    "csrc/metal/MetalKernels.mm",
    "csrc/metal/AccelerateOps.mm",
    "csrc/metal/EventSync.mm",
]

ext = Extension(
    name="mccl._C",
    sources=CPP_SOURCES + MM_SOURCES,
    include_dirs=["csrc"],
    language="c++",
)

setup(
    name="mccl",
    version="0.3.0",
    description="MPS-native ProcessGroup backend for PyTorch Distributed on Apple Silicon",
    packages=["mccl"],
    ext_modules=[ext],
    cmdclass={"build_ext": MCCLBuildExt},
    python_requires=">=3.11",
)
