"""
MCCL build script.

Builds the C++/Obj-C++ extension that provides the ProcessGroupMCCL backend on
macOS Apple Silicon.

If ``xcrun metal`` is available, ``mccl_shaders.metallib`` is built next to ``_C``.
If not (CLT-only machine), the build **warns and skips** metallib and still copies
``shaders.metal`` beside the extension for **runtime JIT**.

For **PyPI / release wheels**, set ``MCCL_REQUIRE_METALLIB=1`` so the build **fails**
when the Metal CLI is missing (ensures every wheel ships a ``.metallib``).
"""
import os
import platform
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _torch_include_dirs():
    import sysconfig
    import torch
    from torch.utils.cpp_extension import include_paths
    torch_root = os.path.join(os.path.dirname(torch.__file__), "include")
    distributed_inc = os.path.join(torch_root, "torch", "csrc", "distributed")
    python_inc = sysconfig.get_path("include")
    return include_paths() + [distributed_inc, python_inc]


def _torch_library_dirs():
    from torch.utils.cpp_extension import library_paths
    return library_paths()


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
            "-ltorch_python",
            "-lc10",
            f"-Wl,-rpath,{torch_lib}",
            "-undefined", "dynamic_lookup",
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

        self._fixup_rpath(ext_path)
        out_dir = os.path.dirname(ext_path)
        self._compile_metallib(out_dir)
        self._install_shaders_metal_next_to_extension(out_dir)

    @staticmethod
    def _fixup_rpath(ext_path):
        """Replace the build-time torch rpath with a relative one that works at runtime."""
        import torch
        runtime_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        result = subprocess.run(
            ["otool", "-l", ext_path], capture_output=True, text=True
        )
        for i, line in enumerate(result.stdout.splitlines()):
            if "path" in line and "torch" in line and "pip-build-env" in line:
                stale = line.strip().split()[1]
                subprocess.check_call([
                    "install_name_tool", "-delete_rpath", stale, ext_path
                ])
        subprocess.run([
            "install_name_tool", "-add_rpath", runtime_torch_lib, ext_path
        ], capture_output=True)

    @staticmethod
    def _detect_metal_std():
        """Pick the highest Metal language standard the host toolchain supports."""
        mac_ver = platform.mac_ver()[0]  # e.g. "15.1.1" or "14.7.2"
        if mac_ver:
            major = int(mac_ver.split(".")[0])
            if major >= 15:
                return "metal3.1"
        return "metal3.0"

    def _install_shaders_metal_next_to_extension(self, output_dir: str) -> None:
        """Copy ``shaders.metal`` beside ``_C`` so wheels/runtime can JIT-compile if needed."""
        shader_src = os.path.join("csrc", "metal", "shaders.metal")
        if not os.path.isfile(shader_src):
            raise RuntimeError(
                f"MCCL build requires {shader_src} in the source tree."
            )
        dst = os.path.join(output_dir, "shaders.metal")
        shutil.copy2(shader_src, dst)
        self.announce(f"Installed shaders.metal next to extension: {dst}", level=2)

    def _compile_metallib(self, output_dir):
        shader_src = os.path.join("csrc", "metal", "shaders.metal")
        if not os.path.isfile(shader_src):
            raise RuntimeError(
                f"MCCL build requires {shader_src} in the source tree."
            )

        require_metallib = os.environ.get("MCCL_REQUIRE_METALLIB", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        try:
            subprocess.check_output(
                ["xcrun", "--find", "metal"], text=True, stderr=subprocess.STDOUT
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = (
                "Metal shader compiler not found (`xcrun metal`). Skipping precompiled "
                "mccl_shaders.metallib; shaders.metal is still installed next to _C for runtime JIT. "
                "For PyPI wheels, build on a Mac with full Xcode and set MCCL_REQUIRE_METALLIB=1 "
                "so this step cannot be skipped silently."
            )
            if require_metallib:
                raise RuntimeError(
                    "MCCL_REQUIRE_METALLIB=1 but `xcrun metal` is not available — "
                    "install full Xcode (not Command Line Tools only)."
                ) from e
            self.warn(msg)
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

        if not os.path.isfile(lib_path):
            raise RuntimeError(f"metallib build did not produce {lib_path}")

        self.announce(f"Precompiled metallib: {lib_path}", level=2)


CPP_SOURCES = [
    "csrc/backend/ProcessGroupMCCL.cpp",
    "csrc/backend/WorkMCCL.cpp",
    "csrc/backend/Registration.cpp",
    "csrc/backend/MPSDispatch.cpp",
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
    version="0.3.1",
    description="MPS-native ProcessGroup backend for PyTorch Distributed on Apple Silicon",
    packages=["mccl"],
    ext_modules=[ext],
    cmdclass={"build_ext": MCCLBuildExt},
    python_requires=">=3.11",
)
