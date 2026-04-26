from __future__ import annotations

import os
import pathlib
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


AUTO_MODEL_PROFILE = "auto"
BASELINE_MODEL_PROFILE = "baseline"
GPU_MODEL_PROFILE = "gpu"
SUPPORTED_MODEL_PROFILES = {
    AUTO_MODEL_PROFILE,
    BASELINE_MODEL_PROFILE,
    GPU_MODEL_PROFILE,
}

_FORCE_DEVICE_ENV = "TOKENSMITH_FORCE_DEVICE"


@dataclass(frozen=True)
class HardwareProfile:
    gpu_available: bool
    backend: str
    reason: str


def _run_command(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _linux_gpu_profile() -> HardwareProfile:
    if shutil.which("nvidia-smi"):
        output = _run_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        )
        if output:
            first_line = output.splitlines()[0]
            return HardwareProfile(True, "cuda", f"Detected NVIDIA GPU via nvidia-smi: {first_line}")

    if shutil.which("rocm-smi"):
        output = _run_command(["rocm-smi", "--showproductname"])
        if output:
            first_line = output.splitlines()[0]
            return HardwareProfile(True, "hipblas", f"Detected AMD GPU via rocm-smi: {first_line}")

    if shutil.which("vulkaninfo"):
        output = _run_command(["vulkaninfo", "--summary"])
        if output:
            return HardwareProfile(True, "vulkan", "Detected Vulkan-capable GPU via vulkaninfo")

    if shutil.which("lspci"):
        output = _run_command(["lspci"])
        lowered = output.lower()
        if any(marker in lowered for marker in ("nvidia", "geforce", "quadro")):
            return HardwareProfile(True, "cuda", "Detected NVIDIA GPU via lspci")
        if any(marker in lowered for marker in ("amd", "radeon", "advanced micro devices", "ati")):
            return HardwareProfile(True, "vulkan", "Detected AMD GPU via lspci")

    if pathlib.Path("/dev/dri/renderD128").exists():
        return HardwareProfile(True, "vulkan", "Detected render node at /dev/dri/renderD128")

    return HardwareProfile(False, "cpu", "No supported GPU backend detected on Linux")


@lru_cache(maxsize=1)
def detect_hardware_profile() -> HardwareProfile:
    forced = os.getenv(_FORCE_DEVICE_ENV, "").strip().lower()
    if forced == "gpu":
        return HardwareProfile(True, "forced", f"{_FORCE_DEVICE_ENV}=gpu")
    if forced == "cpu":
        return HardwareProfile(False, "cpu", f"{_FORCE_DEVICE_ENV}=cpu")

    system = platform.system()
    if system == "Linux":
        return _linux_gpu_profile()
    if system == "Darwin":
        if platform.machine().lower() == "arm64":
            return HardwareProfile(True, "metal", "Detected Apple Silicon GPU backend")
        return HardwareProfile(False, "cpu", "No accelerated GPU backend configured for this macOS host")
    if system == "Windows":
        if shutil.which("nvidia-smi"):
            output = _run_command(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
            )
            if output:
                first_line = output.splitlines()[0]
                return HardwareProfile(True, "cuda", f"Detected NVIDIA GPU via nvidia-smi: {first_line}")
        return HardwareProfile(False, "cpu", "No supported GPU backend detected on Windows")
    return HardwareProfile(False, "cpu", f"Unsupported platform for GPU autodetect: {system}")


def gguf_path_available(model_name: str) -> bool:
    candidate = pathlib.Path(model_name)
    return candidate.suffix.lower() != ".gguf" or candidate.exists()
