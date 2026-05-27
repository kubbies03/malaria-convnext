#!/usr/bin/env python3
"""Runtime readiness check for Jetson Nano deployment."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def to_gb(value: int) -> float:
    return value / (1024 ** 3)


def read_meminfo() -> dict[str, int]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return {}
    data: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        data[key] = int(raw.strip().split()[0]) * 1024
    return data


def main() -> int:
    print("Jetson Nano System Check")
    print("========================")

    print("\n[Platform]")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    print("\n[Storage]")
    total, used, free = shutil.disk_usage("/")
    print(f"Root total: {to_gb(total):.2f} GB")
    print(f"Root used:  {to_gb(used):.2f} GB")
    print(f"Root free:  {to_gb(free):.2f} GB")

    print("\n[Memory]")
    meminfo = read_meminfo()
    if meminfo:
        print(f"RAM total:     {to_gb(meminfo.get('MemTotal', 0)):.2f} GB")
        print(f"RAM available: {to_gb(meminfo.get('MemAvailable', 0)):.2f} GB")
    else:
        print("Could not read /proc/meminfo")

    print("\n[PyTorch]")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU memory: {to_gb(props.total_memory):.2f} GB")
    except Exception as exc:
        print(f"PyTorch import failed: {exc}")

    print("\n[Environment]")
    print(f"VIRTUAL_ENV: {os.getenv('VIRTUAL_ENV', 'not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
