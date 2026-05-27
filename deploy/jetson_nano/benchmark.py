#!/usr/bin/env python3
"""Latency benchmark utility for TorchScript inference."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from infer import build_transform, load_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TorchScript inference")
    parser.add_argument("--model", required=True, help="Path to TorchScript model")
    parser.add_argument("--image", required=True, help="Path to sample image")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs before timing")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    image = load_image(args.image, build_transform()).to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)

    print(f"device={device}")
    print(f"runs={args.runs}")
    print(f"mean_ms={statistics.mean(timings):.2f}")
    print(f"median_ms={statistics.median(timings):.2f}")
    print(f"min_ms={min(timings):.2f}")
    print(f"max_ms={max(timings):.2f}")
    print(f"fps={1000.0 / statistics.mean(timings):.2f}")
    if device.type == "cuda":
        print(f"gpu_mem_allocated_mb={torch.cuda.memory_allocated() / (1024 ** 2):.2f}")
        print(f"gpu_mem_reserved_mb={torch.cuda.memory_reserved() / (1024 ** 2):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
