#!/usr/bin/env python3
"""Minimal TorchScript inference CLI for Jetson Nano demos."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


LABELS = ["Parasitized", "Uninfected"]
IMAGE_SIZE = 224


def build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: Path, transform: T.Compose) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model: torch.jit.ScriptModule, tensor: torch.Tensor, device: torch.device) -> tuple[int, np.ndarray, float]:
    tensor = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    return int(np.argmax(probs)), probs, latency_ms


def run_single(model: torch.jit.ScriptModule, image_path: Path, transform: T.Compose, device: torch.device) -> None:
    pred, probs, latency_ms = predict(model, load_image(image_path, transform), device)
    print(f"image={image_path.name}")
    print(f"prediction={LABELS[pred]}")
    print(f"confidence={probs[pred]:.4f}")
    print(f"latency_ms={latency_ms:.2f}")


def run_folder(model: torch.jit.ScriptModule, directory: Path, transform: T.Compose, device: torch.device) -> None:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        run_single(model, path, transform, device)


def main() -> int:
    parser = argparse.ArgumentParser(description="TorchScript inference for Jetson Nano")
    parser.add_argument("--model", required=True, help="Path to TorchScript model")
    parser.add_argument("--image", help="Path to one image")
    parser.add_argument("--dir", dest="image_dir", help="Path to image directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --dir")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    transform = build_transform()

    if args.image:
        run_single(model, Path(args.image), transform, device)
    if args.image_dir:
        run_folder(model, Path(args.image_dir), transform, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
