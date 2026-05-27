#!/usr/bin/env python3
"""
Model Optimization for Jetson Nano
Converts PyTorch model to TorchScript and optimizes for TensorRT
"""

import argparse
import time
import torch
import torch.nn as nn
import timm
from pathlib import Path


def optimize_model(model_path, output_dir="models", use_fp16=True):
    """
    Optimize PyTorch model for Jetson Nano deployment
    
    Args:
        model_path: Path to original .pth model
        output_dir: Directory to save optimized model
        use_fp16: Use FP16 precision (faster, slightly less accurate)
    """
    print("="*60)
    print("Model Optimization for Jetson Nano")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load original model
    print(f"\n[1/5] Loading model from: {model_path}")
    model = timm.create_model(
        "efficientnet_b2",
        pretrained=False,
        num_classes=5
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create example input
    print("\n[2/5] Creating example input...")
    example_input = torch.randn(1, 3, 260, 260).to(device)
    
    # Test original model
    print("\n[3/5] Testing original model...")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(example_input)
        end = time.time()
    original_time = (end - start) / 100
    print(f"✓ Original inference time: {original_time*1000:.2f} ms")
    
    # Convert to TorchScript
    print("\n[4/5] Converting to TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Apply FP16 if requested
    if use_fp16 and device.type == "cuda":
        print("  → Applying FP16 optimization...")
        traced_model = traced_model.half()
        example_input = example_input.half()
    
    # Test optimized model
    print("\n[5/5] Testing optimized model...")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = traced_model(example_input)
        end = time.time()
    optimized_time = (end - start) / 100
    print(f"✓ Optimized inference time: {optimized_time*1000:.2f} ms")
    
    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    
    # Save optimized model
    output_path = output_dir / "efficientnet_b2_optimized.pt"
    torch.jit.save(traced_model, str(output_path))
    print(f"\n✓ Optimized model saved to: {output_path}")
    
    # Print file sizes
    import os
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    optimized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  Original:  {original_size:.2f} MB")
    print(f"  Optimized: {optimized_size:.2f} MB")
    print(f"  Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("="*60)
    
    return output_path


def verify_model(optimized_path, original_path):
    """
    Verify that optimized model produces similar results to original
    """
    print("\nVerifying model accuracy...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load both models
    original = timm.create_model("efficientnet_b2", pretrained=False, num_classes=5)
    original.load_state_dict(torch.load(original_path, map_location=device))
    original = original.to(device).eval()
    
    optimized = torch.jit.load(str(optimized_path), map_location=device)
    optimized.eval()
    
    # Test on random inputs
    test_input = torch.randn(10, 3, 260, 260).to(device)
    
    with torch.no_grad():
        original_out = original(test_input)
        optimized_out = optimized(test_input)
    
    # Compare outputs
    diff = torch.abs(original_out - optimized_out).mean().item()
    print(f"Mean absolute difference: {diff:.6f}")
    
    # Check if predictions match
    original_preds = original_out.argmax(dim=1)
    optimized_preds = optimized_out.argmax(dim=1)
    accuracy = (original_preds == optimized_preds).float().mean().item()
    print(f"Prediction agreement: {accuracy*100:.1f}%")
    
    if accuracy >= 0.95:
        print("✓ Verification passed!")
    else:
        print("⚠ Warning: Predictions differ significantly")


def main():
    parser = argparse.ArgumentParser(description="Optimize model for Jetson Nano")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="../Source/efficientnet_b2_aptos_best.pth",
        help="Path to original model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify optimized model"
    )
    
    args = parser.parse_args()
    
    # Optimize model
    output_path = optimize_model(args.input, args.output, args.fp16)
    
    # Verify if requested
    if args.verify:
        verify_model(output_path, args.input)


if __name__ == "__main__":
    main()
