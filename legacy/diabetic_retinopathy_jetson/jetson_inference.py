#!/usr/bin/env python3
"""
Jetson Nano Inference Script
Diabetic Retinopathy Classification using EfficientNet-B2
Supports both original .pth and optimized TorchScript models
"""

import argparse
import time
import os
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Try to import timm (required for loading original .pth model)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Configuration
IMG_SIZE = 260
NUM_CLASSES = 5
DEFAULT_MODEL = "./efficientnet_b2_aptos_best.pth"

DR_LABELS = {
    0: "No DR (Không bệnh)",
    1: "Mild DR (Nhẹ)",
    2: "Moderate DR (Trung bình)",
    3: "Severe DR (Nặng)",
    4: "Proliferative DR (Tăng sinh)"
}


def resize_with_padding(img, size=260):
    """Resize image with padding to maintain aspect ratio"""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    new_img = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


class DRInference:
    """Inference class for Diabetic Retinopathy classification"""
    
    def __init__(self, model_path, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Detect model type and load
        self.model = self._load_model(model_path)
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # Setup transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Warm up model
        self._warmup()
    
    def _load_model(self, model_path):
        """Load model - supports both .pth (state_dict) and .pt (TorchScript)"""
        print(f"Loading model from: {model_path}")
        
        ext = Path(model_path).suffix.lower()
        
        if ext == ".pt":
            # TorchScript model (optimized)
            print("  → Detected TorchScript model (.pt)")
            model = torch.jit.load(model_path, map_location=self.device)
            
        elif ext == ".pth":
            # Original PyTorch state_dict
            print("  → Detected PyTorch state_dict (.pth)")
            
            if not TIMM_AVAILABLE:
                raise ImportError(
                    "timm package required to load .pth model. "
                    "Install with: pip install timm"
                )
            
            # Create model architecture
            model = timm.create_model(
                "efficientnet_b2",
                pretrained=False,
                num_classes=NUM_CLASSES
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            
        else:
            raise ValueError(f"Unsupported model format: {ext}")
        
        return model
    
    def _warmup(self):
        """Warm up model for accurate timing"""
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("✓ Warmup complete")
    
    def predict(self, image_path, verbose=True):
        """
        Predict diabetic retinopathy level
        
        Returns:
            pred_class, probs, inference_time (ms)
        """
        # Load and preprocess
        img = Image.open(image_path).convert("RGB")
        img = resize_with_padding(img, IMG_SIZE)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference with timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if verbose:
            self._print_results(image_path, pred_class, probs, inference_time)
        
        return pred_class, probs, inference_time
    
    def _print_results(self, image_path, pred_class, probs, inference_time):
        """Print formatted results"""
        print("\n" + "="*70)
        print(f"Image: {Path(image_path).name}")
        print("="*70)
        print(f"Prediction: Class {pred_class} - {DR_LABELS[pred_class]}")
        print(f"Confidence: {probs[pred_class]*100:.2f}%")
        print(f"Inference Time: {inference_time:.2f} ms")
        print("\nProbability Distribution:")
        for i in range(NUM_CLASSES):
            bar = "█" * int(probs[i] * 40)
            print(f"  {i} {DR_LABELS[i]:30s}: {probs[i]*100:5.2f}% {bar}")
        print("="*70)
    
    def predict_batch(self, image_dir, extensions=('.png', '.jpg', '.jpeg')):
        """Predict on all images in a directory"""
        image_dir = Path(image_dir)
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"\nFound {len(image_files)} images")
        print("="*80)
        
        results = []
        total_time = 0
        
        for img_path in image_files:
            pred_class, probs, inf_time = self.predict(img_path, verbose=False)
            total_time += inf_time
            results.append({
                'filename': img_path.name,
                'prediction': pred_class,
                'label': DR_LABELS[pred_class],
                'confidence': probs[pred_class],
                'time_ms': inf_time
            })
            print(f"{img_path.name:40s} → Class {pred_class} ({DR_LABELS[pred_class]:25s}) - {probs[pred_class]*100:.1f}% [{inf_time:.1f}ms]")
        
        print("="*80)
        
        # Summary
        print(f"\nProcessed {len(results)} images in {total_time:.1f} ms")
        print(f"Average: {total_time/len(results):.2f} ms/image")
        print("\nClass Distribution:")
        for i in range(NUM_CLASSES):
            count = sum(1 for r in results if r['prediction'] == i)
            print(f"  {DR_LABELS[i]:30s}: {count:3d} images ({count/len(results)*100:.1f}%)")
        
        return results
    
    def benchmark(self, image_path, num_runs=100):
        """Benchmark inference performance"""
        print(f"\nBenchmarking with {num_runs} runs...")
        
        # Load image once
        img = Image.open(image_path).convert("RGB")
        img = resize_with_padding(img, IMG_SIZE)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = self.model(img_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
        
        # Statistics
        times = np.array(times)
        print("\nBenchmark Results:")
        print(f"  Mean:   {times.mean():.2f} ms")
        print(f"  Median: {np.median(times):.2f} ms")
        print(f"  Min:    {times.min():.2f} ms")
        print(f"  Max:    {times.max():.2f} ms")
        print(f"  Std:    {times.std():.2f} ms")
        print(f"  FPS:    {1000/times.mean():.2f}")
        
        # Memory usage
        if self.device.type == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / (1024**2)
            mem_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"\nGPU Memory:")
            print(f"  Allocated: {mem_allocated:.2f} MB")
            print(f"  Reserved:  {mem_reserved:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Diabetic Retinopathy Classification - Jetson Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 jetson_inference.py --image test.png
  python3 jetson_inference.py --image test.png --benchmark
  python3 jetson_inference.py --dir ./test_images
  python3 jetson_inference.py --model optimized.pt --image test.png
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to model file (.pth or .pt) [default: {DEFAULT_MODEL}]"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to single image"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Path to directory of images"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (default: use CUDA if available)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Please ensure the model file exists.")
        return
    
    # Initialize inference
    try:
        inferencer = DRInference(args.model, use_cuda=not args.cpu)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.image:
        # Single image inference
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        inferencer.predict(args.image)
        
        if args.benchmark:
            inferencer.benchmark(args.image)
            
    elif args.dir:
        # Batch inference
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
        
        inferencer.predict_batch(args.dir)
        
    else:
        # Interactive mode with menu
        interactive_mode(args.model, not args.cpu)


def interactive_mode(model_path, use_cuda):
    """Interactive menu for inference"""
    
    print("\n" + "="*60)
    print("  DIABETIC RETINOPATHY CLASSIFICATION - JETSON NANO")
    print("="*60)
    
    # Validate model
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        return
    
    # Load model once
    print(f"\nLoading model: {model_path}")
    try:
        inferencer = DRInference(model_path, use_cuda=use_cuda)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    while True:
        print("\n" + "-"*60)
        print("MENU:")
        print("-"*60)
        print("  1. Predict single image")
        print("  2. Predict folder of images")
        print("  3. Predict single image + Benchmark")
        print("  4. Predict folder of images + Benchmark")
        print("  5. Exit")
        print("-"*60)
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Single image
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                inferencer.predict(img_path)
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif choice == "2":
            # Folder of images
            dir_path = input("Enter folder path: ").strip()
            if os.path.isdir(dir_path):
                inferencer.predict_batch(dir_path)
            else:
                print(f"❌ Folder not found: {dir_path}")
        
        elif choice == "3":
            # Single image with benchmark
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                inferencer.predict(img_path)
                inferencer.benchmark(img_path)
            else:
                print(f"❌ Image not found: {img_path}")
        
        elif choice == "4":
            # Folder with benchmark
            dir_path = input("Enter folder path: ").strip()
            if os.path.isdir(dir_path):
                results = inferencer.predict_batch(dir_path)
                if results:
                    # Use first image for benchmark
                    first_img = os.path.join(dir_path, results[0]['filename'])
                    print("\n📊 Running benchmark on first image...")
                    inferencer.benchmark(first_img)
            else:
                print(f"❌ Folder not found: {dir_path}")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
