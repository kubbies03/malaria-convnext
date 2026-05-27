# Jetson Nano Deployment Guide

Complete guide for deploying the Diabetic Retinopathy classification model on NVIDIA Jetson Nano.

## 📋 Prerequisites

### Hardware
- **NVIDIA Jetson Nano Developer Kit** (4GB recommended)
- **MicroSD Card**: 32GB+ (64GB recommended)
- **Power Supply**: 5V 4A barrel jack adapter
- **Camera** (optional): USB webcam or CSI camera for real-time demo
- **Monitor, keyboard, mouse** for initial setup

### Software
- **JetPack 4.6+** (includes CUDA, cuDNN, TensorRT)
- **Python 3.6+**
- **Ubuntu 18.04** (pre-installed with JetPack)

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone or copy project to Jetson Nano
cd ~
mkdir diabetic-retinopathy
cd diabetic-retinopathy

# Copy files from your PC:
# - jetson_setup.sh
# - optimize_for_jetson.py
# - jetson_inference.py
# - realtime_camera.py
# - efficientnet_b2_aptos_best.pth (from Source/)
```

### 2. Run Setup Script

```bash
# Make script executable
chmod +x jetson_setup.sh

# Run setup (takes 30-60 minutes)
./jetson_setup.sh

# Activate virtual environment
source ~/dr_env/bin/activate
```

### 3. Optimize Model

```bash
# Convert model for Jetson Nano
python3 optimize_for_jetson.py \
    --input efficientnet_b2_aptos_best.pth \
    --output models \
    --fp16 \
    --verify

# Expected output:
# ✓ Original inference time: ~150 ms
# ✓ Optimized inference time: ~50 ms
# 🚀 Speedup: 3.0x
```

### 4. Test Inference

```bash
# Single image prediction
python3 jetson_inference.py \
    --model models/efficientnet_b2_optimized.pt \
    --image test_image.png

# With benchmark
python3 jetson_inference.py \
    --model models/efficientnet_b2_optimized.pt \
    --image test_image.png \
    --benchmark
```

### 5. Real-time Camera Demo (Optional)

```bash
# Run real-time inference
python3 realtime_camera.py \
    --model models/efficientnet_b2_optimized.pt \
    --camera 0

# Press 'q' to quit
# Press 's' to save screenshot
```

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **Inference Time** | 40-60 ms |
| **FPS** | 15-25 |
| **GPU Memory** | ~1.5 GB |
| **Accuracy** | ~77% (same as original) |
| **Model Size** | ~15 MB (optimized) |

## 📁 File Structure

```
diabetic-retinopathy/
├── jetson_setup.sh                    # Setup script
├── optimize_for_jetson.py             # Model optimization
├── jetson_inference.py                # Inference script
├── realtime_camera.py                 # Real-time camera demo
├── efficientnet_b2_aptos_best.pth    # Original model
└── models/
    └── efficientnet_b2_optimized.pt  # Optimized model
```

## 🔧 Detailed Setup Steps

### Step 1: Flash JetPack to SD Card

1. Download **NVIDIA SDK Manager** on your PC
2. Flash **JetPack 4.6** to microSD card
3. Boot Jetson Nano and complete initial setup

### Step 2: Increase Swap Space

```bash
# Check current swap
free -h

# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Step 3: Install Dependencies

The `jetson_setup.sh` script handles this automatically, but manual steps:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system packages
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libjpeg-dev \
    cmake

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
    -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install TorchVision
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user

# Install other packages
pip3 install timm pillow pandas scikit-learn opencv-python
```

### Step 4: Verify Installation

```bash
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

Expected output:
```
PyTorch: 1.10.0
CUDA Available: True
GPU: NVIDIA Tegra X1
```

## 🎯 Usage Examples

### Example 1: Batch Processing

```python
from jetson_inference import JetsonInference
from pathlib import Path

# Initialize
model = JetsonInference("models/efficientnet_b2_optimized.pt")

# Process all images in folder
image_dir = Path("test_images")
for img_path in image_dir.glob("*.png"):
    pred, probs, time = model.predict(str(img_path), verbose=False)
    print(f"{img_path.name}: Class {pred} ({time:.2f}ms)")
```

### Example 2: Save Results to CSV

```python
import pandas as pd
from jetson_inference import JetsonInference

model = JetsonInference("models/efficientnet_b2_optimized.pt")

results = []
for img in image_list:
    pred, probs, time = model.predict(img, verbose=False)
    results.append({
        'image': img,
        'prediction': pred,
        'confidence': probs[pred],
        'inference_time_ms': time
    })

df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
```

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution:**
```bash
# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Use CPU inference
python3 jetson_inference.py --image test.png --cpu
```

### Issue: Slow inference (>200ms)

**Possible causes:**
- Model not optimized → Run `optimize_for_jetson.py`
- CPU mode → Check CUDA availability
- Thermal throttling → Check temperature

```bash
# Check GPU temperature
tegrastats

# Enable max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall in virtual environment
source ~/dr_env/bin/activate
pip3 install --upgrade torch torchvision timm
```

### Issue: Camera not detected

**Solution:**
```bash
# List available cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different camera ID
python3 realtime_camera.py --camera 1
```

## ⚡ Performance Optimization Tips

### 1. Enable Max Performance Mode

```bash
# Set to maximum power mode (10W)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify
sudo nvpmodel -q
```

### 2. Optimize OpenCV

```bash
# Build OpenCV with CUDA support (optional, advanced)
# This can improve preprocessing speed
```

### 3. Batch Processing

Process multiple images in batches for better GPU utilization:

```python
# Process in batches of 4
batch_size = 4
# ... batch processing code
```

## 📈 Benchmarking

### Run Full Benchmark

```bash
python3 jetson_inference.py \
    --image test.png \
    --benchmark

# Output:
# Mean:   45.23 ms
# Median: 44.50 ms
# FPS:    22.11
# GPU Memory: 1.45 GB
```

### Compare with PC

| Device | Inference Time | FPS | GPU Memory |
|--------|---------------|-----|------------|
| **PC (RTX 3060)** | 15 ms | 66 | 2.5 GB |
| **Jetson Nano** | 50 ms | 20 | 1.5 GB |

## 🎓 Additional Resources

- [NVIDIA Jetson Nano Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## 📝 Notes

- **Power Supply**: Use 5V 4A barrel jack for stable performance
- **Cooling**: Consider adding a fan for sustained workloads
- **Storage**: Use high-quality SD card (UHS-1 or better)
- **Network**: Ethernet recommended for file transfers

## ✅ Checklist

- [ ] JetPack 4.6+ installed
- [ ] Swap space configured (4GB)
- [ ] Dependencies installed (`jetson_setup.sh`)
- [ ] Model optimized (`optimize_for_jetson.py`)
- [ ] Inference tested (`jetson_inference.py`)
- [ ] Performance verified (< 100ms per image)
- [ ] Real-time demo working (optional)

## 🎉 Success Criteria

Your deployment is successful if:
- ✅ Model loads without errors
- ✅ Inference time < 100ms per image
- ✅ Predictions match PC results (±2% accuracy)
- ✅ GPU memory usage < 2GB
- ✅ Real-time camera achieves >10 FPS

---

**Congratulations!** You've successfully deployed the diabetic retinopathy classification model on Jetson Nano! 🚀
