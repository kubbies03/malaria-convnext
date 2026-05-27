#!/bin/bash
# ============================================================================
# Jetson Nano AI Environment Setup Script
# Target: Jetson Nano - Ubuntu 18.04 - JetPack 4.x
# Purpose: Diabetic Retinopathy Classification Model Deployment
# ============================================================================
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# This script will:
#   1. Configure swap space (required for PyTorch)
#   2. Install system dependencies
#   3. Install PyTorch for Jetson (with multiple fallback sources)
#   4. Install TorchVision (pre-built wheel or source)
#   5. Install Python packages (timm, pillow, numpy, etc.)
#   6. Verify installation
#
# ============================================================================

# === Configuration ===
VENV_DIR="$HOME/dr_env"
WORK_DIR="$HOME/dr_setup_temp"
SWAP_SIZE="4G"

# PyTorch wheel URLs (multiple sources for reliability)
# JetPack 4.6 - Python 3.6 - PyTorch 1.10.0
PYTORCH_WHEEL_URLS=(
    "https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
    "https://developer.download.nvidia.cn/compute/redist/jp/v46/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
)

# TorchVision wheel (pre-built for JetPack 4.6)
TORCHVISION_WHEEL_URL="https://nvidia.box.com/shared/static/1tye0vl06c5g6vbplj5rnm8pqz6eo2e1.whl"

# === Color Codes ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === Helper Functions ===
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        JETSON_INFO=$(cat /etc/nv_tegra_release | head -1)
        log_success "Jetson detected: $JETSON_INFO"
        return 0
    else
        log_warning "Not running on Jetson Nano"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

check_python_version() {
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    log_info "Python version: $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 6 ]; then
        log_success "Python 3.6 detected - compatible with JetPack 4.x wheels"
        PYTHON_TAG="cp36"
    elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 8 ]; then
        log_warning "Python 3.8 detected - may need different wheels"
        PYTHON_TAG="cp38"
    else
        log_warning "Python $PYTHON_VERSION - compatibility unknown"
        PYTHON_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}"
    fi
}

# === Main Installation Steps ===

step_disable_nvidia_repos() {
    log_step "Step 0/7: Handling NVIDIA Repositories"
    
    # Disable NVIDIA repos to avoid update errors
    if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
        log_info "Disabling NVIDIA repos to prevent update errors..."
        sudo mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
               /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak 2>/dev/null || true
        log_success "NVIDIA repos disabled (can be restored later)"
    else
        log_info "NVIDIA repos not found or already disabled"
    fi
}

step_configure_swap() {
    log_step "Step 1/7: Configuring Swap Space"
    
    CURRENT_SWAP=$(free -g | grep Swap | awk '{print $2}')
    log_info "Current swap: ${CURRENT_SWAP}GB"
    
    if [ "$CURRENT_SWAP" -lt 4 ]; then
        log_info "Creating $SWAP_SIZE swap file..."
        
        if [ -f /swapfile ]; then
            log_info "Removing existing swapfile..."
            sudo swapoff /swapfile 2>/dev/null || true
            sudo rm -f /swapfile
        fi
        
        sudo fallocate -l $SWAP_SIZE /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Add to fstab if not already present
        if ! grep -q "swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        
        log_success "Swap configured: $SWAP_SIZE"
    else
        log_success "Swap already sufficient: ${CURRENT_SWAP}GB"
    fi
}

step_install_system_deps() {
    log_step "Step 2/7: Installing System Dependencies"
    
    log_info "Updating package lists (ignoring errors)..."
    sudo apt-get update 2>/dev/null || log_warning "Some repos failed to update (this is OK)"
    
    log_info "Installing required packages..."
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-setuptools \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        cmake \
        git \
        wget \
        curl \
        2>/dev/null || log_warning "Some packages may have failed"
    
    log_success "System dependencies installed"
}

step_create_venv() {
    log_step "Step 3/7: Creating Python Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created at $VENV_DIR"
    fi
    
    # Activate
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    log_success "pip upgraded"
}

step_install_pytorch() {
    log_step "Step 4/7: Installing PyTorch for Jetson"
    
    source "$VENV_DIR/bin/activate"
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # Check if PyTorch already installed
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
        EXISTING_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_warning "PyTorch $EXISTING_VERSION already installed"
        read -p "Reinstall? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    log_info "Downloading PyTorch wheel for Jetson..."
    
    PYTORCH_INSTALLED=false
    
    for url in "${PYTORCH_WHEEL_URLS[@]}"; do
        log_info "Trying: $url"
        WHEEL_FILE="torch_jetson.whl"
        
        if wget -q --show-progress -O "$WHEEL_FILE" "$url" 2>/dev/null; then
            log_info "Installing PyTorch..."
            if pip install "$WHEEL_FILE"; then
                log_success "PyTorch installed successfully"
                rm -f "$WHEEL_FILE"
                PYTORCH_INSTALLED=true
                break
            else
                log_warning "Installation failed, trying next source..."
                rm -f "$WHEEL_FILE"
            fi
        else
            log_warning "Download failed, trying next source..."
        fi
    done
    
    if [ "$PYTORCH_INSTALLED" = false ]; then
        log_error "Failed to install PyTorch from all sources"
        log_info "Trying pip install as fallback..."
        pip install torch || {
            log_error "PyTorch installation failed completely"
            log_info "Please manually install PyTorch for Jetson from:"
            log_info "https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
            exit 1
        }
    fi
    
    # Verify CUDA
    log_info "Verifying PyTorch CUDA support..."
    python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available - inference will be slow")
EOF
}

step_install_torchvision() {
    log_step "Step 5/7: Installing TorchVision"
    
    source "$VENV_DIR/bin/activate"
    cd "$WORK_DIR"
    
    # Check if already installed
    if python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null; then
        EXISTING_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)")
        log_warning "TorchVision $EXISTING_VERSION already installed"
        read -p "Reinstall? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    # Method 1: Try pre-built wheel
    log_info "Trying pre-built TorchVision wheel..."
    WHEEL_FILE="torchvision_jetson.whl"
    
    if wget -q --show-progress -O "$WHEEL_FILE" "$TORCHVISION_WHEEL_URL" 2>/dev/null; then
        if pip install "$WHEEL_FILE" 2>/dev/null; then
            log_success "TorchVision installed from wheel"
            rm -f "$WHEEL_FILE"
            return 0
        fi
        rm -f "$WHEEL_FILE"
    fi
    
    # Method 2: Build from source
    log_warning "Pre-built wheel failed. Building TorchVision from source..."
    log_info "This may take 20-30 minutes..."
    
    # Install build dependencies
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev \
        libavcodec-dev libavformat-dev libswscale-dev 2>/dev/null || true
    
    # Clone and build
    if [ -d "torchvision" ]; then
        rm -rf torchvision
    fi
    
    git clone --branch v0.11.1 --depth 1 https://github.com/pytorch/vision torchvision
    cd torchvision
    
    export BUILD_VERSION=0.11.1
    export TORCH_CUDA_ARCH_LIST="5.3"  # Jetson Nano GPU arch
    
    python setup.py install
    
    cd ..
    rm -rf torchvision
    
    log_success "TorchVision installed from source"
}

step_install_python_packages() {
    log_step "Step 6/7: Installing Python Packages"
    
    source "$VENV_DIR/bin/activate"
    
    log_info "Installing core packages..."
    
    # Install packages one by one for better error handling
    PACKAGES=(
        "numpy"
        "pillow"
        "pandas"
        "scikit-learn"
        "tqdm"
        "timm"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        log_info "Installing $pkg..."
        pip install "$pkg" --no-cache-dir || log_warning "Failed to install $pkg"
    done
    
    # OpenCV - may need special handling
    log_info "Installing OpenCV..."
    pip install opencv-python-headless --no-cache-dir 2>/dev/null || \
    pip install opencv-python --no-cache-dir 2>/dev/null || \
    sudo apt-get install -y python3-opencv 2>/dev/null || \
    log_warning "OpenCV installation may have issues"
    
    log_success "Python packages installed"
}

step_verify_installation() {
    log_step "Step 7/7: Verifying Installation"
    
    source "$VENV_DIR/bin/activate"
    
    echo ""
    log_info "Testing imports..."
    
    python3 << 'EOF'
import sys

def test_import(name, pkg_name=None):
    pkg_name = pkg_name or name
    try:
        module = __import__(pkg_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: FAILED ({e})")
        return False

print("\n=== Package Verification ===")
print(f"Python: {sys.version.split()[0]}")
print("")

results = []
results.append(test_import("PyTorch", "torch"))
results.append(test_import("TorchVision", "torchvision"))
results.append(test_import("TIMM", "timm"))
results.append(test_import("NumPy", "numpy"))
results.append(test_import("Pillow", "PIL"))
results.append(test_import("Pandas", "pandas"))
results.append(test_import("OpenCV", "cv2"))
results.append(test_import("scikit-learn", "sklearn"))
results.append(test_import("tqdm", "tqdm"))

print("")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.version.cuda}")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Quick GPU test
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("  ✓ GPU compute: OK")
    except Exception as e:
        print(f"  ✗ GPU compute: FAILED ({e})")
else:
    print("  ⚠ CUDA: Not available (CPU only)")

print("")
if all(results):
    print("=== All packages installed successfully! ===")
else:
    print("=== Some packages failed to install ===")
    sys.exit(1)
EOF
    
    VERIFY_STATUS=$?
    
    if [ $VERIFY_STATUS -eq 0 ]; then
        log_success "All verifications passed!"
    else
        log_error "Some verifications failed"
    fi
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "$WORK_DIR"
    
    # Restore NVIDIA repos
    if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak ]; then
        sudo mv /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.bak \
               /etc/apt/sources.list.d/nvidia-l4t-apt-source.list 2>/dev/null || true
        log_info "NVIDIA repos restored"
    fi
}

print_final_instructions() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}   Installation Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "To use the environment:"
    echo -e "  ${CYAN}source $VENV_DIR/bin/activate${NC}"
    echo ""
    echo -e "Quick test:"
    echo -e "  ${CYAN}python3 -c \"import torch; print(torch.cuda.is_available())\"${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Copy model file: ${YELLOW}efficientnet_b2_aptos_best.pth${NC}"
    echo -e "  2. Run optimization (optional): ${YELLOW}python3 optimize_for_jetson.py${NC}"
    echo -e "  3. Test inference: ${YELLOW}python3 jetson_inference.py --image test.png${NC}"
    echo ""
    echo -e "For better performance:"
    echo -e "  ${CYAN}sudo nvpmodel -m 0${NC}     # Max power mode"
    echo -e "  ${CYAN}sudo jetson_clocks${NC}    # Lock clocks to maximum"
    echo ""
}

# === Main Execution ===
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   Jetson Nano AI Environment Setup                         ║${NC}"
    echo -e "${CYAN}║   Diabetic Retinopathy Classification                      ║${NC}"
    echo -e "${CYAN}║   Ubuntu 18.04 / JetPack 4.x / Python 3.6                  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Pre-flight checks
    check_jetson
    check_python_version
    
    # Installation steps
    step_disable_nvidia_repos
    step_configure_swap
    step_install_system_deps
    step_create_venv
    step_install_pytorch
    step_install_torchvision
    step_install_python_packages
    step_verify_installation
    
    # Cleanup
    cleanup
    
    # Done
    print_final_instructions
}

# Run
main "$@"
