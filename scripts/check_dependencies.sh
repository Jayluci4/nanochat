#!/bin/bash
#
# check_dependencies.sh - Verify all dependencies are installed
# Run this FIRST when you SSH into the VM
#
# Usage: bash scripts/check_dependencies.sh
#

set -e
echo "========================================================================"
echo "Dependency Check for Nanochat D32 Remediation"
echo "========================================================================"
echo ""

EXIT_CODE=0

# Function to check command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✅ $1 is installed"
        if [ ! -z "$2" ]; then
            echo "   Version: $($1 $2 2>&1 | head -1)"
        fi
        return 0
    else
        echo "❌ $1 is NOT installed"
        EXIT_CODE=1
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python3 -c "import $1" 2>/dev/null; then
        VERSION=$(python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo "✅ Python package '$1' is installed (version: $VERSION)"
        return 0
    else
        echo "❌ Python package '$1' is NOT installed"
        EXIT_CODE=1
        return 1
    fi
}

echo "1. Checking System Commands"
echo "----------------------------"
check_command "python3" "--version"
check_command "git" "--version"
check_command "uv" "--version"
check_command "rustc" "--version"
check_command "cargo" "--version"
echo ""

echo "2. Checking NVIDIA/CUDA"
echo "-----------------------"
check_command "nvidia-smi" "--version"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   GPU Count: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -eq 8 ]; then
        echo "✅ Found 8 GPUs (as expected for L4 setup)"
    else
        echo "⚠️  Expected 8 GPUs but found $GPU_COUNT"
    fi
    
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
echo ""

echo "3. Checking Python Environment"
echo "-------------------------------"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Virtual environment exists at .venv"
else
    echo "❌ Virtual environment NOT found at .venv"
    echo "   Run: uv venv"
    EXIT_CODE=1
fi

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VERSION" == "3.10" ]] || [[ "$PYTHON_VERSION" == "3.11" ]]; then
        echo "✅ Python version is acceptable: $PYTHON_VERSION"
    else
        echo "⚠️  Python version is $PYTHON_VERSION (expected 3.10 or 3.11)"
    fi
    
    # Check critical packages
    check_python_package "torch"
    check_python_package "numpy"
    
    # Check torch CUDA
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        echo "✅ PyTorch can access CUDA (version: $CUDA_VERSION)"
    else
        echo "❌ PyTorch CANNOT access CUDA"
        EXIT_CODE=1
    fi
    
    # Check GPU count in PyTorch
    GPU_COUNT_TORCH=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "   PyTorch sees $GPU_COUNT_TORCH GPUs"
    
fi
echo ""

echo "4. Checking Nanochat Imports"
echo "-----------------------------"
if [ -d ".venv" ]; then
    source .venv/bin/activate
    
    check_python_package "nanochat"
    
    # Try importing key modules
    if python3 -c "from nanochat.gpt import GPT" 2>/dev/null; then
        echo "✅ Can import nanochat.gpt.GPT"
    else
        echo "❌ Cannot import nanochat.gpt.GPT"
        EXIT_CODE=1
    fi
    
    if python3 -c "from nanochat.tokenizer import get_tokenizer" 2>/dev/null; then
        echo "✅ Can import nanochat.tokenizer"
    else
        echo "❌ Cannot import nanochat.tokenizer"
        EXIT_CODE=1
    fi
fi
echo ""

echo "5. Checking AlgebraTutor Task"
echo "------------------------------"
if [ -f "tasks/algebra_tutor.py" ]; then
    echo "✅ tasks/algebra_tutor.py exists"
    
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        
        if python3 -c "from tasks.algebra_tutor import AlgebraTutor" 2>/dev/null; then
            echo "✅ Can import AlgebraTutor"
        else
            echo "❌ Cannot import AlgebraTutor (check for syntax errors)"
            EXIT_CODE=1
        fi
    fi
else
    echo "❌ tasks/algebra_tutor.py NOT found"
    EXIT_CODE=1
fi
echo ""

echo "6. Checking Base Checkpoint"
echo "----------------------------"
if [ -d "base_checkpoints/d32" ]; then
    echo "✅ base_checkpoints/d32 directory exists"
    
    CHECKPOINT_SIZE=$(du -sh base_checkpoints/d32 | cut -f1)
    echo "   Checkpoint size: $CHECKPOINT_SIZE"
    
    # List checkpoint files
    echo "   Files:"
    ls -lh base_checkpoints/d32/ | tail -n +2 | awk '{print "   - " $9 " (" $5 ")"}'
    
    # Check if model.pt exists
    if [ -f "base_checkpoints/d32/model.pt" ]; then
        echo "✅ model.pt found"
    else
        echo "⚠️  model.pt not found (checkpoint structure may differ)"
    fi
else
    echo "❌ base_checkpoints/d32 directory NOT found"
    echo "   You need to download the d32 checkpoint"
    EXIT_CODE=1
fi
echo ""

echo "7. Checking Disk Space"
echo "----------------------"
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
echo "Available disk space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt 100 ]; then
    echo "⚠️  Less than 100GB available (may run out during training)"
    EXIT_CODE=1
elif [ "$AVAILABLE_GB" -lt 150 ]; then
    echo "⚠️  Less than 150GB available (recommended: 200GB+)"
else
    echo "✅ Sufficient disk space"
fi
echo ""

echo "8. Checking rustbpe Build"
echo "-------------------------"
if [ -f "rustbpe/target/release/librustbpe.so" ] || [ -f "rustbpe/target/release/librustbpe.dylib" ]; then
    echo "✅ rustbpe tokenizer is built"
else
    echo "⚠️  rustbpe tokenizer not built"
    echo "   Run: cd rustbpe && cargo build --release"
fi
echo ""

echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - Ready for training!"
else
    echo "❌ SOME CHECKS FAILED - Fix issues before proceeding"
fi
echo "========================================================================"
echo ""

exit $EXIT_CODE