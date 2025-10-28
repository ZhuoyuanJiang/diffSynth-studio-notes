#!/bin/bash

################################################################################
# RenderMe360 Training Environment Setup Script
#
# This script automatically creates all necessary directories and symlinks
# based on your config_renderme360.sh settings
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "RenderMe360 Training Environment Setup"
echo "=========================================="
echo ""

# Source configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/config_renderme360.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: config_renderme360.sh not found!${NC}"
    echo "Expected location: $CONFIG_FILE"
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"
echo ""

# Check if configuration was edited
if [[ "$SOURCE_SERVER" == "xxx.xxx.xxx.xxx" ]]; then
    echo -e "${RED}ERROR: Please edit config_renderme360.sh first!${NC}"
    echo "You need to set SOURCE_SERVER to the actual server IP/hostname."
    echo "Edit: $CONFIG_FILE"
    exit 1
fi

echo -e "${GREEN}Configuration loaded successfully!${NC}"
echo "  Source server: $SOURCE_SERVER"
echo "  Local storage: $LOCAL_STORAGE_BASE"
echo ""

# Function to create directory
create_directory() {
    local dir=$1
    local description=$2

    if [ -d "$dir" ]; then
        echo -e "${YELLOW}✓ Already exists:${NC} $dir"
    else
        echo "Creating $description: $dir"

        if mkdir -p "$dir" 2>/dev/null; then
            echo -e "${GREEN}✓ Created:${NC} $dir"
        else
            echo -e "${RED}✗ Failed to create:${NC} $dir"
            echo ""
            echo "Permission denied. You don't have write access to this path."
            echo ""
            echo "Solutions:"
            echo "  1. Edit config_renderme360.sh"
            echo "  2. Change LOCAL_STORAGE_BASE to a path you can write to"
            echo "  3. Check available storage: df -h"
            echo "  4. Run this script again"
            echo ""
            exit 1
        fi
    fi
}

echo "=========================================="
echo "Step 1: Creating Local Directories"
echo "=========================================="
echo ""

# Create base storage directory
create_directory "$LOCAL_STORAGE_BASE" "storage base"

# Create dataset directory
create_directory "$LOCAL_DATASET_PATH" "dataset directory"

# Create model directory
create_directory "$LOCAL_MODEL_BASE" "model base directory"
create_directory "$LOCAL_MODEL_PATH" "model directory"

# Create HF cache directory
create_directory "$HF_CACHE_PATH" "HuggingFace cache"

# Create training output directories
create_directory "$TRAINING_OUTPUT_BASE" "training output base"
create_directory "$TEST_OUTPUT_PATH" "test output directory"
create_directory "$FULL_OUTPUT_PATH" "full training output directory"

echo ""
echo -e "${GREEN}✓ All directories created successfully!${NC}"
echo ""

echo "=========================================="
echo "Step 2: Creating Model Symlink"
echo "=========================================="
echo ""

# Project root is one level up from setup/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYMLINK_TARGET="${LOCAL_MODEL_BASE}/Wan-AI"
SYMLINK_SOURCE="${PROJECT_ROOT}/models/Wan-AI"

if [ -L "$SYMLINK_SOURCE" ]; then
    echo -e "${YELLOW}✓ Symlink already exists:${NC} $SYMLINK_SOURCE"
    echo "  Points to: $(readlink -f $SYMLINK_SOURCE)"
else
    echo "Creating symlink..."
    echo "  From: $SYMLINK_SOURCE"
    echo "  To:   $SYMLINK_TARGET"

    ln -s "$SYMLINK_TARGET" "$SYMLINK_SOURCE"
    echo -e "${GREEN}✓ Symlink created successfully!${NC}"
fi

echo ""

echo "=========================================="
echo "Step 3: Configuring Environment Variables"
echo "=========================================="
echo ""

BASHRC="$HOME/.bashrc"
ENV_VAR_MARKER="# RenderMe360 Training Environment Variables"

# Check if already configured
if grep -q "$ENV_VAR_MARKER" "$BASHRC" 2>/dev/null; then
    echo -e "${YELLOW}✓ Environment variables already configured in ~/.bashrc${NC}"
    echo "  (If you changed paths, please manually update ~/.bashrc)"
else
    echo "Adding environment variables to ~/.bashrc..."

    cat >> "$BASHRC" << EOF

$ENV_VAR_MARKER
export HF_HOME="${HF_CACHE_PATH}"
export CUDA_HOME=\$CONDA_PREFIX
export TOKENIZERS_PARALLELISM=false
EOF

    echo -e "${GREEN}✓ Environment variables added to ~/.bashrc${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Run this to apply changes:${NC}"
    echo "  source ~/.bashrc"
fi

echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ All directories created${NC}"
echo -e "${GREEN}✓ Model symlink configured${NC}"
echo -e "${GREEN}✓ Environment variables set${NC}"
echo ""
echo "Next steps:"
echo "  1. Apply environment variables: source ~/.bashrc"
echo "  2. Transfer data: bash transfer_renderme360_data.sh"
echo "  3. Install conda environment: conda env create -f environment_renderme360.yml"
echo "  4. Run smoke test: bash examples/wanvideo/model_training/lora/run_renderme360_test.sh"
echo ""
echo "Directory structure created:"
echo "  Dataset: $LOCAL_DATASET_PATH"
echo "  Models:  $LOCAL_MODEL_PATH"
echo "  Outputs: $TRAINING_OUTPUT_BASE"
echo ""
