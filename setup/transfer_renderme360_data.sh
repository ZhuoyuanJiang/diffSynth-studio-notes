#!/bin/bash

################################################################################
# RenderMe360 Data Transfer Script
#
# Automatically transfers dataset and models from source server
# using rsync based on config_renderme360.sh settings
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "RenderMe360 Data Transfer"
echo "=========================================="
echo ""

# Source configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/config_renderme360.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: config_renderme360.sh not found!${NC}"
    exit 1
fi

echo "Loading configuration..."
source "$CONFIG_FILE"
echo ""

# Check configuration
if [[ "$SOURCE_SERVER" == "xxx.xxx.xxx.xxx" ]]; then
    echo -e "${RED}ERROR: Please edit config_renderme360.sh first!${NC}"
    echo "You need to set SOURCE_SERVER to the actual server IP/hostname."
    exit 1
fi

# Check if directories exist
if [ ! -d "$LOCAL_DATASET_PATH" ] || [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Local directories not created yet!${NC}"
    echo "Please run: bash setup_renderme360.sh first"
    exit 1
fi

echo -e "${GREEN}Configuration loaded:${NC}"
echo "  Source: ${SOURCE_USER}@${SOURCE_SERVER}"
echo "  Dataset: $SOURCE_DATASET_PATH → $LOCAL_DATASET_PATH"
echo "  Models:  $SOURCE_MODEL_PATH → $LOCAL_MODEL_PATH"
echo ""

# Test SSH connection
echo "=========================================="
echo "Testing SSH Connection"
echo "=========================================="
echo ""
echo "Testing connection to ${SOURCE_USER}@${SOURCE_SERVER}..."

if ssh -o BatchMode=yes -o ConnectTimeout=5 "${SOURCE_USER}@${SOURCE_SERVER}" "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH connection successful!${NC}"
else
    echo -e "${YELLOW}⚠ SSH connection failed or requires password${NC}"
    echo ""
    echo "You may need to:"
    echo "  1. Set up SSH keys for passwordless login"
    echo "  2. Or enter password when prompted during rsync"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

echo ""

# Transfer dataset
echo "=========================================="
echo "Step 1: Transferring Dataset (~50GB)"
echo "=========================================="
echo ""
echo "This will take 30-60 minutes depending on network speed..."
echo ""
echo "From: ${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_DATASET_PATH}"
echo "To:   ${LOCAL_DATASET_PATH}"
echo ""

echo "Choose transfer method:"
echo "  1) rsync (recommended - can resume if interrupted)"
echo "  2) scp (alternative)"
read -p "Enter choice (1 or 2): " -n 1 -r TRANSFER_METHOD
echo ""
echo ""

if [[ $TRANSFER_METHOD =~ ^[12]$ ]]; then
    if [[ $TRANSFER_METHOD == "1" ]]; then
        echo "Using rsync..."
        rsync -avz --progress \
            "${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_DATASET_PATH}/" \
            "${LOCAL_DATASET_PATH}/"
    else
        echo "Using scp..."
        scp -r "${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_DATASET_PATH}/" \
            "${LOCAL_DATASET_PATH}/"
    fi

    echo ""
    echo -e "${GREEN}✓ Dataset transfer complete!${NC}"

    # Verify dataset
    echo ""
    echo "Verifying dataset..."
    if [ -f "${METADATA_PATH}" ]; then
        SAMPLE_COUNT=$(wc -l < "${METADATA_PATH}")
        echo -e "${GREEN}✓ Metadata found: $SAMPLE_COUNT lines (should be 1049)${NC}"
    else
        echo -e "${RED}✗ WARNING: Metadata file not found!${NC}"
    fi

    if [ -d "${LOCAL_DATASET_PATH}/0001" ]; then
        echo -e "${GREEN}✓ Sample subject 0001 found${NC}"
    else
        echo -e "${RED}✗ WARNING: Sample subject directory not found!${NC}"
    fi
else
    echo "Skipping dataset transfer."
fi

echo ""

# Transfer models
echo "=========================================="
echo "Step 2: Transferring Models (~43GB)"
echo "=========================================="
echo ""
echo "This will take 30-60 minutes depending on network speed..."
echo ""
echo "From: ${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_MODEL_PATH}"
echo "To:   ${LOCAL_MODEL_PATH}"
echo ""

echo "Choose transfer method:"
echo "  1) rsync (recommended - can resume if interrupted)"
echo "  2) scp (alternative)"
read -p "Enter choice (1 or 2): " -n 1 -r TRANSFER_METHOD
echo ""
echo ""

if [[ $TRANSFER_METHOD =~ ^[12]$ ]]; then
    if [[ $TRANSFER_METHOD == "1" ]]; then
        echo "Using rsync..."
        rsync -avz --progress \
            "${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_MODEL_PATH}/" \
            "${LOCAL_MODEL_PATH}/"
    else
        echo "Using scp..."
        scp -r "${SOURCE_USER}@${SOURCE_SERVER}:${SOURCE_MODEL_PATH}/" \
            "${LOCAL_MODEL_PATH}/"
    fi

    echo ""
    echo -e "${GREEN}✓ Model transfer complete!${NC}"

    # Verify models
    echo ""
    echo "Verifying models..."

    REQUIRED_FILES=(
        "diffusion_pytorch_model-00001-of-00004.safetensors"
        "diffusion_pytorch_model-00002-of-00004.safetensors"
        "diffusion_pytorch_model-00003-of-00004.safetensors"
        "diffusion_pytorch_model-00004-of-00004.safetensors"
        "text_encoder/model.safetensors"
        "vae/diffusion_pytorch_model.safetensors"
        "wav2vec2-large-xlsr-53-english/pytorch_model.bin"
    )

    ALL_FOUND=true
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${LOCAL_MODEL_PATH}/${file}" ]; then
            echo -e "${GREEN}✓${NC} $file"
        else
            echo -e "${RED}✗${NC} $file (MISSING!)"
            ALL_FOUND=false
        fi
    done

    if [ "$ALL_FOUND" = true ]; then
        echo ""
        echo -e "${GREEN}✓ All required model files verified!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Some model files are missing! Please check the transfer.${NC}"
    fi
else
    echo "Skipping model transfer."
fi

echo ""

echo "=========================================="
echo "Transfer Summary"
echo "=========================================="
echo ""
echo "Dataset location: $LOCAL_DATASET_PATH"
echo "Model location:   $LOCAL_MODEL_PATH"
echo ""

# Check disk space
echo "Disk space used:"
du -sh "$LOCAL_DATASET_PATH" 2>/dev/null || echo "  Dataset: Not yet transferred"
du -sh "$LOCAL_MODEL_PATH" 2>/dev/null || echo "  Models: Not yet transferred"
echo ""

echo "Next steps:"
echo "  1. Install environment: conda env create -f environment_renderme360.yml"
echo "  2. Activate environment: conda activate diffsynth-s2v"
echo "  3. Install DiffSynth: pip install -e ."
echo "  4. Run smoke test: bash examples/wanvideo/model_training/lora/run_renderme360_test.sh"
echo ""
