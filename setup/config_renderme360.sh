#!/bin/bash

################################################################################
# RenderMe360 Training Configuration
#
# EDIT THESE VARIABLES ONCE - Everything else will be configured automatically
################################################################################

# ============================================================================
# SECTION 1: Source Server Info (Get from original developer)
# ============================================================================

# Source server IP or hostname (where dataset/models are stored)
SOURCE_SERVER="PASTE_SERVER_IP_HERE"  # ← CHANGE THIS: Paste the server IP/hostname here

# Source server username
SOURCE_USER="zhuoyuan"  # Usually no need to change

# Source paths (on source server)
SOURCE_DATASET_PATH="/ssd4/zhuoyuan/renderme360_4cam"
SOURCE_MODEL_PATH="/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B"


# ============================================================================
# SECTION 2: Your Local Storage Path
# ============================================================================

# Where to store data on YOUR server
#
# Option 1 (Easiest): Keep as "/ssd1/zhuoyuan"
#   → Just create this directory on your server
#   → No other files need editing
#
# Option 2 (Use your username): Change to "/ssd1/<your-username>"
#   → Replace "zhuoyuan" with YOUR actual username
#   → Example: if your username is "alice", use "/ssd1/alice"
#
# ⚠️ If /ssd1 doesn't exist on your server:
#   1. Check what you have: run "df -h" to see available storage
#   2. Use whatever fast local storage you have:
#      - For Option 1: /ssd2/zhuoyuan, /tmp2/zhuoyuan, /data/zhuoyuan, etc.
#      - For Option 2: /ssd2/<your-username>, /tmp2/<your-username>, etc.
#   3. Update the line below with your chosen path
#
LOCAL_STORAGE_BASE="/ssd1/zhuoyuan"  # ← CHANGE THIS if needed


# ============================================================================
# SECTION 3: Auto-Generated Paths (DO NOT EDIT - calculated from above)
# ============================================================================

# Local dataset path
LOCAL_DATASET_PATH="${LOCAL_STORAGE_BASE}/renderme360_4cam"

# Local model path
LOCAL_MODEL_BASE="${LOCAL_STORAGE_BASE}/diffsynth_models/models"
LOCAL_MODEL_PATH="${LOCAL_MODEL_BASE}/Wan-AI/Wan2.2-S2V-14B"

# HuggingFace cache
HF_CACHE_PATH="${LOCAL_STORAGE_BASE}/hf_cache"

# Training output paths
TRAINING_OUTPUT_BASE="${LOCAL_STORAGE_BASE}/diffsynth_training"
TEST_OUTPUT_PATH="${TRAINING_OUTPUT_BASE}/renderme360_test"
FULL_OUTPUT_PATH="${TRAINING_OUTPUT_BASE}/renderme360_lora"

# Metadata path
METADATA_PATH="${LOCAL_DATASET_PATH}/metadata_single2multi.csv"


# ============================================================================
# SECTION 4: Training Hyperparameters (Optional - can adjust if needed)
# ============================================================================

# LoRA configuration
LORA_RANK=32  # Try 8 or 4 if OOM occurs

# Resolution (smoke test uses half)
FULL_HEIGHT=448
FULL_WIDTH=832
FULL_FRAMES=81

TEST_HEIGHT=224
TEST_WIDTH=416
TEST_FRAMES=17

# Training settings
LEARNING_RATE="1e-4"
NUM_EPOCHS_TEST=1
NUM_EPOCHS_FULL=50


################################################################################
# DO NOT EDIT BELOW THIS LINE
################################################################################

export SOURCE_SERVER
export SOURCE_USER
export SOURCE_DATASET_PATH
export SOURCE_MODEL_PATH
export LOCAL_STORAGE_BASE
export LOCAL_DATASET_PATH
export LOCAL_MODEL_BASE
export LOCAL_MODEL_PATH
export HF_CACHE_PATH
export TRAINING_OUTPUT_BASE
export TEST_OUTPUT_PATH
export FULL_OUTPUT_PATH
export METADATA_PATH
export LORA_RANK
export FULL_HEIGHT
export FULL_WIDTH
export FULL_FRAMES
export TEST_HEIGHT
export TEST_WIDTH
export TEST_FRAMES
export LEARNING_RATE
export NUM_EPOCHS_TEST
export NUM_EPOCHS_FULL
