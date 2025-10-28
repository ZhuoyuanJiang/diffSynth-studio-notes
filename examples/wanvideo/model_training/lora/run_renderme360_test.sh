#!/bin/bash
#
# RenderMe360 Training - Smoke Test (8 GPUs, 1 epoch, 100 samples)
# Purpose: Quick validation of training pipeline before full run
#

set -e

echo "========================================"
echo "RenderMe360 Training - SMOKE TEST"
echo "========================================"

# Activate conda environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Server-specific paths (vllab15)
DATASET_BASE="/ssd4/zhuoyuan/renderme360_4cam"
METADATA_PATH="${DATASET_BASE}/metadata_single2multi.csv"
OUTPUT_DIR="/ssd1/zhuoyuan/diffsynth_training/renderme360_test"

# Verify dataset and metadata
if [ ! -d "$DATASET_BASE" ]; then
    echo "ERROR: Dataset not found at $DATASET_BASE"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "ERROR: Metadata not found at $METADATA_PATH"
    exit 1
fi

# Check GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

if [ "$GPU_COUNT" -lt 8 ]; then
    echo ""
    echo "WARNING: Only $GPU_COUNT GPUs detected. This script expects 8 GPUs."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_BASE"
echo "  Metadata: $METADATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: 8 (DeepSpeed ZeRO-2)"
echo "  Epochs: 1"
echo "  Samples: First 100"
echo ""
read -p "Press Enter to start smoke test (Ctrl+C to cancel)..."

# Create test metadata with first 100 samples and adjust num_frames to 17
TEST_METADATA="/tmp/renderme360_test_metadata.csv"
head -101 "$METADATA_PATH" > "$TEST_METADATA"  # Header + 100 rows
# Replace num_frames column from 81 to 17 for memory efficiency
sed -i 's/,81,/,17,/g' "$TEST_METADATA"
echo "Created test metadata: $TEST_METADATA (100 samples, 17 frames)"

# Run training with 8 GPUs (using custom ZeRO-3 config for memory efficiency)
accelerate launch \
  --config_file examples/wanvideo/model_training/lora/accelerate_config_renderme360.yaml \
  examples/wanvideo/model_training/train_renderme360.py \
  --dataset_base_path "$DATASET_BASE" \
  --dataset_metadata_path "$TEST_METADATA" \
  --height 224 \
  --width 416 \
  --num_frames 17 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_config "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio" \
  --use_gradient_checkpointing_offload \
  --save_steps 100

echo ""
echo "========================================"
echo "Smoke test completed!"
echo "Check output at: $OUTPUT_DIR"
echo ""
echo "If successful, run full training with:"
echo "  bash examples/wanvideo/model_training/lora/run_renderme360_full.sh"
echo "========================================"
