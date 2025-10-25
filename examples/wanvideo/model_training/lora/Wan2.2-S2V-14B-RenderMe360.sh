#!/bin/bash
#
# Wan2.2-S2V-14B LoRA Training on RenderMe360 Dataset
# Single-view input (cam_54) → Multi-view output (2×2 grid) with audio
#
# Hardware: 8× RTX 6000 Ada (48GB each) on vllab9
# Dataset: /ssd2/zhuoyuan/renderme360_4cam/
#

# Verify HuggingFace cache location
echo "HF_HOME: ${HF_HOME:-Not set}"
if [ -z "$HF_HOME" ]; then
    echo "WARNING: HF_HOME not set. Models will download to ~/.cache/huggingface"
    echo "Set HF_HOME to avoid filling home directory quota."
fi

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train_s2v.py \
  --dataset_base_path /ssd2/zhuoyuan/renderme360_4cam \
  --dataset_metadata_path /ssd2/zhuoyuan/renderme360_4cam/metadata_single2multi.csv \
  --height 448 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-S2V-14B_RenderMe360_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio,audio_sample_rate" \
  --use_gradient_checkpointing_offload \
  --save_steps 10000

# Notes:
# - No --data_file_keys: Using custom RenderMe360S2VDataset
# - extra_inputs: input_image (cam_54), input_audio (5.0s @ 16kHz), audio_sample_rate (16000)
# - LoRA target modules: May need adjustment based on actual DiT module names
# - Checkpoint every 10k steps to avoid I/O overhead
# - Training will print LoRA module matches at startup - verify >0 matched
