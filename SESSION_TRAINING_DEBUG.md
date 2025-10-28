# Training Debug Session - 2025-10-27

**Goal:** Launch Wan2.2-S2V-14B LoRA training on RenderMe360 dataset
**Server:** vllab15 (8× RTX 6000 Ada, 48GB each)
**Status:** ⚠️ In progress - debugging OOM issues

---

## Major Accomplishments

### 1. ✅ Environment Setup & CUDA Version Matching (CRITICAL FIX)

**Problem:** DeepSpeed reported CUDA version mismatch
```
DeepSpeed Op Builder: Installed CUDA version 12.4 does not match
the version torch was compiled with 11.8
```

**Solution:** Upgraded PyTorch to match system CUDA
```bash
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124
pip install "numpy<2.0"  # Fixed scipy compatibility
```

**Result:** ✅ Environment stable, imports working

---

### 2. ✅ Fixed Dataset Integration

**Issue 1:** Missing `load_from_cache` attribute
```python
AttributeError: 'RenderMe360UnifiedDataset' object has no attribute 'load_from_cache'
```
**Fix:** Added `self.load_from_cache = False` to dataset `__init__`

---

### 3. ❌ Out of Memory (OOM) - Primary Blocker

**Symptom:** All 8 GPUs crash with OOM during forward pass
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 300-740 MiB.
GPU allocated ~45-46 GB out of 47.5 GB total
```

**What we tried:**

| Attempt | Config | Result |
|---------|--------|--------|
| 1 | 448×832×81 frames, ZeRO-2 | ❌ OOM (~46GB used) |
| 2 | 448×832×49 frames, ZeRO-2 | ❌ OOM (~46GB used) |
| 3 | 448×832×41 frames, ZeRO-3 | ❌ Tensor size mismatch error |
| 4 | 224×416×49 frames, ZeRO-2 | ❌ OOM (~45GB used) |
| 5 | 224×416×17 frames, ZeRO-2 | ❌ OOM (~45GB used) |
| 6 | 224×416×17 frames, ZeRO-2 + gradient_accum=4 | ❌ OOM (~45GB used) |
| 7 | 224×416×17 frames, ZeRO-2 + enable_vram_management() | ❌ Device error (pipe on CPU) |

**Key finding:** Memory usage stays at ~45-46GB regardless of video resolution/frames, indicating **model parameters** are the bottleneck, not data.

---

## Current Configuration

### Training Script
- **File:** `examples/wanvideo/model_training/train_renderme360.py`
- **Dataset:** `RenderMe360UnifiedDataset` (custom on-the-fly grid generation)
- **LoRA:** rank 32, target modules: `q,k,v,o,ffn.0,ffn.2`
- **Gradient checkpointing:** Enabled with offload

### Accelerate Config
- **File:** `examples/wanvideo/model_training/lora/accelerate_config_renderme360.yaml`
- **DeepSpeed ZeRO-2:** Optimizer + params offloaded to CPU
- **GPUs:** 8 processes
- **Gradient accumulation:** 1 step
- **Mixed precision:** bfloat16

### Data Config (Smoke Test)
- **Resolution:** 224×416 (per camera view in 2×2 grid)
- **Frames:** 17 (4×4+1, ~1 second @ 16fps)
- **Samples:** 100 (first from metadata)

---

## Analysis

### Why OOM Persists

1. **14B model is huge:** ~30GB DiT + 11GB T5 + 1.2GB Wav2Vec2 = 42GB+ in bfloat16
2. **ZeRO-2 not enough:** Only offloads optimizer states, model still on GPU
3. **ZeRO-3 caused errors:** Tensor shape mismatches in VAE (architectural incompatibility?)
4. **VRAM management incompatible:** Training loads model with `device="cpu"` (for DeepSpeed), but `enable_vram_management()` requires CUDA device to query VRAM

### Why Inference Works

Inference uses:
- **Single GPU** with `enable_vram_management()`
- **No gradients** (no backward pass memory)
- **No optimizer states**
- **Sequential processing** (not distributed)

Training needs:
- **All components simultaneously** (text encoder + DiT + VAE + audio encoder)
- **Gradients + optimizer states**
- **Distributed across 8 GPUs**

---

## Files Created/Modified

### New Files (No Upstream Conflicts)
```
examples/wanvideo/model_training/train_renderme360.py         # Custom training script
examples/wanvideo/model_training/lora/run_renderme360_test.sh # Smoke test launcher
examples/wanvideo/model_training/lora/run_renderme360_full.sh # Full training launcher
examples/wanvideo/model_training/lora/accelerate_config_renderme360.yaml  # Custom DeepSpeed config
diffsynth/trainers/renderme360_unified_dataset.py             # Custom dataset
diffsynth/trainers/renderme360_operators.py                   # Data loading operators (already existed)
SESSION_TRAINING_DEBUG.md                                     # This file
```

### Modified Files (Upstream)
- ❌ None (all custom code in separate files)

---

## Next Steps (Priority Order)

### Option 1: Try Smaller LoRA Rank
```bash
--lora_rank 8  # Instead of 32, reduces trainable params by 75%
```

### Option 2: Try ZeRO-3 with Lower Frames
- Use 17 frames (current)
- Fix ZeRO-3 tensor mismatch (may need upstream investigation)

### Option 3: Switch to Smaller Model
- Try Wan2.1-I2V-1.3B instead of 14B (if available for S2V)

### Option 4: Accept Limitations
- 48GB GPUs may not be sufficient for 14B S2V training
- Upstream might use A100 80GB GPUs
- Alternative: Inference-only fine-tuning techniques (e.g., prompt tuning)

---

## Commands Reference

### Current Smoke Test
```bash
cd ~/projects/DiffSynth-Studio
bash examples/wanvideo/model_training/lora/run_renderme360_test.sh
```

### Check Environment
```bash
conda activate diffsynth-s2v
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

### Monitor Training
```bash
watch -n 1 nvidia-smi  # Real-time GPU usage
```

---

## Key Lessons

1. **CUDA version matching is critical** for DeepSpeed compilation
2. **Model size dominates memory** - data reduction has minimal effect
3. **ZeRO stages aren't drop-in replacements** - ZeRO-3 requires compatible model architecture
4. **Inference ≠ Training** - Memory requirements are vastly different
5. **Custom configs avoid upstream conflicts** - Keep all modifications in separate files

---

**Last Updated:** 2025-10-27 15:15 (vllab15)
**Current Blocker:** Out of memory with 14B model on 48GB GPUs
**Recommendation:** Try smaller LoRA rank or investigate ZeRO-3 compatibility
