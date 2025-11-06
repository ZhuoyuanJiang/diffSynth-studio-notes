# Training Debug Session 2 - Memory Investigation & ZeRO-3 Strategy

**Date:** 2025-11-05
**Server:** vllab15 (8× RTX 6000 Ada, 48GB each)
**Goal:** Investigate OOM root cause and implement ZeRO-3 for memory efficiency
**Status:** 🔬 Investigation phase - Memory profiling plan ready

---

## Executive Summary

**Critical Finding:** The OOM issue is NOT a bug - it's an architectural limitation of DeepSpeed ZeRO-2!

**Key Insight:** ZeRO-2 does NOT shard model parameters - only optimizer states and gradients. With a 42.7GB model, each GPU holds the full model, leaving only ~5GB for activations and overhead. This is insufficient.

**Solution:** Implement ZeRO-3, which shards model parameters across GPUs, reducing per-GPU model memory from 42.7GB to ~5.3GB (42.7÷8).

**Next Steps:**
1. Add memory profiling to understand exact breakdown
2. Implement ZeRO-3 configuration with proper initialization
3. Debug tensor mismatch errors if they occur
4. Alternative: selective component sharding or LoRA rank reduction

---

## Situation Analysis

### What We've Accomplished ✅

**Environment Setup:**
- Server: vllab15 with 8× RTX 6000 Ada (48GB each)
- PyTorch: 2.4.0+cu124 (upgraded to match CUDA)
- Dataset: 1,048 RenderMe360 samples ready
- Metadata: Generated with cam_28 as input camera
- Models: 43GB downloaded and symlinked correctly

**Code Implementation:**
- `train_renderme360.py` - Custom training script
- `renderme360_unified_dataset.py` - Custom dataset with on-the-fly grid generation
- `run_renderme360_test.sh` - Smoke test launcher
- `accelerate_config_renderme360.yaml` - DeepSpeed ZeRO-2 config

**Validation:**
- ✅ Inference works (single GPU with VRAM management)
- ✅ Dataset operators validated (correct shapes, timing)
- ✅ Data loads correctly in training loop

### The OOM Problem 🚨

**Symptom:**
All 8 GPUs crash with OOM during forward pass, consistently using ~45-46GB regardless of:
- Video resolution (448×832 → 224×416)
- Number of frames (81 → 17)
- Gradient accumulation steps
- Batch size

**This is the smoking gun:** If data reduction has zero effect, the bottleneck is **model architecture**, not data.

---

## Hypothesis: Model Memory Breakdown

**Expected model size (from file sizes):**
- DiT: ~30 GB
- T5 Text Encoder: ~11 GB
- Wav2Vec2 + VAE: ~2 GB
- **Total: ~43 GB**

**With ZeRO-2:**
Each GPU holds the full model (~43GB), plus sharded optimizer states and gradients. This leaves minimal memory for activations during forward pass, causing OOM.

---

## ZeRO-2 vs ZeRO-3: The Critical Difference

### What DeepSpeed ZeRO Stages Shard

| Component | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|---------|---------|---------|
| Optimizer States | ✅ Sharded | ✅ Sharded | ✅ Sharded |
| Gradients | ❌ Replicated | ✅ Sharded | ✅ Sharded |
| **Model Parameters** | ❌ **Replicated** | ❌ **Replicated** | ✅ **Sharded** |

**This is why ZeRO-2 doesn't help with our 42.7GB model!**

### Expected Memory with ZeRO-3

With parameter sharding across 8 GPUs:
- Model per GPU: ~43 GB ÷ 8 = **~5-6 GB**
- Optimizer + gradients: ~1-2 GB (sharded)
- **Total per GPU: ~7-8 GB** (vs ~45GB with ZeRO-2)
- **Available for activations: ~40 GB** ← Should resolve OOM

### Why ZeRO-3 Failed Before

From SESSION_TRAINING_DEBUG.md (Attempt #3):
```
Attempt 3: 448×832×41 frames, ZeRO-3
Result: ❌ Tensor size mismatch error
```

**Likely causes:**
1. **VAE architecture incompatibility**: VAE might use custom operations that don't support parameter partitioning
2. **DeepSpeed initialization order**: `zero3_init_flag: false` in config may be incorrect
3. **Pipeline unit incompatibility**: Some units may expect full model on each GPU
4. **State dict mismatch**: Model loading might break with ZeRO-3 weight redistribution

---

## Investigation Plan

### Phase 1: Memory Profiling (PRIORITY) 🔍

**Goal:** Understand exact memory usage at each training stage

**Implementation:**
Add memory profiling utilities to `train_renderme360.py`:

1. **`print_gpu_memory_summary(stage_name, device_id)`**
   - Prints allocated, reserved, max allocated, free memory
   - Optional detailed summary with `torch.cuda.memory_summary()`

2. **`print_model_component_sizes(pipe)`**
   - Calculates and prints size of each component (DiT, T5, VAE, Wav2Vec2)
   - Verifies expected ~42.7GB total

**Profiling checkpoints:**
- After model loading (before DeepSpeed wrapping)
- After DeepSpeed initialization
- After first forward pass
- After first backward pass
- After optimizer step

**Expected discoveries:**
- ✅ Confirm 42.7GB model load
- ✅ Verify ZeRO-2 doesn't reduce model memory
- ✅ Identify activation memory usage
- ✅ Check for duplicate model loads or leaks

**Success criteria:** Detailed memory breakdown report showing bottleneck

---

### Phase 2: Implement ZeRO-3 with Proper Initialization 🔧

**Goal:** Enable ZeRO-3 parameter sharding and resolve tensor mismatch errors

#### Step 2.1: Create ZeRO-3 Configuration

Create `accelerate_config_renderme360_zero3.yaml` with:
- `zero_stage: 3` - Enable parameter sharding
- `zero3_init_flag: true` - Proper weight initialization
- `offload_param_device: none` - Keep params on GPU initially (CPU offload later if needed)
- `zero3_save_16bit_model: true` - Ensure checkpoint compatibility

#### Step 2.2: Update Training Script (if needed)

Check if model loading needs special handling:

```python
# May need DeepSpeed context for ZeRO-3 initialization
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# Option 1: Wrap model loading in DeepSpeed context (if needed)
# Option 2: Use `deepspeed.zero.Init(config=zero3_config)` context
```

#### Step 2.3: Test ZeRO-3 Progressively

**Test 1: Minimal ZeRO-3**
- 17 frames @ 224×416
- No CPU offload (params on GPU)
- Verify parameter sharding works
- **Expected result:** ~6-8GB per GPU

**Test 2: Add CPU Offload** (if Test 1 succeeds)
- Set `offload_param_device: cpu`
- Verify all-gather communication works
- **Expected result:** ~3-4GB per GPU

**Test 3: Increase Data Size** (if Test 2 succeeds)
- Gradually increase frames: 17 → 41 → 81
- Increase resolution: 224×416 → 448×832
- **Expected result:** Full training works

#### Step 2.4: Debug Tensor Mismatch Error (if occurs)

**Investigation steps:**
1. Check which component fails (likely VAE or image_encoder)
2. Try excluding from ZeRO-3 sharding:
   ```python
   # In DeepSpeed config, specify which parameters to shard
   "zero_force_ds_cpu_optimizer": false,
   "zero_hpz_partition_size": 1,  # Don't shard certain layers
   ```

3. Use `deepspeed.zero.partition_parameters()` context for selective sharding:
   ```python
   # Only shard DiT (the 30GB bottleneck)
   with deepspeed.zero.partition_parameters(model.pipe.dit, enabled=True):
       # DiT gets sharded
       pass

   # VAE stays replicated
   model.pipe.vae  # Not in context, stays replicated
   ```

4. Check model state dict structure:
   ```bash
   python -c "import torch; sd = torch.load('models/Wan-AI/...'); print(sd.keys())"
   ```

**Success criteria:** ZeRO-3 initializes without errors, memory drops to ~6-10GB per GPU

---

### Phase 3: Alternative Memory Optimizations ⚙️

**If ZeRO-3 still fails after debugging:**

#### Option 3.1: Reduce LoRA Rank (Quick Win)

Current: `--lora_rank 32`
```bash
# Try rank 8 (reduces trainable params by 75%)
--lora_rank 8
```

**Memory savings:**
- LoRA params: rank 32 → 8 reduces by 75%
- Gradients: Proportional reduction
- **Estimated savings:** ~1-2GB per GPU
- **Trade-off:** May need longer training

#### Option 3.2: Freeze Non-Critical Components

Train only DiT, freeze everything else:

```python
# In training script
trainable_models = ["dit"]  # Only train DiT
# Freeze: text_encoder, vae, wav2vec2
```

**Memory savings:**
- No gradients for T5 (11GB) and Wav2Vec2 (1.2GB)
- **Estimated savings:** ~1.5GB per GPU
- **Trade-off:** Less flexible fine-tuning

#### Option 3.3: Model Parallelism for T5

Place T5 encoder on separate GPU(s):

```python
# In train_renderme360.py
pipe.text_encoder.to("cuda:7")  # Dedicate GPU 7 to T5
# Train DiT on GPUs 0-6
```

**Memory savings:**
- GPU 0-6: Save 11GB each (no T5)
- GPU 7: Dedicated to T5 only
- **Trade-off:** Requires code modification, communication overhead

#### Option 3.4: Gradient Accumulation (Already Tried)

This doesn't help because model parameters dominate memory, not activations.

#### Option 3.5: Accept Hardware Limitation

**Reality check:**
- 48GB GPUs may be insufficient for 14B model training
- Upstream likely uses A100 80GB or H100 80GB
- Alternative: Inference-only fine-tuning (e.g., prompt tuning, adapter layers)

---

## Execution Checklist

### Immediate Tasks (Session 2)

- [ ] **Task 1:** Add memory profiling utilities to `train_renderme360.py`
- [ ] **Task 2:** Run smoke test with memory profiling
- [ ] **Task 3:** Analyze memory breakdown report
- [ ] **Task 4:** Create `accelerate_config_renderme360_zero3.yaml`
- [ ] **Task 5:** Run ZeRO-3 test (17 frames, no CPU offload)
- [ ] **Task 6:** Debug tensor mismatch if occurs
- [ ] **Task 7:** Test with full dataset if ZeRO-3 succeeds

### Success Criteria

**Phase 1 Success:**
- ✅ Memory profiling shows detailed breakdown
- ✅ Confirmed 42.7GB model load per GPU
- ✅ Identified activation memory requirements

**Phase 2 Success:**
- ✅ ZeRO-3 initializes without errors
- ✅ Memory usage drops to ~6-10GB per GPU
- ✅ First batch completes forward pass
- ✅ First batch completes backward pass
- ✅ Training runs for multiple iterations

**Phase 3 Success (if needed):**
- ✅ Alternative optimization reduces memory
- ✅ Training runs stable for full epoch
- ✅ Checkpoints save correctly

---

**For technical details on how ZeRO-3 works, see:** `TECHNICAL_DEEPSPEED_ZERO.md`

---

## Expected Outcomes

### Best Case Scenario
1. ZeRO-3 works out of the box
2. Memory drops to ~8GB per GPU
3. Full training (81 frames @ 448×832) runs successfully
4. LoRA checkpoints save correctly
5. **Timeline:** 1-2 hours to validate, 24-48 hours full training

### Likely Scenario
1. ZeRO-3 has tensor mismatch error
2. Identify incompatible component (VAE or image_encoder)
3. Implement selective sharding (DiT only)
4. Memory drops to ~20-25GB per GPU (sufficient)
5. Full training works with some limitations
6. **Timeline:** 4-6 hours debugging, 24-48 hours training

### Worst Case Scenario
1. ZeRO-3 incompatible with pipeline architecture
2. Fall back to LoRA rank 8 + component freezing
3. Memory reduced to ~42GB per GPU (marginal)
4. Training works with 41-49 frames max
5. Need to train on higher resolution in stages
6. **Timeline:** 8-12 hours experimentation, 48-72 hours training

---

## Files to Create/Modify

### New Files
```
examples/wanvideo/model_training/lora/accelerate_config_renderme360_zero3.yaml  # ZeRO-3 config
examples/wanvideo/model_training/lora/run_renderme360_test_zero3.sh            # ZeRO-3 test script
SESSION2_TRAINING_DEBUG.md                                                      # This file
```

### Modified Files
```
examples/wanvideo/model_training/train_renderme360.py  # Add memory profiling utilities
```

### Log Files (to generate)

These are plans, you should never open these files because they are very long.

```
memory_profiling_zero2.log  # Memory breakdown with current ZeRO-2 
memory_profiling_zero3.log  # Memory breakdown with ZeRO-3
training_zero3_test.log     # ZeRO-3 smoke test results
```

---

## Key References

**DeepSpeed Documentation:**
- ZeRO Stages: https://www.deepspeed.ai/tutorials/zero/
- ZeRO-3 Offload: https://www.deepspeed.ai/tutorials/zero-offload/
- Memory Estimation: https://deepspeed.readthedocs.io/en/latest/memory.html

**Previous Session Docs:**
- `SESSION_TRAINING_DEBUG.md` - Previous OOM investigation
- `HANDOVER_TRAINING_SESSION.md` - Training setup handover
- `test_inference_renderme360_Wan22_documentation.md` - Inference validation

**Codebase:**
- `diffsynth/pipelines/wan_video_new.py` - Pipeline architecture
- `diffsynth/trainers/utils.py` - Training utilities
- `diffsynth/vram_management/` - VRAM management (inference only)

---

## Questions to Answer During Investigation

1. **Is model loading duplicated?**
   - Check if text_encoder/vae loaded multiple times
   - Verify symlink works correctly

2. **Is gradient checkpointing working?**
   - Activation memory should be minimal
   - Check if `use_gradient_checkpointing_offload` actually offloads

3. **Are all components on correct device?**
   - Verify no accidental CPU→GPU copies
   - Check pipeline unit device placement

4. **What's the exact OOM trigger point?**
   - Which layer? Forward or backward?
   - First iteration or later?

5. **Does ZeRO-2 actually shard optimizer states?**
   - Check with memory profiling

---

## Actual Results - Memory Profiling (Phase 1)

**Date:** 2025-11-05
**Test:** ZeRO-2 baseline smoke test (7 GPUs, 17 frames @ 224×416)

### Model Component Sizes (Measured)

```
Component       Parameters          Size (bfloat16)
─────────────────────────────────────────────────────
DiT             16,295,755,609      30.35 GB
Text Encoder     5,680,910,336      10.58 GB
VAE                126,892,531       0.24 GB
─────────────────────────────────────────────────────
TOTAL           22,103,558,476      41.17 GB
```

### GPU Memory at OOM (GPU 0, ZeRO-2)

From PyTorch error message:
```
Total GPU memory:           47.50 GB
PyTorch allocated:          44.52 GB  ← Full model on each GPU
Reserved (unallocated):      1.51 GB
Free at crash:               0.10 GB (95.56 MiB)
```

**OOM Location:** VAE encode during first forward pass (tried to allocate 414 MiB)

### Hypothesis Confirmed ✅

With ZeRO-2, each of the 7 GPUs holds:
- Full model: **~44 GB** (matches 41.17 GB model + overhead)
- Free memory: **~3-4 GB** for activations
- **Result:** Insufficient → OOM during forward pass

**Next step:** Test ZeRO-3 with parameter sharding to reduce per-GPU model memory from 44GB → ~6GB.

---

---

## Session Accomplishments

### Files Created

**Documentation:**
- `SESSION2_TRAINING_DEBUG.md` - This file (investigation plan and results)
- `TECHNICAL_DEEPSPEED_ZERO.md` - Technical deep dive on ZeRO stages and all-gather

**Configuration:**
- `accelerate_config_renderme360_zero3.yaml` - DeepSpeed ZeRO-3 config with parameter sharding

**Scripts:**
- `run_renderme360_test_zero3.sh` - ZeRO-3 smoke test launcher

**Logs:** (never read logs without _temp in the name, those are too long.)
- `memory_profiling_zero2_temp.log` - ZeRO-2 baseline results (OOM at 44GB)  
- `memory_profiling_zero3_temp.log` - ZeRO-3 test results (no OOM, VAE tensor mismatch)

### Files Modified

**Training script:**
- `train_renderme360.py` - Added memory profiling utilities (`print_gpu_memory_summary`, `print_model_component_sizes`)

**Launch scripts:**
- `run_renderme360_test.sh` - Added adaptive GPU detection (auto-detects free GPUs, sets `CUDA_VISIBLE_DEVICES` and `--num_processes`)

**Configuration:**
- `accelerate_config_renderme360.yaml` - Updated for dynamic GPU count (originally hardcoded 8)

### Key Features Added

**1. Adaptive GPU Detection**
- Automatically detects free GPUs (< 1GB used)
- Sets `CUDA_VISIBLE_DEVICES` to skip occupied GPUs
- Dynamically adjusts `--num_processes` for Accelerate
- Works with 6, 7, or 8 GPUs without manual changes

**2. Memory Profiling**
- Prints model component sizes (DiT, T5, VAE) at startup
- Lightweight utilities (50 lines) in training script
- Used to confirm 41.17GB total model size

**3. ZeRO-3 Configuration**
- Parameter sharding across GPUs
- `zero3_init_flag: true` for proper initialization
- Keeps params on GPU initially (CPU offload disabled)

---

**Status:** Phase 1 complete, moving to Phase 2 (ZeRO-3)
**Last Updated:** 2025-11-05
**Next Action:** Run ZeRO-3 smoke test and verify memory reduction

**Update:** One ZeRO-3 test attempt completed. Results in `memory_profiling_zero3.log`.
