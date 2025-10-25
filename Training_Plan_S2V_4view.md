# Training Plan: Wan2.2-S2V-14B Multi-View Fine-tuning on RenderMe360

**Date**: 2025-10-22
**Last Updated**: 2025-10-22 (Changed to single→multi approach per mentor confirmation)

**Rationale**: See `Training_Plan_S2V_4view_Rationale.md` for the decision-making process and reasoning behind this plan.

---

## Approach A: Single-View Image → Multi-View Video (Novel View Synthesis + Animation)

**Goal**: Train model to take a **single frontal face image** + audio → generate animated **2×2 grid video** showing 4 camera angles

**Configuration**:
- **Input**: Single-view image (from cam_54 frontal hemisphere) + audio
- **Output**: Video with 4-camera 2×2 grid layout
- **Resolution**: 224×416 per camera → **448×832 total grid** (original S2V resolution) ✅
  - *Alternative (future)*: 280×520 per camera → 560×1040 total (+56% pixels, documented below)
- **Frames & Timing**: 81 frames @ 16fps = **5.0 seconds** per clip (formula: (81-1)/16 = 5.0s)
- **Audio**: 16kHz mono, 5.0 second segments aligned with video frames
- **Input camera**: cam_54 (frontal hemisphere view)
- **Output cameras**: cam_28 (top-left), cam_37 (top-right), cam_49 (bottom-left), cam_54 (bottom-right) in 2×2 grid
- **Training**: **LoRA fine-tuning** ✅ (see rationale below), then full parameter tuning if needed
- **Timeline**: Start training by end of tomorrow (2025-10-23)

**Training Method Decision: LoRA vs Full Fine-tuning**

**Arguments for Full Fine-tuning:**
- ✅ **Novel capability, not refinement**: View synthesis is fundamentally different from what S2V was trained for
- ✅ **LoRA may lack capacity**: Typical LoRA (rank 32) may not learn to generate 3 new camera angles well
- ✅ **Higher success probability**: Research precedent shows novel view synthesis models use full fine-tuning
- ✅ **We have the compute**: 4-8 × 48GB GPUs can handle full fine-tuning
- **Estimated time**: 3-4 days

**Our Decision: Start with LoRA** ✅
- **Rationale**: Want to validate approach first before committing to 3-4 day full training
- **VRAM uncertainty**: Unsure if full fine-tuning fits in 48GB per GPU
- **Faster feedback**: 1.5-2 days to see if novel view synthesis is working
- **Fallback plan**: If LoRA quality poor after 10k steps → switch to full fine-tuning
- **LoRA config**: rank 32, target modules "q,k,v,o,ffn.0,ffn.2"

**Why 448×832 (original S2V resolution)?**
- ✅ **Non-invasive fine-tuning**: Matches pre-trained model's resolution exactly
- ✅ **Lower risk**: Position encodings, spatial features, attention patterns all match
- ✅ **Faster training**: 56% fewer pixels than 560×1040 alternative
- ✅ **Better for deadline**: Minimizes variables, faster iteration
- ✅ **VRAM safe**: ~25-32GB per GPU (comfortable margin on 48GB GPUs)

**What the model learns**:
- **Novel view synthesis**: Generate 3 unseen camera angles (cam_28, cam_37, cam_49) from 1 input (cam_54)
- **Speech-driven animation**: Animate all 4 views synchronized with audio
- **Spatial consistency**: Keep all 4 views geometrically consistent throughout animation

---

## Phase 1: Data Preparation

### **Recommended: Option A - On-the-Fly Loading** ✅

**Advantages**:
- No storage overhead (keep only original 440GB dataset)
- Fast iteration - change resolution/cameras instantly without reprocessing
- Easy debugging - fix code and restart, no wasted preprocessing time
- Flexible experimentation with different configurations
- **Critical for timeline**: Can start training immediately after code is ready

**Implementation**:

#### 1.1 Create Metadata CSV
Generate a simple CSV pointing to original data with clip definitions:

```csv
subject,performance,start_frame_30fps,num_frames,audio_path,input_camera,prompt
0026,s1_all,0,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
0026,s1_all,40,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
0026,s1_all,80,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
0026,s1_all,120,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
...
```

- **Script**: `scripts/generate_metadata_single2multi.py`
- **Process**:
  - Scan all subjects/performances in `/ssd4/zhuoyuan/renderme360_4cam/`
  - Calculate number of 81-frame clips per performance
  - Generate CSV with clip start positions
  - Always use cam_54 as input camera
  - From avg 1324 frames → ~15 clips per performance
  - Total: 126 performances × 15 clips = **~1,890 training samples**

#### 1.2 Create Custom Dataset Class
- **File**: `diffsynth/trainers/renderme360_dataset.py` (new file)
- **Class**: `RenderMe360S2VDataset` (simpler than UnifiedDataset for row-level loading)
- **Methods to implement**:

  1. **`LoadSingleCameraImage`**: Load single camera frame for input
     - Load frame **start_frame_30fps** from specified camera (cam_54) - the first 30fps frame of the clip
     - Resize to 224×416 (matches S2V original resolution per camera)
     - Return as PIL Image

  2. **`LoadMultiViewGridVideo`**: Load 4 camera frames and arrange in 2×2 grid for each timestep
     - **30fps → 16fps sampling**: `idx_30fps = start_frame_30fps + ((k*30 + 8)//16)` for k in [0..80]
     - For each frame in clip (81 frames):
       - Load frames from `cam_28`, `cam_37`, `cam_49`, `cam_54`
       - **Check file exists** before loading (fail fast if missing frame)
       - Resize each to 224×416
       - Arrange in 2×2 grid: 448×832 total
     - Return as list of PIL Images (video frames)

  3. **`LoadAudioSegment`**: Load audio segment on-the-fly
     - Load full audio file with torchaudio
     - Extract segment based on `start_frame_30fps` and `num_frames` (81 frames = 5.0s at 16fps)
     - Duration formula: (num_frames - 1) / fps = 80 / 16 = 5.0 seconds
     - Resample to 16kHz mono if needed
     - **Enforce exactly 80,000 samples** (crop or pad) and return as float32
     - Return waveform + sample rate (16000)

**Note**: 81 frames + 5.0s audio is correct - 81 timestamps span 80 intervals = 5.0s total (no misalignment).

#### 1.3 DataLoader Configuration
Add passthrough collate function (dataset returns PIL Images + numpy arrays which default collate can't handle):

```python
def passthrough_collate(batch):
    """For batch_size=1, return the single sample directly."""
    assert len(batch) == 1, "This pipeline assumes batch_size=1"
    return batch[0]

train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=passthrough_collate,  # Essential for PIL/numpy data
)

#### 1.4 Dataset Usage
```python
from diffsynth.trainers.renderme360_dataset import RenderMe360S2VDataset

dataset = RenderMe360S2VDataset(
    base_path="/ssd4/zhuoyuan/renderme360_4cam/",
    metadata_csv="metadata_single2multi.csv",
    cameras=["cam_28", "cam_37", "cam_49", "cam_54"],
    repeat=100  # Dataset repetitions per epoch
)
```

**Storage**: Only original 440GB dataset needed ✅

**Camera Layout in 2×2 Grid (448×832 total)**:
```
┌─────────────┬─────────────┐
│   cam_28    │   cam_37    │  Top row (224×416 each)
│ (wide front)│  (profile)  │
├─────────────┼─────────────┤
│   cam_49    │   cam_54    │  Bottom row (224×416 each)
│   (rear)    │  (frontal)  │
└─────────────┴─────────────┘
Total resolution: 448 (height) × 832 (width)
Per camera: 224 (height) × 416 (width)
```

---

### **Alternative Approaches (Documented for Future)**

#### **Alternative 1: Higher Resolution (560×1040)**

Use higher resolution for better quality:
- **Per camera**: 280×520
- **Total grid**: 560×1040
- **Total pixels**: 582,400 (+56% vs original S2V)
- **Pros**: Better face detail, higher quality output
- **Cons**:
  - More invasive fine-tuning (position encodings mismatch)
  - Slower training (~1.3-1.5× slower)
  - Higher VRAM (~35-40GB per GPU vs ~25-32GB)
  - More variables = higher risk
- **When to use**: Phase 2, after 448×832 approach succeeds and quality is validated

**Rationale for not using now**:
- Already doing very hard task (novel view synthesis)
- Don't add resolution change on top
- Timeline constraint (end of tomorrow)
- Want non-invasive fine-tuning for first iteration

#### **Alternative 2: Use All 4 Cameras as Input (Data Augmentation)**

Instead of always using cam_54, randomly select input camera for each sample:
- **Total samples**: ~1,890 × 4 = ~7,560 samples
- **Pros**: Model learns view synthesis from any angle, more robust
- **Cons**: 4× longer training, much harder problem, unlikely to finish by deadline
- **When to use**: Phase 2, after cam_54-only approach succeeds

#### **Alternative 3: Preprocessing Approach**

**Use this if**:
- On-the-fly loading causes CPU bottleneck
- DataLoader workers become the bottleneck
- Need maximum training speed

**Process**:
1. Pre-extract 81-frame clips as 2×2 grid videos (MP4)
2. Pre-extract first frame from cam_54 as input images (JPG)
3. Pre-extract audio segments (MP3, 16kHz)
4. Generate metadata CSV pointing to preprocessed files

**Storage**: `/ssd1/zhuoyuan/renderme360_single2multi_processed/` (~150-200GB)

---

## Phase 2: Training Code Setup

### 2.1 Create/Modify Training Script
- **File**: `examples/wanvideo/model_training/train_s2v.py` (create from train.py)
- **Key Modifications**:
  1. Load audio_processor: `ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/")`
  2. Update `extra_inputs="input_image,input_audio,audio_sample_rate"`
  3. Use `RenderMe360S2VDataset` with passthrough collate function
  4. Set up S2V model files (DiT, T5, wav2vec2, VAE)
  5. Skip pose guidance: no `s2v_pose_video` input

### 2.2 Create LoRA Training Script
- **File**: `examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360-Single2Multi.sh`
- **Config**:
  ```bash
  accelerate launch --config_file examples/wanvideo/model_training/accelerate_config_4gpu.yaml \
    examples/wanvideo/model_training/train_s2v.py \
    --dataset_base_path /ssd4/zhuoyuan/renderme360_4cam \
    --dataset_metadata_path /ssd4/zhuoyuan/renderme360_4cam/metadata_single2multi.csv \
    --height 448 --width 832 --num_frames 81 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
    --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/Wan2.2-S2V-14B_renderme360_single2multi_lora" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --extra_inputs "input_image,input_audio,audio_sample_rate" \
    --use_gradient_checkpointing_offload
  ```

### 2.4 Create Accelerate Config
- **File**: `examples/wanvideo/model_training/accelerate_config_4gpu.yaml`
- **Config**: 4-8 GPU distributed training with DeepSpeed or FSDP

---

## Phase 3: Training Execution

### 3.1 Verify HuggingFace Cache Location
**CRITICAL**: Before downloading models, verify cache is NOT in home directory:
```bash
echo $HF_HOME  # Should be /ssd1/zhuoyuan/hf_cache
```
If not set, add to training script:
```bash
export HF_HOME=/ssd1/zhuoyuan/hf_cache
```

### 3.2 Small-Scale Validation (2-4 hours)
**Test on 1 subject first to catch issues early**:
- Use subject 0026 only (6 performances, ~90 samples)
- Train for 1 epoch with `--dataset_repeat 1`
- Verify:
  - Data loads correctly (single cam_54 input + 2×2 grid output)
  - Audio segments extracted properly (16kHz, 5.0s duration)
  - Memory usage within limits (~25-32GB per GPU)
  - Loss decreases (even slightly)
  - No OOM errors

### 3.2 Full LoRA Training (2-3 days with 4-8 GPUs)
- **Command**:
  ```bash
  accelerate launch --config_file examples/wanvideo/model_training/accelerate_config_4gpu.yaml \
    examples/wanvideo/model_training/train_s2v.py <args from 2.3>
  ```
- **Expected iterations**: ~1,890 samples × 100 repeats × 5 epochs = ~945,000 steps
- **Estimated time**: 2-3 days (with 4-8 GPUs)
- **Output**: LoRA weights (~200MB)
- **Checkpoints**: Save every 10,000 steps

### 3.3 Full Parameter Training (Optional, after LoRA succeeds)
- After LoRA success, train full model
- Remove `--lora_base_model` and add `--trainable_models "dit"`
- Estimated time: 5-7 days
- Output: Full model weights (~28GB)

---

## Phase 4: Inference & Validation

### 4.1 Create Inference Script
- **File**: `examples/wanvideo/model_inference/Wan2.2-S2V-14B-RenderMe360-Single2Multi.py`
- **Input**:
  - Single-view image (frontal face, any resolution → will resize to 224×416)
  - Audio file (any sample rate → will resample to 16kHz)
- **Output**: Animated 2×2 grid video (448×832, 16fps, 81 frames)
- **Process**:
  1. Load LoRA weights
  2. Resize input image to 224×416
  3. Load and resample audio to 16kHz
  4. Generate 2×2 grid video with S2V pipeline

### 4.2 Test on Held-out Data
- Reserve 2-3 subjects for testing (e.g., 0297, 0295, 0290)
- Generate videos from cam_54 input images
- Evaluate:
  - **View consistency**: Do all 4 views look like the same person?
  - **Temporal consistency**: Smooth motion, no flickering?
  - **Lip sync**: Does mouth movement match audio?
  - **Geometric consistency**: Do views maintain 3D structure?

### 4.3 Qualitative Evaluation
- Generate videos for diverse test cases:
  - Different subjects (various face shapes, genders)
  - Different audio (calm speech, expressive speech)
  - Different lighting conditions
- Compare with baseline (if available) or original videos

---

## Timeline: Start Training by End of Tomorrow (2025-10-23)

### Today (2025-10-22):
- [x] Finalize training plan ✅
- [ ] Generate metadata CSV (`scripts/generate_metadata_single2multi.py`)
- [ ] Implement custom dataset (`diffsynth/trainers/renderme360_dataset.py`)
- [ ] Test data loading on 1 sample
- [ ] Create training script (`train_s2v.py`)
- [ ] Create LoRA script (`Wan2.2-S2V-14B-RenderMe360-Single2Multi.sh`)

### Tomorrow (2025-10-23):
- [ ] Verify HF_HOME cache location
- [ ] Run small-scale validation (1 subject, 2-4 hours)
- [ ] Fix any issues found
- [ ] **Launch full LoRA training** 🚀

### Next 2-3 days:
- [ ] Monitor training progress
- [ ] Save checkpoints
- [ ] Debug if needed

### After training completes:
- [ ] Create inference script
- [ ] Test on held-out subjects
- [ ] Evaluate quality

---

## Key Technical Details

### Resolution Matching:
- **Training**: 448×832 (2×2 grid of 224×416 cameras) - **Original S2V resolution** ✅
- **Inference**: 448×832 (MUST match training)
- **Why this resolution**: Non-invasive fine-tuning, matches pre-trained model exactly
- If users want different output resolution, use post-processing upscaling
- *Alternative for future*: 560×1040 (documented above)

### Input/Output Matching:
- **Training input**: Single cam_54 image (224×416)
- **Training output**: 2×2 grid video (448×832, 81 frames)
- **Inference input**: Single frontal face image (resized to 224×416)
- **Inference output**: 2×2 grid video (448×832, 81 frames)
- ✅ Input distribution matches!
- ✅ Resolution matches pre-trained S2V model!

### Audio Processing:
- **Original audio**: MP3, various sample rates
- **Required**: 16kHz mono for S2V model (wav2vec2 requirement)
- **Segment length**: 5.0 seconds (81 frames @ 16fps, duration = 80/16)
- **Processing**: torchaudio load + resample on-the-fly

### Number of Frames:
- **Chosen**: 81 frames per clip
- **Why 81?**
  - S2V model constraint: `num_frames % 4 == 1` (from model architecture)
  - Valid options: 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, **81**, 85, 97, ...
  - 81 frames @ 16fps = ~5 seconds (good length for speech, not too short/long)
  - Standard in S2V examples (common choice for talking head generation)
  - Good balance: enough temporal context to learn motion, fits in memory

### Frame Rate (fps):
- **S2V model fps**: 16 fps (recommended by model)
- **RenderMe360 fps**: 30 fps (native capture rate)
- **Challenge**: Need to downsample 30fps → 16fps while maintaining audio-video sync
  - Ratio: 30 / 16 = 1.875 (not a whole number, makes frame selection use alternating steps)
  - **Critical**: Audio must be extracted for the EXACT time span of selected frames to maintain lip sync
  - 81 frames at 16fps = **5.0 seconds** duration (formula: (81-1)/16 = 80/16 = 5.0s)
  - Frame timestamps: [0/16, 1/16, 2/16, ..., 80/16] = [0.0, 0.0625, 0.125, ..., 5.0] seconds
- **Implementation**: Use nearest-frame rounding:
  ```python
  # For each 16fps frame (k in [0..80]), find nearest 30fps frame
  # Using integer arithmetic to avoid platform-dependent rounding
  idx_30fps = start_frame_30fps + ((k * 30 + 8) // 16)
  ```

### Camera Selection Rationale:
- **Input**: cam_54 (frontal hemisphere) - matches typical user photos
- **Output grid**: All 4 cameras (cam_28, cam_37, cam_49, cam_54)
- **Alternative**: Can extend to all 4 input cameras in Phase 2 for robustness

---

## Key Files to Create

### Scripts (3 files):
1. `scripts/generate_metadata_single2multi.py` - Generate metadata CSV
2. `diffsynth/trainers/renderme360_dataset.py` - Custom dataset class
3. `examples/wanvideo/model_training/train_s2v.py` - Training script

### Config files (2 files):
4. `examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360-Single2Multi.sh` - LoRA training script
5. `examples/wanvideo/model_training/accelerate_config_4gpu.yaml` - Accelerate config (may already exist)

### Inference (1 file):
6. `examples/wanvideo/model_inference/Wan2.2-S2V-14B-RenderMe360-Single2Multi.py` - Inference script

**Total**: 6 new files + modifications to existing files

---

## Risk Mitigation

### High Risk: Novel View Synthesis May Not Work Well
- **Issue**: Generating 3 unseen views from 1 input is very hard
- **Mitigation**:
  - Start with LoRA (faster iteration)
  - Monitor generated samples early
  - If quality poor, fall back to multi→multi approach
  - Consider adding view synthesis pre-training

### Medium Risk: CPU Bottleneck from On-the-Fly Loading
- **Issue**: Loading 4 cameras + grid creation might slow down training
- **Mitigation**:
  - Use `num_workers=8` in DataLoader
  - Monitor GPU utilization (should be >90%)
  - If CPU bottlenecks, switch to preprocessing approach

### Low Risk: VRAM Overflow
- **Issue**: Even at 448×832, might approach 48GB per GPU limit
- **Mitigation**:
  - Use gradient checkpointing (enabled by default)
  - Reduce batch size if needed
  - Monitor VRAM usage in small-scale validation
  - Expected: ~25-32GB per GPU (comfortable margin)

---

## Success Metrics

### Training Metrics:
- [ ] Loss converges (decreases steadily)
- [ ] No OOM errors during training
- [ ] Training throughput: >1 batch per 5 seconds
- [ ] Checkpoints save successfully

### Quality Metrics:
- [ ] Generated views look like the same person
- [ ] Temporal smoothness (no flickering)
- [ ] Lip sync with audio
- [ ] Geometric consistency across views
- [ ] Comparable quality to baseline (if available)

---

## Notes & Open Questions

1. **Pose guidance**: Currently skipped. Add in Phase 2 if head motion looks unrealistic.
2. **Prompt strategy**: Using generic "a person speaking". Can enhance with subject-specific descriptions later.
3. **Validation frequency**: Save checkpoints every 10,000 steps, validate every 50,000 steps.
4. **Learning rate scheduling**: Start with constant 1e-4, add warmup/decay if needed.
