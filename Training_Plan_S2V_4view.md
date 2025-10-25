# Training Plan: Wan2.2-S2V-14B Multi-View Fine-tuning on RenderMe360

**Date**: 2025-10-22

**Rationale**: See `Training_Plan_S2V_4view_Rationale.md` for the decision-making process and reasoning behind this plan.

---

## Approach B: Multi-view Grid → Animated Multi-view Grid

**Goal**: Train model to take a 2×2 grid of 4 synchronized face views + audio → generate animated 2×2 grid video

**Configuration**:
- **Input**: 2×2 grid image (560×1040: 280×520 per camera) + audio
- **Output**: Video with 4-camera 2×2 grid layout
- **Resolution**: 280×520 per camera → 560×1040 total grid
- **Cameras**: cam_28 (top-left), cam_37 (top-right), cam_49 (bottom-left), cam_54 (bottom-right)
- **Training**: LoRA fine-tuning first, then full parameter tuning

---

## Phase 1: Data Preparation

### **Recommended: Option A - On-the-Fly Loading** ✅

**Advantages**:
- No storage overhead (keep only original 440GB dataset)
- Fast iteration - change resolution/cameras instantly without reprocessing
- Easy debugging - fix code and restart, no wasted preprocessing time
- Flexible experimentation with different configurations

**Implementation**:

#### 1.1 Create Metadata CSV
Generate a simple CSV pointing to original data with clip definitions:

```csv
subject,performance,start_frame,num_frames,audio_path,prompt
0026,s1_all,0,81,0026/s1_all/audio/audio.mp3,a person speaking
0026,s1_all,40,81,0026/s1_all/audio/audio.mp3,a person speaking
0026,s1_all,80,81,0026/s1_all/audio/audio.mp3,a person speaking
0026,s1_all,120,81,0026/s1_all/audio/audio.mp3,a person speaking
...
```

- **Script**: `scripts/generate_metadata_onthefly.py`
- **Process**:
  - Scan all subjects/performances
  - Calculate number of 81-frame clips per performance
  - Generate CSV with clip start positions
  - From avg 1324 frames → ~15 clips per performance
  - Total: 126 performances × 15 clips = **~1,890 training samples**

#### 1.2 Create Custom Data Operators
- **File**: `diffsynth/trainers/renderme360_operators.py` (new file)
- **Operators to implement**:
  1. **`LoadMultiViewGrid`**: Load 4 camera frames and arrange in 2×2 grid
     - Load frames from `cam_28`, `cam_37`, `cam_49`, `cam_54`
     - Resize each to 280×520
     - Arrange in 2×2 grid: 560×1040 total
     - Return as PIL Image or list of PIL Images (for video)

  2. **`LoadAudioSegment`**: Load audio segment on-the-fly
     - Load full audio file with librosa
     - Extract segment based on `start_frame` and `num_frames`
     - Resample to 16kHz if needed
     - Return waveform + sample rate

#### 1.3 Update UnifiedDataset Configuration
```python
from diffsynth.trainers.renderme360_operators import LoadMultiViewGrid, LoadAudioSegment

dataset = UnifiedDataset(
    base_path="/ssd4/zhuoyuan/renderme360_4cam/",
    metadata_path="metadata.csv",
    data_file_keys=["video", "audio"],
    main_data_operator=LoadMultiViewGrid(...),  # For video frames
    special_operator_map={
        "audio": LoadAudioSegment(...)  # For audio segments
    }
)
```

**Storage**: Only original 440GB dataset needed ✅

---

### **Fallback: Option B - Preprocessing Approach**

**Use this if**:
- On-the-fly loading causes CPU bottleneck
- DataLoader workers become the bottleneck
- Need maximum training speed

**Process**:

#### 1.1 Create Grid Image Combiner
- **Script**: `scripts/create_multiview_grids.py`
- **Function**: Combine 4 camera views into single 2×2 grid image
  - Load 4 synchronized frames (cam_28, cam_37, cam_49, cam_54)
  - Resize each to 280×520
  - Arrange in 2×2 grid: 560×1040 total
  - Save as single JPG

#### 1.2 Extract Video Clips
- **Script**: `scripts/extract_video_clips.py`
- **Process**:
  - For each performance: extract 81-frame clips
  - Create 2×2 grid for each frame
  - Save as MP4 video (560×1040 resolution)
  - From avg 1324 frames → ~15 clips per performance
  - Total: 126 performances × 15 clips = ~1,890 training samples

#### 1.3 Extract Audio Segments
- **Script**: `scripts/extract_audio_clips.py`
- **Process**:
  - Verify audio is 16kHz (resample if not)
  - Extract audio segments aligned with video clips
  - Duration: ~5.4 seconds per clip (81 frames @ 15fps)
  - Save as MP3 or WAV (16kHz)

#### 1.4 Generate Metadata CSV
```csv
video,audio,prompt,input_image
renderme360_processed/videos/0026_s1_clip00.mp4,renderme360_processed/audio/0026_s1_clip00.mp3,"a person speaking",renderme360_processed/images/0026_s1_clip00_frame0.jpg
renderme360_processed/videos/0026_s1_clip01.mp4,renderme360_processed/audio/0026_s1_clip01.mp3,"a person speaking",renderme360_processed/images/0026_s1_clip01_frame0.jpg
...
```

**Storage**: `/ssd1/zhuoyuan/renderme360_multiview_processed/` (~150-200GB)

---

## Phase 2: Training Code Setup

### 2.1 Implement Custom Data Operators (for Option A - On-the-Fly)
- **File**: `diffsynth/trainers/renderme360_operators.py` (new file)
- **Implement**:
  - `LoadMultiViewGrid`: Load and combine 4 camera views into 2×2 grid
  - `LoadAudioSegment`: Load audio segment from full audio file
  - Integration with existing `ImageCropAndResize` and `LoadVideo` operators

**Note**: If using Option B (preprocessing), skip this step and use standard operators.

### 2.2 Update Training Script
- **File**: `examples/wanvideo/model_training/train_s2v.py` (create from train.py)
- **Modifications**:
  1. Add audio_processor loading
  2. Update `extra_inputs="input_image,input_audio,audio_sample_rate"`
  3. Add special_operator_map for audio files
  4. Configure for S2V model files

### 2.3 Create LoRA Training Script
- **File**: `examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360.sh`
- **Config**:
  ```bash
  --height 560 --width 1040 --num_frames 81
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,..."
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/"
  --lora_base_model "dit" --lora_rank 32
  --learning_rate 1e-4 --num_epochs 5
  --dataset_repeat 100
  --use_gradient_checkpointing_offload
  ```

### 2.4 Create Accelerate Config
- **File**: `examples/wanvideo/model_training/accelerate_config_4gpu.yaml`
- **Config**: 4-8 GPU distributed training

---

## Phase 3: Training Execution

### 3.1 Small-Scale Validation
- Train on 1 subject (6 performances) first
- Verify data loading, memory usage, loss convergence
- Estimated time: 2-4 hours

### 3.2 Full LoRA Training
- **Command**:
  ```bash
  accelerate launch --config_file accelerate_config_4gpu.yaml train_s2v.py ...
  ```
- **Estimated time**: 2-3 days (with 4-8 GPUs)
- **Output**: LoRA weights (~200MB)

### 3.3 Full Parameter Training (Optional)
- After LoRA success, train full model
- Remove `--lora_base_model` and add `--trainable_models "dit"`
- Estimated time: 5-7 days

---

## Phase 4: Inference & Validation

### 4.1 Create Inference Script
- **File**: `examples/wanvideo/model_inference/Wan2.2-S2V-14B-RenderMe360.py`
- **Input**: 2×2 grid image + audio
- **Output**: Animated 2×2 grid video

### 4.2 Test on Held-out Data
- Reserve 2-3 subjects for testing
- Generate videos and evaluate quality

---

## Future: Approach A (Single-view → Multi-view)

**Documentation for later**:

To train single-view → multi-view:
1. Modify preprocessing:
   - Input image: Single camera (e.g., cam_54)
   - Output video: 2×2 grid video
2. Dataset format:
   ```csv
   video,audio,prompt,input_image
   grid_video.mp4,audio.mp3,"...",single_camera_frame0.jpg
   ```
3. Training adjustments:
   - May need novel view synthesis loss
   - Consider pre-training with view synthesis model
   - Higher learning rate for initial epochs

---

## Key Files to Create

1. **Preprocessing scripts** (3-4 scripts)
2. **Training modifications** (2 files)
3. **Training shell script** (1 file)
4. **Inference script** (1 file)
5. **Documentation** (updated TRAINING_PLAN)

**Total estimated time**: 1 week (2 days preprocessing, 1 day setup, 3-4 days training)

---

## Notes & Questions to Resolve

(To be filled in as we refine the plan)
