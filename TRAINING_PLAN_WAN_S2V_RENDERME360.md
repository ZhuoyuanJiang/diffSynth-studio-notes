# Training Plan: Wan S2V Fine-tuning on RenderMe360 Dataset

**Project**: Fine-tune Wan2.2-S2V-14B on RenderMe360 multi-view speech dataset
**Date**: 2025-10-02
**Dataset Location**: `/ssd4/zhuoyuan/renderme360_temp/test_download/subjects/`
**Dataset Size**: 2.1TB, 21 subjects, 126 performances, 166,911 frames

---

## 1. Model Requirements Analysis

### Wan S2V Input Requirements
- **`input_audio`**: Audio waveform (16kHz recommended)
- **`input_image`**: First frame/reference image
- **`prompt`**: Text description
- **`s2v_pose_video`** (optional): Pose guidance video
- **`num_frames`**: 81 frames (constraint: `num_frames % 4 == 1`)
- **Resolution**: 448x832 or 480x832 (16-pixel divisible)

### Model Architecture Components
1. **Audio Encoder**: wav2vec2-large-xlsr-53-english
2. **DiT**: Main diffusion transformer (14B parameters)
3. **Text Encoder**: T5-XXL
4. **VAE**: Video encoder/decoder
5. **Motion Controller**: Uses 73 frames of motion context

**Model Files Required**:
```
Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors
Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth
Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors
Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth
```

---

## 2. Dataset Compatibility Assessment

### ✅ What We Have (RenderMe360)
- **Multi-view synchronized videos**: 20 cameras per performance
- **Audio tracks**: MP3 format, synchronized with video
- **High-resolution images**: 2448x2048 (can resize to training resolution)
- **Rich diversity**: 21 subjects × 6 performances = 126 unique sequences
- **3D Keypoints**: Available for pose extraction
- **Total frames**: 166,911 frames → 3,338,220 images (20 cameras)

### ⚠️ Gaps and Solutions

| Gap | Impact | Solution |
|-----|--------|----------|
| **No text prompts** | Required for training | Generate generic prompts ("A person speaking") or use auto-captioning (BLIP/LLaVA) |
| **Variable video lengths** | Training needs 81 frames | Sample 81-frame clips from longer videos |
| **Audio format verification** | Must be 16kHz | Verify and resample if needed |
| **Pose videos** | Optional but useful | Extract from 3D keypoints or skip initially |

---

## 3. Data Preprocessing Pipeline

### Step 1: Video Clip Extraction
```
Input: 2448x2048 videos (varying lengths: 688-3126 frames)
Process:
  1. Resize frames to 448x832 (or 480x832)
  2. Sample 81 consecutive frames with stride
  3. From avg 1324 frames → ~31 clips per performance
Output: 126 performances × 31 clips × 20 cameras = ~78,000 clips
```

### Step 2: Audio Segment Alignment
```
Input: Full audio MP3 files
Process:
  1. Extract audio segments matching video clips
  2. Verify/resample to 16kHz
  3. Duration: ~5.4 seconds per clip (81 frames @ 15fps)
Output: Synchronized audio clips
```

### Step 3: Metadata CSV Creation
```csv
video,audio,prompt,input_image,s2v_pose_video
subjects/0026/s1_all/video_cam0_clip0.mp4,subjects/0026/s1_all/audio_clip0.mp3,"A person speaking",subjects/0026/s1_all/frame_0_cam0.jpg,subjects/0026/s1_all/pose_clip0.mp4
subjects/0026/s1_all/video_cam0_clip1.mp4,subjects/0026/s1_all/audio_clip1.mp3,"A person speaking",subjects/0026/s1_all/frame_0_cam0.jpg,subjects/0026/s1_all/pose_clip1.mp4
...
```

### Step 4: Dataset Statistics
- **Training samples**: ~78,000 (all 20 cameras) or ~55,000 (14 frontal cameras)
- **Storage requirement**: ~500GB (compressed clips)
- **Storage location**: `/ssd1/zhuoyuan/` or `/ssd2/zhuoyuan/` (NOT /home/)

---

## 4. Training Configuration

### Recommended: LoRA Fine-tuning

**Advantages**:
- Less compute intensive
- Faster iteration
- Smaller checkpoints (~200MB vs full model)
- Compatible with base model

**Configuration**:
```bash
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /ssd1/zhuoyuan/renderme360_processed \
  --dataset_metadata_path /ssd1/zhuoyuan/renderme360_processed/metadata.csv \
  --data_file_keys "video,audio,s2v_pose_video" \
  --height 448 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \
  --audio_processor_path "Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-S2V-14B_renderme360_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video" \
  --use_gradient_checkpointing_offload
```

**Hardware Requirements**:
- **GPU**: 8×80GB (based on Wan2.2-Animate-14B requirements)
- **Training time**: 3-5 days estimated
- **Total iterations**: ~39M steps (78k samples × 100 repeats × 5 epochs)

---

## 5. Implementation Roadmap

### Phase 1: Data Preparation (3-4 days)
- [ ] **Day 1-2**: Write preprocessing scripts
  - Video clip extraction (81 frames)
  - Audio segment extraction and resampling
  - First frame extraction for input_image
  - (Optional) Pose video generation

- [ ] **Day 2-3**: Generate metadata CSV
  - Create prompt strategy (generic or auto-generated)
  - Build UnifiedDataset-compatible CSV
  - Validate data paths and formats

- [ ] **Day 3-4**: Data validation
  - Test data loading with UnifiedDataset
  - Verify audio is 16kHz
  - Check frame counts and resolutions

### Phase 2: Training Setup (1 day)
- [ ] Modify `examples/wanvideo/model_training/train.py` for S2V
  - Add audio processing to `WanTrainingModule`
  - Configure `extra_inputs` for S2V
  - Set up audio_processor loading

- [ ] Create accelerate config for 8×80GB setup
- [ ] Test on small subset (1 subject, 1 camera)

### Phase 3: Training (3-5 days)
- [ ] Launch full training run
- [ ] Monitor loss curves and memory usage
- [ ] Save checkpoints every N steps
- [ ] Validate intermediate results

### Phase 4: Evaluation (2-3 days)
- [ ] Test inference on held-out subjects
- [ ] Compare with baseline Wan2.2-S2V
- [ ] Analyze multi-view consistency
- [ ] Generate demo videos

---

## 6. Key Technical Decisions

### Decision 1: Camera Selection Strategy
**Options**:
- **Option A**: All 20 cameras (~78k samples) - Maximum diversity
- **Option B**: 14 frontal cameras (~55k samples) - Faster, face-focused
- **Recommendation**: Start with Option B, expand to Option A if needed

### Decision 2: Prompt Strategy
**Options**:
- **Generic**: "A person speaking" for all samples
- **Style-based**: "A person speaking calmly/expressively" (manual annotation)
- **Auto-generated**: Use BLIP2/LLaVA to caption first frames
- **Recommendation**: Start with generic, enhance if time permits

### Decision 3: Pose Conditioning
**Options**:
- **Skip initially**: Set `s2v_pose_video=None`
- **Use 3D keypoints**: Render pose videos from existing keypoints
- **Recommendation**: Skip initially for faster iteration

### Decision 4: Training Mode
**Options**:
- **LoRA (Recommended)**: `--lora_base_model "dit" --lora_rank 32`
- **Full fine-tuning**: `--trainable_models "dit"`
- **Recommendation**: LoRA for efficiency

---

## 7. Critical Challenges & Mitigations

| Challenge | Risk | Mitigation |
|-----------|------|------------|
| **8×80GB GPU requirement** | May not have enough GPUs | Use gradient checkpointing, reduce batch size, or use smaller model variant |
| **2.1TB dataset size** | Storage/loading bottleneck | Preprocess to compressed format, use /ssd drives, load on-the-fly |
| **Audio format compatibility** | Training failures | Verify 16kHz early, create resampling pipeline |
| **Missing prompts** | Suboptimal training | Start with generic prompts, acceptable for speech task |
| **VRAM overflow** | OOM errors | Enable `--use_gradient_checkpointing_offload`, reduce tile size |

---

## 8. Success Metrics

### Training Metrics
- [ ] Loss convergence (target: < 0.1 by epoch 5)
- [ ] No memory overflow during training
- [ ] Checkpoints saved successfully

### Quality Metrics
- [ ] Generated videos are temporally coherent
- [ ] Lip sync matches audio
- [ ] Motion naturalness comparable to base model
- [ ] Multi-view consistency (if training on multiple cameras)

### Practical Metrics
- [ ] Inference speed acceptable (< 2 min per video)
- [ ] LoRA weights compatible with base model
- [ ] Can generate for unseen subjects

---

## 9. File Structure (Proposed)

```
/ssd1/zhuoyuan/renderme360_processed/
├── metadata.csv                          # Main training metadata
├── videos/                               # 81-frame video clips
│   ├── 0026_s1_cam00_clip000.mp4
│   ├── 0026_s1_cam00_clip001.mp4
│   └── ...
├── audios/                               # Audio segments (16kHz)
│   ├── 0026_s1_clip000.mp3
│   ├── 0026_s1_clip001.mp3
│   └── ...
├── images/                               # First frames
│   ├── 0026_s1_cam00_frame0.jpg
│   └── ...
└── poses/ (optional)                     # Pose videos
    ├── 0026_s1_cam00_clip000_pose.mp4
    └── ...
```

---

## 10. Next Actions (Prioritized)

### Immediate (This Week)
1. ✅ Verify audio is 16kHz: `librosa.load(audio_path, sr=None)`
2. ✅ Write video clip extraction script
3. ✅ Write audio alignment script
4. ✅ Create metadata CSV generator
5. ✅ Test data loading with one sample

### Short-term (Next Week)
6. Modify training script for S2V inputs
7. Run training on small subset (validation)
8. Launch full training run
9. Monitor and debug

### Long-term (Week 3-4)
10. Evaluate results
11. Iterate on prompts/data if needed
12. Generate demo videos
13. Write technical report

---

## 11. References & Learning Resources

### Key Scripts to Study (in order):
1. `examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py` - Understand S2V inference
2. `diffsynth/pipelines/wan_video_new.py` (line 972-1052) - S2V pipeline unit implementation
3. `examples/wanvideo/model_training/train.py` - Base training script
4. `diffsynth/trainers/unified_dataset.py` - Dataset format requirements
5. `examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh` - LoRA training example

### Documentation:
- DiffSynth-Studio README: `/home/zhuoyuan/projects/DiffSynth-Studio/README.md`
- Wan Video examples: `/home/zhuoyuan/projects/DiffSynth-Studio/examples/wanvideo/`
- This project's CLAUDE.md: `/home/zhuoyuan/projects/DiffSynth-Studio/CLAUDE.md`

---

## Appendix: Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Total subjects | 21 |
| Performances per subject | 6 |
| Cameras per performance | 20 |
| Total performances | 126 |
| Average frames per performance | 1,324 |
| Total frames | 166,911 |
| Total images (20 cams) | 3,338,220 |
| Average audio duration | 44.2s |
| Image resolution | 2448x2048 |
| Dataset size | 2.1TB |
| Estimated clips (81 frames) | ~78,000 |
| Estimated processed size | ~500GB |
