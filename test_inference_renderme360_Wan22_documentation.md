# Wan2.2-S2V Inference Testing with RenderMe360 Dataset

**Date:** 2025-10-27
**Server:** vllab14
**Status:** ✅ Inference pipeline validated with our own dataset - Ready for training integration

---

## Goal & Context

**Goal:** Validate our data pipeline and inference workflow before starting training.

**Task:**
- Test if our RenderMe360 dataset can be loaded correctly
- Run Wan2.2-S2V inference using our dataset samples
- Verify that the pretrained model accepts our data format and the environment
- Generate baseline videos to make sure everything is correct and for comparison

**Reference:** referenced `test_inference_Wan22_documentation.md` for original Wan2.2-S2V inference setup details.

---

## What We Accomplished

### 1. Fixed Input Camera Selection

**Problem:**
- Originally used cam_54 as input camera
- User discovered cam_28 is the actual frontal view

**Solution:**
Updated 3 files to use cam_28 instead of cam_54:

**`scripts/generate_metadata_single2multi.py`:**
- Line 15: `camera="cam_28"` (was cam_54)
- Line 28: `reference_camera="cam_28"` (was cam_54)
- Line 94: `input_camera="cam_28"` (was cam_54)

**`diffsynth/trainers/renderme360_operators.py`:**
- Line 146: `input_camera="cam_28"` (default parameter in `LoadRenderMe360InputImage`)

**Result:**
- Regenerated `/ssd4/zhuoyuan/renderme360_4cam/metadata_single2multi.csv`
- 1,048 training samples, all using cam_28 as input camera

### 2. Validated Data Operators

**Ran:** `python test_renderme360_operators.py`

**Test Results:**
```
✓ Sample 0, 1, 2 - All passed
✓ Video: 81 frames (832×448 2×2 grid)
✓ Input image: (416×224) - cam_28 frontal view
✓ Audio: 81,000 samples @ 16kHz (5.0625s)
✓ Prompt: "a person speaking"
```

**Generated Files:**
- `test_output_grid_frame0.png` - Shows 2×2 grid with 4 camera views (cam_28, cam_37, cam_49, cam_54)
- `test_output_input_image.png` - Shows single cam_28 frontal view

**Key Finding:** All 3 test samples loaded correctly with correct shapes and content.

### 3. Created and Ran Inference Validation Script

**Script:** `test_inference_with_renderme360.py`

**What it does:**
1. Loads sample 0 from RenderMe360UnifiedDataset
2. Extracts input_image (cam_28) and audio (5-second segment)
3. Runs Wan2.2-S2V pretrained model inference
4. Saves generated video and ground truth video for comparison

**Inference Configuration:**
- Model: Wan2.2-S2V-14B (pretrained, not fine-tuned)
- GPU: Single RTX 6000 Ada (48GB)
- VRAM management: Enabled (layer-wise CPU offloading for single GPU)
- Inference steps: 40
- Runtime: ~17 minutes (~25.6s/step with VRAM management)
- Resolution: 448×832 (2×2 grid)
- Frames: 81 @ 16fps = 5.0625 seconds

**Successfully Generated:**
```
✓ test_renderme_input_image.png - Input cam_28 image
✓ test_renderme_ground_truth_frame0.png - Ground truth frame 0 (2×2 grid)
✓ test_renderme_generated_frame0.png - Generated frame 0 from pretrained model
✓ test_renderme_ground_truth.mp4 - Ground truth 2×2 grid video (81 frames with audio)
✓ test_renderme_generated.mp4 - Generated video from pretrained model (80 frames with audio)
✓ temp_audio.wav - Temporary audio file (16kHz, 5.0625s)
```

**Validation Results:**
- ✅ Data pipeline works correctly
- ✅ Model accepts our data format (416×224 input image, 81000-sample audio)
- ✅ Audio correctly loaded and merged with video
- ✅ Pretrained model generated video successfully
- ✅ Both videos have sound and are playable

**Note:** Frame 0 of generated video is the reference input image, so saved video uses frames 1-80 (80 frames total).

---

## Issues Discovered

### Image Cropping Due to Aspect Ratio Mismatch

**Observation:**
- Original images: 1024×1024 (square, aspect ratio 1.0)
- Target size: 224×416 (height×width, aspect ratio 0.538)
- Significant aspect ratio mismatch

**Current Processing:**
Located in `diffsynth/trainers/renderme360_operators.py:155-167`

```python
def _resize_to_target(self, img, target_h, target_w):
    """Resize image to target size maintaining aspect ratio with center crop."""
    w, h = img.size
    scale = max(target_w / w, target_h / h)  # Scale to cover target

    # Step 1: Resize 1024×1024 → 416×416 (to fit width)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Step 2: Center crop 416×416 → 416×224
    # This removes 192 pixels from top and bottom
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    return img
```

**What This Means:**
The center crop removes **46% of the image height** (192 out of 416 pixels):
- **96 pixels chopped off the top** → Missing the top of the head
- **96 pixels chopped off the bottom** → Missing the neck and part of the shoulders
- **Result:** The input image shows only a cropped face view, not the full head and upper body

**Why This Happens:**
- Original images are square (1024×1024)
- Target needs to be wide (416 width × 224 height) to fit 4 cameras in a 2×2 grid
- To maintain aspect ratio without distortion, the code scales to fit width first (→ 416×416)
- Then center-crops to target height (→ 416×224), which chops off nearly half the vertical content

**Impact:**
- May affect model quality since it sees less facial/body context
- May affect novel view synthesis capability (missing spatial information)
- Standard computer vision approach, but loses potentially important content

**Decision:**
**Postponed to later.** Prioritize getting training pipeline working first.

**Alternative Solutions (for future consideration):**
1. **Keep center crop** (current) - Standard CV practice but loses content
2. **Aspect-ratio-preserving resize + padding** - Keep full face, add black bars on sides
3. **Stretch/squash resize** - Keep full content but distorts proportions
4. **Use different target resolution** - e.g., 448×448 (square) to match original aspect ratio

---

## Technical Details

### Data Flow

```
RenderMe360 Dataset (raw)
  ├── 21 subjects (0026, 0041, 0048, ..., 0297)
  ├── 126 performances (6 per subject: s1_all through s6_all)
  ├── 4 cameras per frame (cam_28, cam_37, cam_49, cam_54)
  ├── 30fps @ 1024×1024 per camera
  └── 16kHz audio

      ↓ [generate_metadata_single2multi.py]

metadata_single2multi.csv
  ├── 1,048 training clips
  ├── 81 frames per clip @ 16fps = 5.0625s
  ├── input_camera: cam_28 (frontal view)
  ├── stride: 150 frames @ 30fps (0% overlap)
  └── Frame mapping: 16fps → 30fps using (k*30+8)//16

      ↓ [RenderMe360UnifiedDataset]

Training Sample (dict)
  ├── "video": List[PIL.Image] - 81 frames of 832×448 2×2 grid
  ├── "input_image": PIL.Image - 416×224 cam_28 frontal view
  ├── "input_audio": np.ndarray - [81000] samples @ 16kHz
  ├── "prompt": str - "a person speaking"
  ├── "audio_sample_rate": int - 16000
  └── "metadata": dict - {subject, performance, start_frame_30fps, num_frames}

      ↓ [WanVideoPipeline]

Generated Video
  └── 81 frames @ 448×832 with synchronized audio
```

### Key Code Files

**Data Preparation:**
- `scripts/generate_metadata_single2multi.py` - Metadata CSV generation (updated to use cam_28)
- `/ssd4/zhuoyuan/renderme360_4cam/metadata_single2multi.csv` - 1,048 training samples

**Data Loading:**
- `diffsynth/trainers/renderme360_operators.py` - 4 custom data operators:
  - `LoadRenderMe360GridVideo` - Creates 2×2 grid from 4 cameras on-the-fly
  - `LoadRenderMe360InputImage` - Loads cam_28 input image (updated to default cam_28)
  - `LoadRenderMe360Audio` - Loads 5-second audio segment aligned with video
  - `LoadRenderMe360Prompt` - Loads text prompt
- `diffsynth/trainers/renderme360_unified_dataset.py` - PyTorch Dataset wrapper

**Validation and Testing:**
- `test_renderme360_operators.py` - Data operator validation script (✅ passed)
- `test_inference_with_renderme360.py` - Inference validation script (✅ passed)

---

## File Inventory

### Created This Session

**Scripts:**
```
test_inference_with_renderme360.py    - Inference validation script
```

**Validation Outputs:**
```
test_output_grid_frame0.png           - Data operator test: 2×2 grid frame 0
test_output_input_image.png           - Data operator test: cam_28 input image
test_renderme_input_image.png         - Inference test: input cam_28
test_renderme_ground_truth_frame0.png - Inference test: ground truth frame 0
test_renderme_generated_frame0.png    - Inference test: generated frame 0
test_renderme_ground_truth.mp4        - Inference test: ground truth video (81 frames with audio)
test_renderme_generated.mp4           - Inference test: generated video (80 frames with audio)
temp_audio.wav                        - Temporary audio file (16kHz, 81000 samples)
```

**Documentation:**
```
test_inference_renderme360_Wan22_documentation.md - This file
```

---

## Environment Details

**Server:** vllab14
**GPUs:** 8 × NVIDIA RTX 6000 Ada Generation (48GB each)
**Conda env:** `diffsynth-s2v`
**PyTorch:** 2.4.0+cu118
**CUDA:** Available, 8 GPUs detected

---

**Last Updated:** 2025-10-27 01:55 (vllab14 local time)
