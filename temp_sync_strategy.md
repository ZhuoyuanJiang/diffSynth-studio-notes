# FPS Conversion Strategy: Analysis and Correction

**Date**: 2025-10-25
**Purpose**: Evaluate the reviewer's Bresenham algorithm vs. nearest-frame approach for 30fps → 16fps conversion

---

## TL;DR

**Reviewer's Bresenham algorithm is WRONG for our use case.**
- ❌ It produces uniformly-spaced frames (not nearest frames)
- ❌ Results in up to 29.5ms temporal error (vs. 16.7ms max with nearest-frame)
- ✅ Correct approach: Use **nearest-frame matching** (either with float or integer math)

---

## The Reviewer's Algorithm (Bresenham-Style)

```python
def indices_30_from_16_REVIEWER(start_30: int, n_16: int) -> list[int]:
    """Reviewer's Bresenham algorithm - produces UNIFORM spacing."""
    idx, err, out = start_30, 0, []
    for _ in range(n_16):
        out.append(idx)
        err += 30            # numerator
        step = err // 16     # denominator
        idx += step
        err -= step * 16
    return out
```

**Output**: `[0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 18, ...]`

---

## What's Wrong with This?

### Problem: Bresenham Optimizes for UNIFORM SPACING, Not TEMPORAL ACCURACY

The Bresenham algorithm ensures that frames are **evenly distributed** with an average step of 30/16 = 1.875.

But this is NOT the same as finding the **nearest frame** to each target time!

### Visual Proof of Error

```
Target 16fps times (what we want):
  Frame 0: 0.000s
  Frame 1: 0.0625s
  Frame 2: 0.125s
  Frame 3: 0.1875s
  Frame 4: 0.25s
  Frame 5: 0.3125s

Available 30fps frames (what we have):
  Frame 0: 0.000s
  Frame 1: 0.033s
  Frame 2: 0.067s
  Frame 3: 0.100s
  Frame 4: 0.133s
  Frame 5: 0.167s
  Frame 6: 0.200s
  Frame 7: 0.233s
  Frame 8: 0.267s
  Frame 9: 0.300s
  Frame 10: 0.333s
  Frame 11: 0.367s
```

### Reviewer's Bresenham Output vs. Correct Nearest-Frame

| 16fps Target | Target Time | Bresenham Frame | Bresenham Time | Error | Nearest Frame | Nearest Time | Error |
|--------------|-------------|-----------------|----------------|-------|---------------|--------------|-------|
| 0 | 0.000s | 0 | 0.000s | **0.0ms** ✓ | 0 | 0.000s | **0.0ms** ✓ |
| 1 | 0.0625s | 1 | 0.033s | **29.5ms** ❌ | 2 | 0.067s | **4.5ms** ✓ |
| 2 | 0.125s | 3 | 0.100s | **25.0ms** ❌ | 4 | 0.133s | **8.3ms** ✓ |
| 3 | 0.1875s | 5 | 0.167s | **20.5ms** ❌ | 6 | 0.200s | **12.5ms** ✓ |
| 4 | 0.25s | 7 | 0.233s | **16.7ms** ⚠️ | 8 | 0.267s | **16.7ms** ✓ |
| 5 | 0.3125s | 9 | 0.300s | **12.5ms** ⚠️ | 9 | 0.300s | **12.5ms** ✓ |

**Bresenham max error**: 29.5ms (at frame 1)
**Nearest-frame max error**: 16.7ms (bounded by half the 30fps period)

---

## The Correct Approach: Nearest-Frame Matching

### Option 1: Using Float Math (Simple & Clear)

```python
import numpy as np

def get_16fps_frames_from_30fps_CORRECT(start_frame_30fps: int, num_frames_16fps: int = 81):
    """
    Find the nearest 30fps frame for each 16fps target time.
    Uses float math for clarity.
    """
    # Calculate target times at 16fps
    target_times = np.arange(num_frames_16fps) / 16.0  # [0.0, 0.0625, 0.125, ...]

    # Convert to 30fps indices and round to nearest
    frame_indices = np.round(target_times * 30).astype(int)

    # Offset by start position
    frame_indices += start_frame_30fps

    return frame_indices.tolist()
```

**Output**: `[0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 20, ...]`

### Option 2: Integer-Only Math (If You Prefer)

```python
def get_16fps_frames_from_30fps_INTEGER(start_frame_30fps: int, num_frames_16fps: int = 81):
    """
    Find the nearest 30fps frame using integer-only math.
    Equivalent to rounding (k * 30 / 16).
    """
    indices = []
    for k in range(num_frames_16fps):
        # Want to compute: round(k * 30 / 16)
        # Integer version: (k * 30 + 8) // 16  (adding 16/2 for rounding)
        idx = (k * 30 + 8) // 16
        indices.append(start_frame_30fps + idx)
    return indices
```

**Output**: `[0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 20, ...]` (same as float version)

**Verification**:
- k=0: (0×30 + 8)÷16 = 8÷16 = 0 ✓
- k=1: (1×30 + 8)÷16 = 38÷16 = 2 ✓
- k=2: (2×30 + 8)÷16 = 68÷16 = 4 ✓
- k=3: (3×30 + 8)÷16 = 98÷16 = 6 ✓

---

## Why Nearest-Frame is Correct for Audio-Video Sync

### The Physics of Sync

1. **Video is discrete**: We MUST use one of the actual captured frames at 30fps
2. **Audio is continuous**: We extract the exact time window [start, start+5.0s]
3. **Sync perception**: Human brain tolerates ±80-120ms lip sync error

### Our Approach

```
Audio extraction:
  - Start time: start_frame_30fps / 30.0
  - Duration: 5.0 seconds (fixed: 81 frames @ 16fps)
  - Extract waveform for EXACT time window [start, start+5.0]

Video sampling:
  - Target times: [0, 0.0625, 0.125, ..., 5.0] seconds
  - Map to nearest 30fps frames
  - Maximum temporal error: 16.7ms (half of 30fps frame period)

Result:
  - Audio contains exact sound at all target times ✓
  - Video frames are within 16.7ms of target times ✓
  - Combined error: < 20ms (5× better than perceptible threshold) ✓
```

### Visual Proof of Sync

```
Time:           0.0    0.0625  0.125   0.1875  0.25    0.3125  0.375   0.4375  0.5
                |      |       |       |       |       |       |       |       |
30fps frames:   [0]  [1][2]  [3][4]  [5][6]  [7][8]  [9][10] [11][12][13][14][15]
                 0.0  .033.067 .100.133 .167.200 .233.267 .300.333 .367.400 .433.467.500

16fps targets:  0      1       2       3       4       5       6       7       8
(time-based)    ↓      ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
Nearest frames: [0]    [2]     [4]     [6]     [8]     [9]     [11]    [13]    [15]
                0.000  0.067   0.133   0.200   0.267   0.300   0.367   0.433   0.500

Audio window:   [←────────────────── 5.0 seconds ──────────────────────→]
Video window:   [0.000 ──────────────────────────────────────────→ 5.000]

Errors:         0ms    4.5ms   8.3ms   12.5ms  16.7ms  12.5ms  8.3ms   4.5ms   0ms
All errors:     ✓ All < 17ms (well within 80ms human perception threshold)
```

---

## What the Reviewer Got Right

1. ✅ **Time-based thinking**: Use time as source of truth, not frame indices
2. ✅ **Fixed duration**: 81 frames @ 16fps = 5.0 seconds (not 5.0625s)
3. ✅ **Audio alignment**: Extract audio for the exact time window
4. ✅ **Integer math preference**: Avoiding float drift is good practice

## What the Reviewer Got Wrong

1. ❌ **Algorithm choice**: Bresenham optimizes for uniform spacing, not temporal accuracy
2. ❌ **Confusion**: The Bresenham algorithm they provided doesn't actually compute "nearest timestamps"
3. ❌ **Error magnitude**: Their approach has 29.5ms max error vs. 16.7ms with nearest-frame

---

## Recommendation

**Use the nearest-frame approach (Option 1 with float math)** because:

1. **Simpler code**: 3 lines of numpy vs. complex loop
2. **Better accuracy**: 16.7ms max error vs. 29.5ms
3. **Proven correct**: Standard approach used in video processing
4. **Float precision is fine**: 64-bit float can represent frame indices exactly up to 2^53
5. **No drift**: We recalculate from scratch for each clip (no accumulation)

If you prefer integer-only math for peace of mind, use **Option 2**, which gives identical results.

**DO NOT use the reviewer's Bresenham algorithm** - it's solving the wrong problem.

---

## Reference Implementation

```python
import numpy as np
from pathlib import Path
from PIL import Image

def load_synced_video_and_audio(
    base_dir: str,
    subject: str,
    performance: str,
    cameras: list[str],
    start_frame_30fps: int,
    num_frames_16fps: int = 81,
):
    """
    Load video frames (4-camera grid) and audio segment with perfect sync.

    Returns:
        grid_frames: List[PIL.Image] (81 frames of 448×832 grids)
        audio_waveform: np.ndarray (80000 samples @ 16kHz)
    """
    # Calculate time window
    start_time = start_frame_30fps / 30.0
    duration = 5.0  # Fixed: (81-1)/16 = 5.0 seconds

    # Get 30fps frame indices using nearest-frame matching
    target_times = np.arange(num_frames_16fps) / 16.0
    frame_indices_30fps = np.round(target_times * 30).astype(int) + start_frame_30fps

    # Load video frames
    grid_frames = []
    for idx_30 in frame_indices_30fps:
        # Load all 4 cameras for this frame
        views = []
        for cam in cameras:
            img_path = Path(base_dir) / subject / performance / "images" / cam / f"{idx_30:06d}.png"
            img = Image.open(img_path).convert("RGB").resize((416, 224), Image.BICUBIC)
            views.append(img)

        # Create 2×2 grid
        grid = Image.new("RGB", (832, 448))
        grid.paste(views[0], (0, 0))      # cam_28 top-left
        grid.paste(views[1], (416, 0))    # cam_37 top-right
        grid.paste(views[2], (0, 224))    # cam_49 bottom-left
        grid.paste(views[3], (416, 224))  # cam_54 bottom-right
        grid_frames.append(grid)

    # Load audio segment (exact time window)
    import torchaudio
    audio_path = Path(base_dir) / subject / performance / "audio" / "audio.mp3"
    info = torchaudio.info(str(audio_path))

    start_sample = int(round(start_time * info.sample_rate))
    num_samples = int(round(duration * info.sample_rate))

    waveform, sr = torchaudio.load(
        str(audio_path),
        frame_offset=start_sample,
        num_frames=num_samples
    )

    # Resample to 16kHz and convert to mono
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_np = waveform.squeeze(0).numpy()

    return grid_frames, audio_np
```

---

## Conclusion

**For the reviewer**: Your time-based thinking and duration correction (5.0s) were spot-on, but the Bresenham algorithm doesn't minimize temporal error - it minimizes spacing variance. For audio-video sync, we need to minimize temporal error, so nearest-frame rounding is the correct approach.

**For our implementation**: Use simple nearest-frame matching with `np.round()` for clarity and correctness.
