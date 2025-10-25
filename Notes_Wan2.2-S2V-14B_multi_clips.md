# Notes: Understanding Wan2.2-S2V-14B Multi-Clips Inference

**Purpose:** Document confusions and clarifications while learning the S2V inference script.

**Reference File:** `examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py`

---

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Frame Counting: infer_frames vs num_frames](#frame-counting-infer_frames-vs-num_frames)
3. [Understanding fps and Sample Rate](#understanding-fps-and-sample-rate)
4. [Motion Context: motion_frames and motion_videos](#motion-context-motion_frames-and-motion_videos)
5. [How Motion Frames Are Actually Used](#how-motion-frames-are-actually-used)
6. [Audio Processing Pipeline](#audio-processing-pipeline)
7. [The pre_calculate_audio_pose() Function](#the-pre_calculate_audio_pose-function)
8. [Frame Dropping Logic](#frame-dropping-logic)
9. [Reference Image vs Generated Frames](#reference-image-vs-generated-frames)
10. [Complete Multi-Clip Generation Example](#complete-multi-clip-generation-example)
11. [Common Confusions Clarified](#common-confusions-clarified)

---

## Core Concepts

### Required Inputs
- **prompt**: Text description (e.g., "a person is singing")
- **input_image**: Reference image (the face/person to animate)
- **audio_path**: Path to audio file

### Optional Inputs
- **pose_video_path**: Pose guidance video (can be None)
- **negative_prompt**: What to avoid in generation

### Critical Constraints
- **Audio sample rate**: Must be 16kHz
- **Video fps**: Fixed at 16 fps (do not change)
- **Frame count**: `num_frames % 4 == 1` (e.g., 81, 85, 89, 93...)

### 4 Required Model Components
1. **DiT** (Diffusion Transformer) - 14B parameters
2. **T5 Text Encoder** - Text understanding
3. **wav2vec2 Audio Encoder** - Audio processing
4. **VAE** - Video encoding/decoding

---

## Frame Counting: infer_frames vs num_frames

### **Confusion:** Why `infer_frames=80` but `num_frames=81`?

### **Answer:**

```python
infer_frames = 80  # Number of NEW frames to generate per clip
num_frames = infer_frames + 1 = 81  # Total frames INCLUDING reference frame
```

**Explanation:**
- The model needs a **reference frame** (frame 0) as the starting point
- It then generates `infer_frames` NEW frames after it
- Total: 1 reference + 80 generated = 81 frames

**The +1 satisfies the constraint:**
- Model requires: `num_frames % 4 == 1`
- 80 does NOT satisfy this (80 % 4 = 0) ❌
- 81 DOES satisfy this (81 % 4 = 1) ✅

**Valid combinations:**
```python
infer_frames = 40  → num_frames = 41  ✅
infer_frames = 80  → num_frames = 81  ✅
infer_frames = 84  → num_frames = 85  ✅
infer_frames = 160 → num_frames = 161 ✅
```

**Is infer_frames a hyperparameter?**
- **YES**, you can change it based on your needs
- Larger values = fewer clips = faster generation, more VRAM
- Smaller values = more clips = slower generation, less VRAM

**Clip duration calculation:**
```python
clip_duration = num_frames / fps
              = 81 / 16
              = 5.0625 seconds per clip
```

---

## Understanding fps and Sample Rate

### **Confusion:** What is fps? Is it for video or audio?

### **Answer:**

**fps (Frames Per Second)** = Video-only concept
- S2V uses **16 fps** (fixed, do not change)
- This means 16 video frames are displayed per second

**Sample Rate (Hz)** = Audio-only concept
- S2V requires **16000 Hz** (16 kHz)
- This means 16000 audio samples per second

**Example:**
```python
# Video:
81 frames at 16 fps = 81/16 = 5.0625 seconds of video

# Audio:
5.0625 seconds at 16000 Hz = 81000 audio samples
```

**Audio does NOT have fps!**
- Audio has sample rate (Hz)
- Video has frame rate (fps)
- These are completely different concepts

---

## Motion Context: motion_frames and motion_videos

### **Confusion 1:** What is `motion_frames=73`?

### **Answer:**

**motion_frames** = A hyperparameter defining how many recent frames to use as temporal context

```python
motion_frames = 73  # Keep last 73 frames as "motion memory"
```

**Purpose:** Ensure smooth transitions between clips

**Duration:**
```python
73 frames / 16 fps = 4.5625 seconds of motion context
```

---

### **Confusion 2:** What is `motion_videos`?

### **Answer:**

**motion_videos** = A sliding window buffer that stores the most recent 73 frames

**How it works:**
```python
# Initially empty
motion_videos = []

# After clip 0:
motion_videos = [frame_8, frame_9, ..., frame_80]  (73 frames)

# After clip 1:
motion_videos = [frame_89, frame_90, ..., frame_161]  (73 frames)
# Old frames [8-80] are DISCARDED

# After clip 2:
motion_videos = [frame_170, frame_171, ..., frame_242]  (73 frames)
# Old frames [89-161] are DISCARDED
```

**Key insight:** motion_videos is ALWAYS exactly 73 frames (no matter how many clips)

---

### **Confusion 3:** Why this line: `motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]`?

### **Answer:**

**Breaking it down:**

```python
overlap_frames_num = min(motion_frames, len(current_clip))
                   = min(73, 80)  # Usually 73
                   = 73

motion_videos = motion_videos[73:] + current_clip[-73:]
#               └─ Part A      └─ Part B
```

**Part A: `motion_videos[73:]`**
- "Keep everything after index 73"
- If motion_videos has exactly 73 frames (indices 0-72), then `[73:]` = **empty list []**
- This DISCARDS all old frames

**Part B: `current_clip[-73:]`**
- "Take the last 73 frames from current_clip"
- This KEEPS the most recent 73 frames

**Result:**
```python
motion_videos = [] + [last 73 frames] = [last 73 frames]
```

**Why this pattern?**
- Shows "sliding window" intent explicitly
- If you change `motion_frames` to a different value, code still works
- Handles edge cases (e.g., last clip might be shorter)

**Equivalent simpler form:**
```python
motion_videos = current_clip[-73:]  # Same result!
```

---

### **Confusion 4:** Why is motion_videos ALWAYS 73 frames?

### **Answer:**

**Trace through multiple clips:**

```python
# Clip 0 (r=0):
motion_videos = []  # 0 frames
current_clip has 77 frames (after processing)
motion_videos = [][73:] + [last 73 frames] = [73 frames]

# Clip 1 (r=1):
motion_videos = [73 frames]
current_clip has 80 frames
motion_videos = [73 frames][73:] + [last 73 frames]
              = []               + [73 frames]
              = [73 frames]  ✅

# Clip 2 (r=2):
motion_videos = [73 frames]
current_clip has 80 frames
motion_videos = [73 frames][73:] + [last 73 frames]
              = []               + [73 frames]
              = [73 frames]  ✅
```

**Why `[73 frames][73:]` is always empty:**
```python
my_list = [item_0, item_1, ..., item_72]  # 73 items (indices 0-72)
my_list[73:]  # "From index 73 onwards" = EMPTY []
```

**Visualization of sliding window:**
```
Time →
─────────────────────────────────────────
Clip 0: [frames 8-80]        ← motion_videos
Clip 1:                [frames 89-161]   ← motion_videos (old discarded)
Clip 2:                           [frames 170-242] ← motion_videos (old discarded)
```

---

### **Confusion 5:** How are motion frames used? I thought we only need 1 reference frame + audio to generate?

### **Answer:**

**The 73 motion frames ARE passed as INPUT to the model!**

The model doesn't just use:
- ❌ 1 reference frame + audio → generate new frames

The model actually uses:
- ✅ 1 reference frame + audio + **73 previous frames** → generate new frames

**Why?** The DiT transformer receives the 73 frames (encoded as latents) as conditioning input, enabling it to continue motion smoothly instead of restarting from scratch each clip.

**Analogy:**

**Without motion context:**
```
You: "A person is singing." (prompt + audio)
AI: Generates a video from scratch

You: "A person is singing." (same prompt + new audio)
AI: Generates ANOTHER video from scratch (might look different!)
```

**With motion context:**
```
You: "A person is singing." (prompt + audio)
AI: Generates a video from scratch

You: "Continue the story. Here's what happened in the last 4 seconds [shows frames 8-80].
     Now, a person is singing." (prompt + new audio + motion context)
AI: Generates a video that CONTINUES the previous motion naturally
```

---

## Audio Processing Pipeline

### **Confusion:** Can I input any sample rate audio? Will it auto-convert to 16kHz?

### **Answer:**

**YES!** The `librosa.load()` function handles resampling:

```python
input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
#                                                     └─ Target sample rate (16000)
```

**What happens:**
- If your audio is 44.1kHz → librosa resamples to 16kHz ✅
- If your audio is 22.05kHz → librosa resamples to 16kHz ✅
- If your audio is already 16kHz → no resampling needed ✅

**For training:** Your CSV can point to audio files of any sample rate, as long as your dataset loader uses librosa or similar resampling.

---

## The pre_calculate_audio_pose() Function

### **Confusion:** What does `WanVideoUnit_S2V.pre_calculate_audio_pose()` do?

### **Answer:**

**Purpose:** Process and chunk audio/pose BEFORE the generation loop.

**What it does:**
1. Calculate clips needed: `num_repeat = ceil(audio_duration / clip_duration)`
2. Split audio into clip-aligned chunks (e.g., 10s audio → 2 chunks of ~5.06s)
3. Encode each audio chunk with wav2vec2 → `audio_embeds = [embed_0, embed_1, ...]`
4. Encode pose video chunks with VAE (if provided) → `pose_latents = [...]`

**Returns:** `audio_embeds`, `pose_latents`, `num_repeat`

**Why pre-calculate?**
1. **Efficiency:** Audio encoding is slow, do it once upfront (not in every denoising step)
2. **Chunking:** Complex logic to split audio into clip-aligned segments
3. **Batch processing:** Can encode multiple chunks in parallel

---

## Frame Dropping Logic

### **Confusion:** Why drop frames? Does this reduce video/audio length?

### **Answer:**

**Frame removal affects VIDEO ONLY, not audio!**

**Timeline:**
```
Audio (unchanged):
[0s────────5.06s────────10.0s]
 └─audio_embeds[0]┘└─audio_embeds[1]┘

Video frames (after dropping):
Clip 0: 81 → 80 → 77 frames (dropped 4 total)
Clip 1: 81 → 80 frames (dropped 1)

Final video: 77 + 80 = 157 frames
Duration: 157/16 = 9.8125 seconds

Mismatch: 10.0s audio vs 9.8s video = 0.2s difference
```

**Two types of dropping:**

**Type 1: Duplicate reference frame (ALL clips)**
```python
pipe(...) returns [reference_copy, new_1, ..., new_80]
current_clip[-80:] → [new_1, ..., new_80]
```
Model outputs reference image as frame 0, we drop this duplicate.

**Type 2: Warm-up frames (ONLY clip 0)**
```python
if r == 0:
    current_clip[3:] → [new_4, ..., new_80]
```
First 3 frames have initialization artifacts (stuttery/blurry). Clip 1+ doesn't need this because it has motion context.

---

## Reference Image vs Generated Frames

### **Confusion:** How is the second clip generated? Does it use the same reference image or motion images?

### **Answer:**

**EVERY clip uses the SAME reference image as input!**

```python
# Defined ONCE outside the loop:
input_image = Image.open("reference_face.jpg")

# Used for EVERY clip:
for r in range(num_repeat):
    current_clip = pipe(
        input_image=input_image,  # ← SAME reference image every time!
        audio_embeds=audio_embeds[r],  # ← DIFFERENT audio per clip
        motion_video=motion_videos,    # ← Motion context from previous clip
        ...
    )
```

**What the model sees for each clip:**

#### **Clip 0:**
```
Inputs:
  - input_image: reference_face.jpg
  - audio_embeds[0]: Audio for seconds 0-5.06
  - motion_video: [] (empty, no prior context)
  - prompt: "a person is singing"

Model thinks:
  "Generate 80 new frames starting from this reference face,
   synchronized with this audio, no prior motion."
```

#### **Clip 1:**
```
Inputs:
  - input_image: reference_face.jpg (SAME as clip 0!)
  - audio_embeds[1]: Audio for seconds 5.06-10.0
  - motion_video: [frame_8, frame_9, ..., frame_80] (73 frames from clip 0)
  - prompt: "a person is singing"

Model thinks:
  "Generate 80 new frames starting from this reference face,
   synchronized with this audio, continuing the motion from frames 8-80."
```

**Role separation:**
```
Reference image → "What the person looks like"
Motion context  → "How the person has been moving"
Audio          → "What sound to sync with"
```

**Result:** Each clip looks like the reference person, continues previous motion, syncs with new audio.

---

## Complete Multi-Clip Generation Example

### **Scenario:**
- Audio: 10 seconds (160000 samples at 16kHz)
- Settings: `infer_frames=80`, `num_frames=81`, `fps=16`, `motion_frames=73`
- Result: `num_repeat=2` clips

---

### **Preprocessing:**

```python
audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(...)

# Results:
num_repeat = 2
audio_embeds = [
    embed_clip0,  # Audio seconds 0.0-5.0625 (81000 samples)
    embed_clip1   # Audio seconds 5.0625-10.0 (79000 samples, shorter!)
]
pose_latents = None  # Assuming no pose video
```

---

### **📹 CLIP 0 GENERATION (r=0)**

#### **Initial state:**
```python
motion_videos = []
video = []
```

#### **Step 1: Generate frames**
```python
output_0 = pipe(
    input_image=reference_face,
    audio_embeds=audio_embeds[0],
    motion_video=[],  # ← Empty, no prior context
    num_frames=81,
    ...
)

# Model output:
# [reference_face_copy, new_1, new_2, new_3, new_4, ..., new_80]
#  └─ position 0     └────── positions 1-80 ──────────┘
```

#### **Step 2: Drop reference frame**
```python
current_clip = output_0[-80:]
# [new_1, new_2, new_3, new_4, new_5, ..., new_80]
```

#### **Step 3: Drop warm-up frames (r=0 only)**
```python
if r == 0:
    current_clip = current_clip[3:]

# [new_4, new_5, new_6, ..., new_80]  (77 frames)
```

#### **Step 4: Update motion_videos**
```python
overlap_frames_num = min(73, 77) = 73

motion_videos = motion_videos[73:] + current_clip[-73:]
              = [][73:]            + [new_8, new_9, ..., new_80]
              = []                 + [73 frames]
              = [new_8, new_9, ..., new_80]  (73 frames)
```

**Why starts at new_8?**
```python
current_clip = [new_4, new_5, new_6, new_7, new_8, ..., new_80]  (77 frames)
#               index 0-3 = first 4 frames
#               index 4-76 = last 73 frames
current_clip[-73:] = [new_8, new_9, ..., new_80]
```

#### **Step 5: Add to final video**
```python
video.extend(current_clip)
# video = [new_4, new_5, new_6, ..., new_80]  (77 frames)
```

#### **Step 6: Save progress**
```python
save_video_with_audio(video, "output.mp4", audio_path, fps=16)
# Saved: 77 frames = 4.8125 seconds
```

---

### **📹 CLIP 1 GENERATION (r=1)**

#### **Initial state:**
```python
motion_videos = [new_8, new_9, ..., new_80]  (73 frames)
video = [new_4, new_5, ..., new_80]  (77 frames)
```

#### **Step 1: Generate frames**
```python
output_1 = pipe(
    input_image=reference_face,  # ← SAME reference!
    audio_embeds=audio_embeds[1],
    motion_video=[new_8, new_9, ..., new_80],  # ← Motion context!
    num_frames=81,
    ...
)

# Model output:
# [reference_face_copy, new_81, new_82, new_83, ..., new_160]
#  └─ position 0     └────── positions 1-80 ────────┘
#
# Note: The model saw frames [new_8...new_80] as context,
#       so new_81 smoothly continues the motion!
```

#### **Step 2: Drop reference frame**
```python
current_clip = output_1[-80:]
# [new_81, new_82, new_83, ..., new_160]  (80 frames)
```

#### **Step 3: No warm-up drop (r>0)**
```python
if r == 0:  # False, skip
    current_clip = current_clip[3:]

# [new_81, new_82, new_83, ..., new_160]  (80 frames, unchanged)
```

#### **Step 4: Update motion_videos**
```python
overlap_frames_num = min(73, 80) = 73

motion_videos = motion_videos[73:] + current_clip[-73:]
              = [new_8,...,new_80][73:] + [new_88, new_89, ..., new_160]
              = []                       + [73 frames]
              = [new_88, new_89, ..., new_160]  (73 frames)
```

**Why starts at new_88?**
```python
current_clip = [new_81, new_82, ..., new_87, new_88, ..., new_160]  (80 frames)
#               index 0-6 = first 7 frames
#               index 7-79 = last 73 frames
current_clip[-73:] = [new_88, new_89, ..., new_160]
```

**Why is old motion context discarded?**
```python
[new_8, new_9, ..., new_80]  # 73 frames (indices 0-72)
[new_8, new_9, ..., new_80][73:]  # "From index 73 onwards" = []
```

#### **Step 5: Add to final video**
```python
video.extend(current_clip)
# video = [new_4, ..., new_80, new_81, new_82, ..., new_160]
#         └─── 77 frames ───┘ └────── 80 frames ───────┘
#                           (157 frames total)
```

#### **Step 6: Save progress**
```python
save_video_with_audio(video, "output.mp4", audio_path, fps=16)
# Saved: 157 frames = 9.8125 seconds
```

---

### **Final Output:**

```
Final video frames:
[new_4, new_5, ..., new_80, new_81, new_82, ..., new_160]
 └──────── 77 ─────────┘ └─────── 80 ──────────┘

Total frames: 157
Duration: 157 / 16 fps = 9.8125 seconds

Missing frames:
- reference_face (clip 0 position 0) → dropped
- new_1, new_2, new_3 (warm-up) → dropped
- reference_face (clip 1 position 0) → dropped

Audio: 10.0 seconds
Video: 9.8125 seconds
Mismatch: ~0.2 seconds (acceptable)
```

---

### **Visual Timeline:**

```
Audio timeline:
├─────────5.06s──────┼─────────10.0s─────┤
└─ audio_embeds[0] ──┘└─ audio_embeds[1] ─┘

Generated frames:
Clip 0: [REF, 1, 2, 3, 4, 5, ..., 80] (81 frames)
         ↓   Drop REF
        [1, 2, 3, 4, 5, ..., 80] (80 frames)
         ↓   Drop warm-up (1, 2, 3)
           [4, 5, 6, ..., 80] (77 frames)
                └─ Last 73: [8, 9, ..., 80] → motion_videos

Clip 1: [REF, 81, 82, 83, ..., 160] (81 frames)
         ↓   Drop REF
           [81, 82, 83, ..., 160] (80 frames)
                     └─ Last 73: [88, 89, ..., 160] → motion_videos

Final video:
[4, 5, ..., 80, 81, 82, ..., 160] (157 frames = 9.8s)
```

---

## Common Confusions Clarified

### **Q1: Is clip length fixed at 5 seconds?**

**A:** NO, it's configurable via `infer_frames` hyperparameter.

```python
infer_frames = 80  → clip_duration = 81/16 = 5.06s
infer_frames = 160 → clip_duration = 161/16 = 10.06s
infer_frames = 40  → clip_duration = 41/16 = 2.56s
```

Constraint: `(infer_frames + 1) % 4 == 1`

---

### **Q2: Why is motion_videos always 73 frames?**

**A:** It's a sliding window buffer that keeps ONLY the most recent 73 frames.

```python
motion_videos = motion_videos[73:] + current_clip[-73:]
#               └─ Discard old  └─ Keep new 73
#                  (always empty)
```

Each iteration: old 73 frames discarded, new 73 frames stored.

---

### **Q3: What is overlap_frames_num?**

**A:** Safety check to prevent indexing errors.

```python
overlap_frames_num = min(73, len(current_clip))
```

Normally it's 73, but protects against edge cases (e.g., last clip shorter than expected).

---

### **Q4: Why does every clip use the same reference image?**

**A:** The reference image defines "what the person looks like". The motion_video provides "how they've been moving".

```python
for r in range(num_repeat):
    pipe(
        input_image=reference_face,  # ← Identity
        motion_video=motion_videos,  # ← Motion continuity
        audio_embeds=audio_embeds[r] # ← New audio
    )
```

---

### **Q5: What gets returned by pipe()?**

**A:** `[reference_image_copy, new_frame_1, new_frame_2, ..., new_frame_80]`

The model always outputs the reference image as frame 0 (convenience), which we drop to avoid duplicates.

---

### **Q6: Why drop the first 3 frames of clip 0?**

**A:** Initialization artifacts (stuttery/blurry motion). The model needs 3 frames to transition from static image to smooth motion.

Only clip 0 needs this because subsequent clips have motion context.

---

### **Q7: Is there a gap between clips in the final video?**

**A:** Technically yes (the duplicate reference frame is missing), but:
- Motion context ensures smooth transitions
- The duplicate would look out of place
- Humans don't perceive the gap

---

### **Q8: Why is audio_processor_config separate from model_configs?**

**A:** Different purposes:

```python
model_configs = [
    ...,
    ModelConfig(..., "model.safetensors"),  # wav2vec2 model WEIGHTS
    ...
]

audio_processor_config = ModelConfig(
    ..., "wav2vec2-large-xlsr-53-english/"  # wav2vec2 CONFIG directory
)
```

- **model_configs**: Load neural network weights (GPU memory)
- **audio_processor_config**: Load preprocessing configs (CPU memory)

The processor prepares raw audio into the format the model expects (normalization, padding, etc.).

---

## Key Takeaways for Training

1. **Required inputs:** audio (16kHz) + reference image + prompt
2. **Frame constraint:** `num_frames % 4 == 1`
3. **Motion context:** Enables smooth multi-clip generation
4. **Audio preprocessing:** Done BEFORE generation loop (via pre_calculate_audio_pose)
5. **Reference image:** Same for all clips, motion context provides continuity
6. **Frame dropping:** Removes duplicates and warm-up artifacts

**For training, you need:**
- CSV with columns: `video`, `audio`, `prompt`, `input_image`, `[s2v_pose_video]`
- Audio files (any sample rate, will be resampled to 16kHz)
- Video clips with `num_frames % 4 == 1`
- Training script will use `WanVideoUnit_S2V` to process audio automatically

---

**End of Notes**
