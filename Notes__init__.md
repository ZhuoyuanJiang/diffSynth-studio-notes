# Learning Notes for diffsynth/utils/__init__.py

This document contains detailed explanations for key functions in the foundation classes.

---

## Q1: Does the pipeline automatically handle mismatched dimensions internally so that I don't need to manually resize my images?

**Short Answer:**
- **For Inference:** ✅ YES - The pipeline auto-resizes your input images internally
- **For Training:** ❌ NO - You must manually resize images during data preprocessing

**Explanation:**

There are TWO layers of resizing happening during inference:

**Layer 1: Dimension Adjustment**
1. `check_resize_height_width()` adjusts target dimensions (e.g., 450→464, 830→832)
   - This ensures dimensions are divisible by required factors (16 for height/width)
   - Only changes the dimension VALUES (variables), not actual images

**Layer 2: Automatic Image Resizing (During Inference)**

Both the inference script AND pipeline units resize images:

1. **Example Inference Script** (line 94 in `Wan2.2-S2V-14B_multi_clips.py`):
   ```python
   input_image = Image.open("...").convert("RGB").resize((width, height))
   #                                                ↑ Manual resize in script
   ```

2. **Pipeline Units** (line 708 in `wan_video_new.py`):
   ```python
   # Inside WanVideoUnit_ImageEmbedderVAE.process():
   image = pipe.preprocess_image(input_image.resize((width, height)))
   #                              ↑ Automatic resize happens here!
   ```

**Result:** Even if you pass a mismatched image to `pipe()`, the `ImageEmbedder` units will auto-resize it to match the adjusted dimensions.

**During Training:**
- Data loads directly from CSV, bypassing `ImageEmbedder` units
- No automatic resizing happens
- You must preprocess videos to correct dimensions before training

**Key Takeaway:** For inference, you can pass any size image - the pipeline handles dimension validation AND automatic resizing internally. For training, you must manually preprocess your data to the correct dimensions.

---

## Q2: Understanding `preprocess_video()` and `torch.stack()`

### **What does `preprocess_video()` do?**

Converts a list of PIL.Image frames into a single video tensor with shape `(B, C, T, H, W)`.

### **Step-by-Step Breakdown**

#### **Step 1: Input**
```python
# Input: List of PIL.Image objects (video frames)
video = [
    PIL.Image(...),  # Frame 0: 832x448 RGB image
    PIL.Image(...),  # Frame 1: 832x448 RGB image
    PIL.Image(...),  # Frame 2: 832x448 RGB image
    # ... 81 frames total
]
```

#### **Step 2: Preprocess Each Frame**
```python
video = [self.preprocess_image(image, ...) for image in video]
```

**What happens:**
- Each `PIL.Image` → tensor of shape `(1, 3, H, W)`
  - `1` = batch dimension
  - `3` = RGB channels
  - `H` = height (448)
  - `W` = width (832)

**Result:**
```python
video = [
    tensor(1, 3, 448, 832),  # Frame 0
    tensor(1, 3, 448, 832),  # Frame 1
    tensor(1, 3, 448, 832),  # Frame 2
    # ... 81 frames total (LIST of tensors)
]
```

#### **Step 3: Stack Into Single Tensor** ⭐
```python
video = torch.stack(video, dim=pattern.index("T") // 2)
```

### **Understanding `torch.stack()`**

`torch.stack()` takes a **list of tensors** and combines them along a **new dimension**.

**Simple Example:**
```python
# List of 3 tensors, each shape (2, 3)
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])
c = torch.tensor([[13, 14, 15], [16, 17, 18]])

tensors = [a, b, c]

# Stack along dim=0 (add new dimension at front)
result = torch.stack(tensors, dim=0)
# Result shape: (3, 2, 3)
#               ↑ new dimension with size 3

# Stack along dim=1 (add new dimension at position 1)
result = torch.stack(tensors, dim=1)
# Result shape: (2, 3, 3)
#                  ↑ new dimension inserted here
```

### **Applying to Video**

#### **Calculating `dim`:**
```python
pattern = "B C T H W"  # Target pattern
pattern.index("T")  # Find position of "T" in string
# "B C T H W"
#  0 2 4 6 8  ← character positions
# "T" is at position 4

pattern.index("T") // 2  # Integer division
# 4 // 2 = 2
```

**So `dim = 2`**

#### **Stacking along `dim=2`:**
```python
# Before: List of 81 tensors, each shape (1, 3, H, W)
#                                        ↓  ↓  ↓  ↓
#                                      dim: 0  1  2  3

# Stack along dim=2:
video = torch.stack(video, dim=2)

# After: Single tensor of shape (1, 3, 81, H, W)
#                                ↓  ↓  ↓↓  ↓  ↓
#                              dim: 0  1  2   3  4
#                                         ↑
#                                    New dimension inserted here!
```

### **Complete Example with Numbers**

```python
# === INPUT ===
video = [PIL.Image(832x448), PIL.Image(832x448), PIL.Image(832x448)]  # 3 frames

# === STEP 1: Preprocess each frame ===
video = [
    tensor(1, 3, 448, 832),  # Frame 0
    tensor(1, 3, 448, 832),  # Frame 1
    tensor(1, 3, 448, 832),  # Frame 2
]
# List of 3 tensors

# === STEP 2: Stack along dim=2 ===
pattern = "B C T H W"
dim = pattern.index("T") // 2  # 4 // 2 = 2

video = torch.stack(video, dim=2)

# Result: (1, 3, 3, 448, 832)
#          ↑  ↑  ↑   ↑    ↑
#          B  C  T   H    W
#
# Where:
# - B = 1 (batch size)
# - C = 3 (RGB channels)
# - T = 3 (number of frames)
# - H = 448 (height)
# - W = 832 (width)
```

### **Why This Dimension Order: `(B, C, T, H, W)`?**

This is the **standard video tensor format** for PyTorch:

- **B** (Batch): Standard PyTorch convention (batch first)
- **C** (Channels): Matches image format (B, C, H, W)
- **T** (Time): Extends image format to video naturally
- **H, W** (Height, Width): Spatial dimensions last (standard for convolutions)

### **Why `pattern.index("T") // 2` Instead of Hardcoding `2`?**

**Flexibility!** The code adapts to different patterns:

```python
# Wan Video: "B C T H W"
pattern.index("T") // 2  # → 4 // 2 = 2

# Other models might use: "B T C H W"
pattern.index("T") // 2  # → 2 // 2 = 1

# Code automatically adapts!
```

### **Visual Summary**

```
Input:  [PIL, PIL, PIL, ..., PIL]  ← 81 PIL.Image frames
           ↓
Step 1: [tensor(1,3,H,W), tensor(1,3,H,W), ...]  ← List of 81 tensors
           ↓
Step 2: torch.stack(dim=2)
           ↓
Output: tensor(1, 3, 81, H, W)  ← Single video tensor
        (B, C, T,  H, W)
```

**Final shape for 81-frame video at 832x448:** `(1, 3, 81, 448, 832)`

---

## Q3: Understanding `blend_with_mask()` and Inpainting

### **What is `blend_with_mask()` used for?**

This method is for **inpainting** - when you want to modify only PART of an image/video while keeping other parts unchanged.

### **Example Scenario**

```
Imagine you have a video of a person walking.
You want to change ONLY the background, but keep the person unchanged.

- mask = 1 (white) → areas you want to REGENERATE (background)
- mask = 0 (black) → areas you want to KEEP ORIGINAL (the person)
```

### **The Formula**

```python
def blend_with_mask(self, base, addition, mask):
    return base * (1 - mask) + addition * mask
```

**Parameters:**
- `base`: The original/expected content
- `addition`: The newly generated content
- `mask`: Controls which one to use (values 0 to 1)

### **How It Works**

**Examples:**
- If `mask = 0` (keep original):
  ```python
  result = base * 1 + addition * 0 = base  ✓
  ```
- If `mask = 1` (use new):
  ```python
  result = base * 0 + addition * 1 = addition  ✓
  ```
- If `mask = 0.5` (blend 50/50):
  ```python
  result = base * 0.5 + addition * 0.5  (smooth transition)
  ```

### **Why Blend During Denoising?**

During the diffusion process, we denoise step-by-step. For inpainting:
- **Masked regions (mask=1)**: Use the model's predicted noise (generate new content)
- **Unmasked regions (mask=0)**: Use expected noise from original (preserve original)

### **Visual Example: Inpainting with Mask**

```
Original: [Person in room]
Mask: Keep person (0), change background (1)

Step 0:  Pure Noise [t=999]
         ↓ blend_with_mask
         Person region: use original's noise (preserved)
         Background: use model's generated noise (new content)

Step 25: [t=500]
         Person: still from original (preserved)
         Background: new scene emerging

Step 49: [t=0]
         Result: Original person + New background ✓
```

---

## Q4: Understanding Timesteps and Schedulers

### **What is a Scheduler?**

A **scheduler** controls the denoising process in diffusion models. Think of it as a "recipe" for how to remove noise step-by-step.

### **The Diffusion Process**

```
FORWARD (training):
Clean Image → +noise → +noise → +noise → Pure Noise
              t=0      t=250    t=500    t=1000

REVERSE (generation/inference):
Pure Noise → -noise → -noise → -noise → Clean Image
t=1000      t=500    t=250    t=0
```

### **What is a Timestep?**

**Timestep** = a value that indicates "how noisy" the image is at the current step.

- **t=1000** (or t=999): Maximum noise (pure random noise)
- **t=500**: Medium noise (half denoised)
- **t=0**: No noise (clean image)

### **Understanding `scheduler.timesteps[progress_id]`**

```python
timestep = scheduler.timesteps[progress_id]
# "Get the timestep value for this progress step from scheduler's timestep schedule"
```

**Breaking it down:**

#### **1. `scheduler.timesteps`** - A pre-computed list of timestep values

Example for 50 denoising steps:
```python
scheduler.timesteps = [999, 979, 959, 939, ..., 39, 19, 0]
#                      ↑ start (most noisy)        ↑ end (clean)
```

#### **2. `progress_id`** - Which step you're currently at (0 to 49 for 50 steps)

```python
progress_id = 0  → timestep = 999  (first step, very noisy)
progress_id = 25 → timestep = 499  (halfway, medium noise)
progress_id = 49 → timestep = 0    (last step, clean)
```

#### **3. Why does the scheduler need timestep?**

The scheduler uses `timestep` to:
- Know how much noise to remove in this step
- Calculate the denoising strength (remove more noise at high t, less at low t)
- Follow the noise schedule (linear, cosine, etc.)

### **How `step()` Works with Timesteps**

```python
def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
    # 1. Get current noise level
    timestep = scheduler.timesteps[progress_id]  # e.g., t=500

    # 2. Handle inpainting (if mask provided)
    if inpaint_mask is not None:
        # Calculate what noise SHOULD be at t=500 for original image
        noise_pred_expected = scheduler.return_to_timestep(
            scheduler.timesteps[progress_id], latents, input_latents
        )

        # Blend: masked areas use model's prediction, unmasked use original
        noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)

    # 3. Denoise: latents(t=500) - noise → latents(t=400)
    latents_next = scheduler.step(noise_pred, timestep, latents)

    return latents_next
```

### **The Denoising Process - Step by Step**

**Normal Generation (no mask):**
```
Step 0:  Pure Noise [t=999]
Step 1:  Remove noise → Blurry shapes [t=800]
Step 25: Remove noise → Recognizable image [t=500]
Step 49: Remove noise → Sharp image [t=0]
```

### **Key Relationships**

```
progress_id → timestep → scheduler.step() → less noisy latents
(which step)  (noise level) (how to denoise)  (result)
```

**Summary:**
- **Timestep**: A number indicating noise level (1000=noisy, 0=clean)
- **Scheduler**: Stores the sequence of timesteps and knows how to remove noise at each step
- **`progress_id`**: Which denoising step you're on (0, 1, 2, ..., 49)
- **Different schedulers**: Different denoising strategies (DDPM, DDIM, Euler, etc.)

---

## Q5: Understanding Distributed Training and Rank 0

### **What is Distributed Training?**

**Distributed training** = Training a model using **multiple GPUs** simultaneously to speed up training.

**Example:**
```
Single GPU:     [GPU 0] ← processes entire batch (slow)

Distributed:    [GPU 0] [GPU 1] [GPU 2] [GPU 3]
                   ↓        ↓        ↓        ↓
                Each processes part of the batch (4x faster!)
```

### **What is Rank?**

**Rank** = The ID number assigned to each GPU process in distributed training.

```
GPU 0 → Rank 0  (the "leader" or "main" process)
GPU 1 → Rank 1
GPU 2 → Rank 2
GPU 3 → Rank 3
```

### **Why Only Rank 0 Downloads?**

**Problem:** If all 4 GPUs try to download the same model file simultaneously:
- Wastes bandwidth (downloading same file 4 times)
- Risk of file corruption (multiple processes writing to same file)
- Slower overall (network bottleneck)

**Solution:**
```python
# STEP 2: Handle distributed training (only rank 0 downloads)
if use_usp:  # If using distributed training
    import torch.distributed as dist
    skip_download = self.skip_download or dist.get_rank() != 0
    # Only rank 0 downloads (rank 0 → False, others → True)
else:
    skip_download = self.skip_download
```

### **How it Works:**

1. **Rank 0 downloads the model:**
   ```
   Rank 0: Downloads model to ./models/Wan2.2-S2V-14B/
   Rank 1: Waits...
   Rank 2: Waits...
   Rank 3: Waits...
   ```

2. **All ranks wait at barrier:**
   ```python
   # STEP 5: Wait for rank 0 to finish downloading
   if use_usp:
       import torch.distributed as dist
       dist.barrier(device_ids=[dist.get_rank()])
       # All GPUs synchronize here - wait until rank 0 finishes
   ```

3. **All ranks load the downloaded model:**
   ```
   Rank 0: Loads from ./models/Wan2.2-S2V-14B/ ✓
   Rank 1: Loads from ./models/Wan2.2-S2V-14B/ ✓
   Rank 2: Loads from ./models/Wan2.2-S2V-14B/ ✓
   Rank 3: Loads from ./models/Wan2.2-S2V-14B/ ✓
   ```

### **Visual Timeline**

```
Time →
Rank 0: [Download model...............] → [Load] → [Train]
Rank 1: [Wait at barrier...............] → [Load] → [Train]
Rank 2: [Wait at barrier...............] → [Load] → [Train]
Rank 3: [Wait at barrier...............] → [Load] → [Train]
                                         ↑
                                    dist.barrier()
                                    (synchronization point)
```

### **Key Takeaways**

- **Rank 0** = Main GPU process (the "leader")
- Only Rank 0 downloads to avoid waste/corruption
- `dist.barrier()` synchronizes all ranks (everyone waits for rank 0 to finish)
- After synchronization, all ranks can load the same downloaded model

---

## Q6: Understanding CFG (Classifier-Free Guidance)

### **What is CFG?**

**CFG (Classifier-Free Guidance)** is a technique to make diffusion models follow prompts more accurately.

### **The Idea**

Generate TWO predictions:
1. **With prompt** (positive): "A cat in a garden"
2. **Without prompt** (negative): "" (empty or "low quality")

Then combine them to **amplify the difference** (make the prompt stronger).

### **The CFG Formula**

```python
final_output = negative_output + cfg_scale * (positive_output - negative_output)
```

**Example with `cfg_scale = 7.5`:**
```python
final = negative + 7.5 * (positive - negative)
      = negative + 7.5 * positive - 7.5 * negative
      = 7.5 * positive - 6.5 * negative
```

### **What This Means**

- Higher `cfg_scale` → stronger adherence to prompt (more creative/stylized)
- `cfg_scale = 1` → no guidance (just use positive, ignore negative)
- `cfg_scale = 0` → just use negative (ignore prompt)
- Common values: 7.5 for images, 5-7 for videos

### **Why CFG Works**

By amplifying the difference between "with prompt" and "without prompt", the model generates results that more strongly exhibit the characteristics described in the prompt.

**Visual Example:**
```
Without CFG (cfg_scale=1):
  Result: A cat (generic, might not match "in a garden")

With CFG (cfg_scale=7.5):
  Result: A cat clearly in a garden (stronger prompt adherence)
```

---

## Q7: Understanding PipelineUnit

### **What is PipelineUnit?**

A **PipelineUnit** is a modular processing step in the pipeline. Think of it like a **Lego block** that processes inputs and outputs results.

The pipeline chains units together:
```
Input → [Unit 1] → [Unit 2] → [Unit 3] → Output
```

### **The Three Input Dictionaries**

The pipeline maintains **three dictionaries** of inputs:

1. **`inputs_shared`**: CFG-insensitive inputs (same for both positive/negative)
   - Example: `height=832, width=448, num_frames=81, seed=42`

2. **`inputs_posi`**: Positive conditioning (with prompt)
   - Example: `prompt_embeds` for "A beautiful sunset over ocean"

3. **`inputs_nega`**: Negative conditioning (without prompt or negative prompt)
   - Example: `prompt_embeds` for "" or "low quality, blurry"
