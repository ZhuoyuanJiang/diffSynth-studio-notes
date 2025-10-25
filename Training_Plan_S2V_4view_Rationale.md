# Training Plan Rationale: Decision-Making Process for S2V Multi-View Training

**Date**: 2025-10-22
**Last Updated**: 2025-10-22 (Changed to single→multi after mentor confirmation)

This document captures the questions, options, and reasoning that led to our final training plan documented in `Training_Plan_S2V_4view.md`.

---

## FINAL DECISION (After Mentor Confirmation)

**Confirmed Goal**: Single-view image → Multi-view video (2×2 grid)
- **Input**: Single frontal face image (cam_54) + audio
- **Output**: Animated 2×2 grid video (4 camera angles)
- **Approach**: Approach A (single→multi, novel view synthesis + animation)
- **Input camera**: Always cam_54 (frontal hemisphere)
- **Training samples**: ~1,890 (one per clip, cam_54 only)
- **Resolution**: 448×832 (224×416 per camera) - **Original S2V resolution** ✅
- **Timeline**: Start training by end of tomorrow (2025-10-23)

This is the **most ambitious approach** but aligns with the end-user vision: users upload one photo + audio, get multi-view talking video.

**Key Decision - Use Original S2V Resolution (448×832)**:
- ✅ **Non-invasive fine-tuning**: Matches pre-trained model exactly
- ✅ **Lower risk**: Position encodings, spatial features match
- ✅ **Faster training**: 56% fewer pixels than 560×1040 alternative
- ✅ **Better for deadline**: Minimizes variables

**Alternatives documented for future**:
- Higher resolution (560×1040): +56% pixels, better quality but more invasive
- Use all 4 cameras as input (data augmentation): ~7,560 samples, more robust but 4× slower training

---

## Overview: The Decision Flow

```
Q1 (Goal) → Determines data format (grids vs separate views)
    ↓
Q2 (Resolution) → Blocked on Q3 (need GPU info to recommend)
    ↓
Q3 (GPUs) → Unlocks Q2 (now can calculate what fits)
    ↓
Q4 (Pose) → Determines preprocessing complexity
    ↓
Q5 (Input/Output) → CRITICAL: single→multi or multi→multi?
    ↓
Q6 (Input camera) → Conditional on Q5 answer
    ↓
Q7 (Approach) → Risk management: incremental vs all-in
    ↓
Q8 (Final resolution) → Quality vs speed trade-off
    ↓
Implementation Choice → On-the-fly vs preprocessing
```

---

## Fundamental Principle: Training Input MUST Match Inference Input

**CRITICAL CONCEPT**: The input distribution during training must match the input distribution during inference. You cannot train a model with one type of input and expect it to work with a different type of input.

### Why This Matters:

**Example of MISMATCH (will not work)** ❌:
- **Training**: Input = 2×2 grid image → Output = 2×2 grid video
- **Inference**: Input = Single-view image → Output = ???
- **Problem**: Model has NEVER seen single-view input during training, doesn't know how to handle it!

**Example of MATCH (will work)** ✅:
- **Training**: Input = 2×2 grid image → Output = 2×2 grid video
- **Inference**: Input = 2×2 grid image → Output = 2×2 grid video
- **Success**: Model knows exactly how to handle this input!

### Implications for Our Planning:

If we want users to upload **single-view images** at inference, we have TWO options:

**Option A: Train with single-view input directly**
- Training: Single-view image + audio → 2×2 grid video (novel view synthesis + animation)
- Inference: Single-view image + audio → 2×2 grid video ✅ Matches!
- Challenge: Very hard to train (two difficult problems at once)

**Option B: Two-stage pipeline**
- **Stage 1**: Single-view image → Generate 2×2 grid (using separate view synthesis model like Zero-1-to-3, SyncDreamer)
- **Stage 2**: 2×2 grid + audio → Animated 2×2 grid video (our trained model)
- Inference works because our model always receives 2×2 grid input ✅
- Advantage: Separates concerns, each model does one job

**Our Choice (Phase 1)**: Train single→multi (Approach A, Option A above)
- Training: **Single-view image (cam_54) + audio → 2×2 grid video**
- Inference: **Single-view image + audio → 2×2 grid video** ✅ Matches!
- Directly achieves end-user vision (upload one photo, get multi-view video)
- We keep **multi→multi (Option B)** documented as **fallback** if Approach A quality is insufficient

---

## Fundamental Principle: Training Resolution MUST Match Inference Resolution

**CRITICAL CONCEPT**: The resolution you train at should be the EXACT same resolution you use during inference. Changing resolution between training and inference can cause quality degradation or artifacts.

### Why Resolution Matching Matters:

**Example of EXACT MATCH (ideal)** ✅:
- **Training**: Model trained at 560×1040 resolution
- **Inference**: Generate videos at 560×1040 resolution
- **Result**: Perfect quality, model performs as expected

**Example of MISMATCH (problematic)** ⚠️:
- **Training**: Model trained at 560×1040 resolution
- **Inference**: Generate videos at 640×1280 resolution
- **Problem**: Model has never seen this resolution, position encodings may break, attention patterns distorted
- **Result**: Quality degradation, possible visual artifacts, unpredictable behavior

### Technical Reasons:

1. **Position Encodings**:
   - Diffusion models use position encodings that are resolution-specific
   - Changing resolution means positions the model has never seen

2. **Learned Spatial Features**:
   - Model learns features at specific spatial scales
   - Different resolution = different scales = model confusion

3. **Attention Patterns**:
   - Self-attention in transformers (DiT) learns patterns at trained resolution
   - New resolution disrupts these learned patterns

### Can Any Resolution Change Be Tolerated?

**Small changes might work, but NOT recommended**:
- Train 560×1040 → Infer 480×832: Often OK (downscaling ~15%)
- Train 560×1040 → Infer 640×1280: Likely issues (upscaling ~20%)
- Train 560×1040 → Infer 800×1600: Significant problems (upscaling ~40%)

**Why some flexibility exists**:
- VAE can encode/decode different resolutions (it's resolution-agnostic)
- Some newer architectures use relative position encodings (more flexible)
- BUT the DiT (diffusion transformer) still trained at specific resolution

**General rule**: Stay within ±10% of training resolution for acceptable results, but exact match is always best.

### What If Users Want Different Output Resolution?

**The correct approach**:
1. ✅ Inference at your training resolution (e.g., 448×832 for Phase 1)
2. ✅ Post-process: Upscale or downscale the output video
3. ❌ Do NOT change inference resolution directly

**Example workflow**:
```
User wants 1080p output
→ Infer at 448×832 (training resolution for Phase 1)
→ Upscale output video to 1920×1080 using video upscaler
→ Deliver 1080p result
```

### Our Plan (Phase 1):

**We will train and infer at IDENTICAL resolution: 448×832** (original S2V resolution)
- Training data: 448×832 grid videos (224×416 per camera)
- Inference input: 448×832 grid images
- Inference output: 448×832 grid videos
- ✅ Perfect match throughout!
- **Phase 2 (optional)**: Raise to 560×1040 after 448×832 approach is validated

If we later want to support different output resolutions, we'll use post-processing upscaling (e.g., ESRGAN, Real-ESRGAN) rather than changing inference resolution.

---

## Q1: What is your primary goal for using all 4 cameras?

### Options Presented:

**Option 1: Generate multi-view outputs (2×2 grid videos)** ✅ **CHOSEN**
- Train model to output 4-camera views simultaneously in a 2×2 grid format
- **Why this option**: Novel capability, enables 3D-aware talking head generation
- **Implication**: Need to train on grid images/videos, model learns spatial relationships between views
- **Use case**: User wants final output to be a surveillance-style multi-camera video

**Option 2: Improve single-view quality via multi-view learning**
- Use 4 cameras as training data diversity, but output standard single-view videos
- **Why this option**: Simpler approach, just 4× more training data, each camera is separate sample
- **Implication**: Would train 4 separate samples per timestamp, output is standard single-view like original S2V
- **Use case**: User wants to improve quality of single-view generation using diverse camera angles

**Option 3: Both - flexible multi-view generation**
- Support both single-view and multi-view (2×2 grid) generation
- **Why this option**: Most flexible but complex
- **Implication**: Would need to train with mixed data (sometimes 1 view, sometimes 4 views), model learns to handle both
- **Use case**: User wants maximum flexibility but willing to accept training complexity

### Rationale for Asking:
I needed to understand **what the model should learn**:
- **Spatial relationships** between cameras? (Option 1)
- OR just **more data diversity** with independent views? (Option 2)
- This fundamentally determines:
  - Data preparation strategy (create grids vs separate samples)
  - Model architecture considerations (spatial attention across views vs single-view)
  - Final inference behavior

### Detailed Explanation of Option 2:

To clarify what "Option 2: Improve single-view quality via multi-view learning" means:

**Training Setup**:
- Treat each camera as a SEPARATE training sample
- From one timestamp with 4 cameras, create 4 independent samples:
  - Sample 1: cam_28 video (single-view) + audio → train model
  - Sample 2: cam_37 video (single-view) + audio → train model
  - Sample 3: cam_49 video (single-view) + audio → train model
  - Sample 4: cam_54 video (single-view) + audio → train model
- Total training samples: ~1,890 clips × 4 cameras = **~7,560 samples**

**Inference**:
- Input: Single-view image + audio
- Output: Single-view video (standard S2V behavior)
- Model doesn't learn multi-view relationships, just benefits from diverse training data

**Why "simpler"**:
- No grid creation needed
- Use original S2V pipeline unchanged
- Just 4× more training samples from different camera angles
- Hope: Seeing faces from diverse angles during training improves overall quality

**Comparison**:

| Aspect | Option 2 (not chosen) | Option 1 (chosen) |
|--------|---------------------|------------------|
| Training input | Single-view image | 2×2 grid image |
| Training output | Single-view video | 2×2 grid video |
| Inference input | Single-view image | 2×2 grid image |
| Inference output | Single-view video | 2×2 grid video |
| Training samples | ~7,560 (1,890 × 4) | ~1,890 |
| Model learns | Better single-view from diverse data | Multi-view spatial consistency |
| Grid creation | Never needed | Core functionality |

### Decision Impact:
✅ **Chose Option 1** → We will create 2×2 grid images/videos for training

---

## Q2: What resolution per camera view would you prefer?

### Options Presented:

**Option 1: 224×416 per camera → 448×832 total**
- Standard S2V resolution, safest choice
- **Why**: Matches original training (448×832), guaranteed to work, lowest VRAM
- **Trade-off**: Each camera is only 224×416 - might be too small for face details in a 2×2 grid
- **VRAM estimate**: ~24-32GB per GPU with gradient checkpointing

**Option 2: 240×416 per camera → 480×832 total**
- Slightly higher, still conservative
- **Why**: 7% more pixels, barely more VRAM, still within S2V tested range
- **Trade-off**: Marginal improvement, playing it very safe
- **VRAM estimate**: ~26-34GB per GPU with gradient checkpointing

**Option 3: 280×520 per camera → 560×1040 total**
- 56% more pixels than standard S2V
- **Why**: Noticeable quality improvement, well within modern GPU capacity
- **Trade-off**: Higher VRAM usage, but still safe with sufficient GPUs
- **VRAM estimate**: ~32-40GB per GPU with gradient checkpointing
- **Quality benefit**: 280×520 per camera provides much better face detail
- **Note**: This option was considered but NOT chosen for Phase 1 (documented for future use)

**Option 4: 320×640 per camera → 640×1280 total**
- 120% more pixels than standard S2V
- **Why**: Best quality, pushing towards 1080p territory
- **Trade-off**: Slower training, higher risk of VRAM issues
- **VRAM estimate**: ~38-48GB per GPU with gradient checkpointing

### Rationale for Asking:
Without knowing GPU capacity, I couldn't recommend resolution safely:
- Too high → OOM errors, wasted debugging time
- Too low → poor quality, wasted GPU potential
- Resolution scales **quadratically** with VRAM: 2× resolution = 4× memory
- Blocked on Q3 to make informed recommendation

### Standard S2V Resolution Baseline:

The original **Wan2.2-S2V-14B** was trained at:
- **Resolution**: 448 (height) × 832 (width)
- **Total pixels**: 448 × 832 = **372,736 pixels**

Our options compared to this baseline:

| Option | Resolution | Total Pixels | vs Standard S2V |
|--------|------------|--------------|-----------------|
| Standard S2V | 448×832 | 372,736 | 100% (baseline) |
| **Our Option 1** ✅ | 448×832 | 372,736 | 100% (same) |
| **Our Option 2** | 480×832 | 399,360 | 107% (+7%) |
| **Our Option 3** 📝 | 560×1040 | 582,400 | **156% (+56%)** |
| **Our Option 4** | 640×1280 | 819,200 | 220% (+120%) |

**What "56% more pixels" means**:
- Standard S2V: 372,736 pixels
- Our grid: 560×1040 = 582,400 pixels
- Ratio: 582,400 / 372,736 = 1.56 = **56% increase**

This means significantly better quality for face details while staying within safe VRAM limits.

### Decision Impact:
✅ **Chose Option 1** (448×832 original S2V resolution) for Phase 1 → Non-invasive fine-tuning, minimizes variables
📝 **Option 3** (560×1040) documented for Phase 2 after validating 448×832 approach

---

## Q3: How much GPU memory do you have available for training?

### Options Presented:

**Option 1: Single GPU (40-80GB)**
- **Implication**: Conservative approach needed, recommend Resolution Option 1 or 2
- **Why**: Limited VRAM requires lower resolution or aggressive optimization
- **Batch size**: Likely 1 per GPU
- **Training time**: Slower (no distributed training)

**Option 2: 2-4 GPUs (distributed)** ✅ **PARTIALLY MATCHED (you have 4 minimum)**
- **Implication**: Can handle Resolution Option 2 or 3 comfortably with gradient checkpointing
- **Why**: Distributed training enables higher resolution and faster iteration
- **Batch size**: 1-2 per GPU
- **Training time**: Moderate (distributed across multiple GPUs)

**Option 3: 8+ GPUs (distributed)** ✅ **PARTIALLY MATCHED (you sometimes have 8)**
- **Implication**: Can push for Resolution Option 3 or 4, or use larger batch sizes
- **Why**: Maximum VRAM pool enables highest quality and fastest training
- **Batch size**: 2-4 per GPU
- **Training time**: Fast (maximum parallelization)

### Actual Answer:
**4-8 × RTX 6000 Ada (48GB each)**
- Total VRAM: 192-384GB
- Per-GPU: 48GB
- **Conclusion**: Well-equipped for Resolution Option 3, could even try Option 4

### Rationale for Asking:
Direct calculation needed:
- 14B parameter model = ~28GB just for weights (bf16)
- Video training with 81 frames = large activation memory
- Resolution scales quadratically: 560×1040 = 582k pixels vs 448×832 = 373k pixels
- **Math for Option 3**:
  - Model weights: ~28GB
  - Activations (560×1040×81 frames): ~15-20GB
  - Gradients + optimizer states: ~10-15GB
  - **Total per GPU**: ~32-40GB → **Fits comfortably in 48GB!**

### Decision Impact:
✅ **4-8 × 48GB GPUs** → Unlocked Resolution Option 3 (560×1040) as safe choice

---

## Q4: Do you want to use pose guidance (s2v_pose_video) in training?

### Options Presented:

**Option 1: Skip pose guidance initially** ✅ **CHOSEN**
- Train with audio + image only, no pose conditioning
- **Why**: Simplify first iteration, one less thing to debug
- **Trade-off**: Less control over head motion, might get unrealistic movements
- **When to use**: First experiments, validate core approach works
- **Preprocessing needed**: None (just audio + images)

**Option 2: Use pose guidance from keypoints3d**
- Extract pose videos from the 3D keypoints data in RenderMe360
- **Why**: More accurate control, better lip-sync and head motion realism
- **Trade-off**: Need preprocessing to render pose videos from keypoints3d
- **When to use**: After validating basic approach, want higher quality
- **Preprocessing needed**: Render keypoints3d → pose skeleton videos

### Rationale for Asking:
Looking at S2V model architecture:
```python
# From inference code
s2v_pose_video=None  # This parameter is OPTIONAL!
```
- Original S2V supports pose conditioning but doesn't require it
- Your RenderMe360 dataset HAS keypoints3d data, so it's technically available
- **Trade-off analysis**:
  - **With pose**: Better control, more realistic motion, BUT more complexity
  - **Without pose**: Simpler, faster iteration, BUT less motion control

I wanted to know if you prioritize:
- **Speed of iteration** (skip pose)
- OR **Quality of output** (use pose)

### Decision Impact:
✅ **Skip pose initially** → Simpler data pipeline, faster first results, can add later

---

## Q5: What should be the INPUT IMAGE during inference?

### Options Presented:

⚠️ **REMINDER**: Training input must match inference input (see "Fundamental Principle" section above). The options below reflect this constraint - whatever input we choose for inference, we must train with that same input type.

**Option 1: Single-view image → Multi-view video (2×2 grid)**
- Input: 1 camera view (e.g., frontal face)
- Output: 4-camera 2×2 grid video showing multiple angles
- **Why this could be amazing**: Users just upload 1 photo, get full multi-angle talking video
- **Challenge**: Model must learn **novel view synthesis** + animation simultaneously
- **Risk**: Very hard problem - two research challenges at once
  - Novel view synthesis is an entire field (Zero-1-to-3, SyncDreamer, etc.)
  - Combining it with temporal animation is cutting-edge research
- **What it requires**: Model learns 3D structure of faces, can hallucinate unseen angles with temporal consistency

**Option 2: Multi-view image (2×2 grid) → Animated multi-view video**
- Input: 2×2 grid of 4 views (all 4 camera angles)
- Output: Animated 2×2 grid video
- **Why this is safer**: Model only learns animation with spatial consistency, not view synthesis
- **Advantage**: More reliable, validates your dataset works, separates concerns
- **Limitation**: Users need 4-view input (but can be generated separately in future)
- **What it requires**: Model learns temporal consistency across 4 synchronized views
- **Status**: Documented as **fallback** if Option 1 (single→multi) doesn't achieve sufficient quality

### Clarification from User:
> "I want users to upload any frontal face photo and audio, then generate multi-view video"

This revealed the **end goal is Option 1 (single→multi)**, and we decided to:
1. **Phase 1 (now)**: Pursue Option 1 (single→multi) directly - ambitious but aligned with end goal
2. **Fallback**: If Option 1 quality is insufficient, pivot to Option 2 (multi→multi) to validate data pipeline separately

### Rationale for Asking:
This is THE MOST CRITICAL architectural question because:

**Option 1 requires model to learn TWO hard things simultaneously**:
1. **Novel view synthesis**: Generate 3 unseen camera angles from 1 input
   - Requires understanding 3D geometry, face structure, lighting
   - State-of-the-art: Zero-1-to-3, SyncDreamer (separate research models)
2. **Speech-driven animation**: Sync lip movements with audio across all views
   - Requires audio-visual alignment, temporal consistency

**Option 2 requires model to learn ONE hard thing**:
1. **Multi-view animation**: Keep 4 views spatially consistent while animating
   - Easier problem: views already provided, just need temporal consistency

**Analogy**:
- Option 1 = Learning to play piano while simultaneously learning music theory
- Option 2 = Learning to play piano with sheet music already provided

### Decision Impact:
✅ **Chose Option 1 (single→multi) for Phase 1** → Matches end-user vision (one photo + audio → multi-view video)
📝 **Option 2 (multi→multi)** kept as **fallback** if Approach A stalls or quality is insufficient

---

## Q6: If single-view input, which camera should be the input?

⚠️ **NOTE**: This question **IS applicable** to our chosen approach (single→multi). For Phase 1 we fix the input camera to **cam_54** (frontal hemisphere view).

### Options Presented:

**Option 1: cam_54 (front hemisphere)**
- Most natural frontal view
- **Why**: Users typically have frontal photos, matches real-world use case
- **Trade-off**: Model only learns to generate views from frontal perspective

**Option 2: cam_28 (wide front context)**
- Wider field of view
- **Why**: More context, might help with background/scene understanding
- **Trade-off**: Less common user input angle

**Option 3: Any camera (data augmentation)**
- Randomly select input camera during training
- **Why**: Model becomes robust to any input angle, most flexible
- **Trade-off**: Harder to learn, more training time needed

### Rationale for Asking:
This question was **conditional on choosing Option 1 in Q5** (single→multi).

If training single→multi, I needed to know:
- **Fixed input angle**: Easier to learn, matches user expectations (e.g., always frontal)
- OR **Variable input angle**: Harder to learn, more flexible (any angle → multi-view)

### Decision Impact:
✅ **Chose cam_54 as fixed input view for Phase 1** → Matches typical frontal face photos users upload
📝 **Can extend to Option 3 (any camera) in Phase 2** → For more robust model that handles any input angle

---

## Q7: Which approach do you want to pursue for this training?

### Options Presented:

**Approach A: Single view → Multi-view (Direct, High Risk)**
- Input 1 camera, output 4-camera grid
- **Why I offered it**: You expressed desire for this in Q5
- **Risk**: Novel view synthesis + animation = two hard problems at once
- **Example models**: Zero-1-to-3, SyncDreamer do view synthesis ONLY (no animation)
- **Time estimate if it works**: 2-3 weeks
- **Time estimate if it fails**: 1-2 weeks wasted before pivoting

**Approach C: Multi-view → Multi-view (Safer, Recommended)**
- Input 2×2 grid, output animated 2×2 grid
- **Why I recommended it**: Separates concerns, validates dataset first
- **Advantage**: If this fails → know dataset/animation has issues. If Approach A fails → don't know if issue is data, view synthesis, or animation
- **Foundation**: Once working, can add view synthesis (either end-to-end or separate model)
- **Time estimate**: 1-2 weeks to working baseline
- **Status**: Documented as **fallback option** if Approach A doesn't achieve sufficient quality

**Approach: Try A, fallback to C if needed**
- Attempt ambitious first, pivot if needed
- **Why I offered it**: Maybe you want to shoot for the moon
- **Risk**: Might waste 1-2 weeks on failed experiments before pivoting

### Rationale for Asking:
Classic engineering risk management decision:

**Incremental approach (C → A)**:
- ✅ De-risk each component
- ✅ Validate dataset works
- ✅ Build foundation for harder problem
- ❌ Slower to final goal

**All-in approach (A directly)**:
- ✅ Faster if it works
- ❌ High risk of failure
- ❌ If fails, don't know root cause

**Optimistic approach (Try A, fallback to C)**:
- ✅ Attempt ambitious goal first
- ❌ Might waste 1-2 weeks before realizing need to pivot
- ❌ Same time as incremental if A fails

I wanted to know your **risk tolerance** and **timeline pressure**.

### Decision Impact:
✅ **Chose Approach A (single→multi)** → Ambitious, aligned with mentor confirmation and end-user vision
⚠️ **High risk acknowledged** → Novel view synthesis + animation is challenging, but directly achieves goal
📝 **Approach C (multi→multi)** documented as fallback → Use if Approach A quality is insufficient

---

## Q8: Which resolution do you want for your 2×2 grid training?

### Options Presented (refined after knowing GPU capacity):

**Option 2: 240×416 → 480×832 (Conservative)**
- 7% more pixels than standard S2V (448×832)
- **Why**: If you're unsure about VRAM, play it safe
- **When to choose**: First time training, want guaranteed success
- **Face detail**: Each camera is 240×416 - acceptable but not great

**Option 3: 280×520 → 560×1040 (Recommended)**
- 56% more pixels than standard S2V
- **Why**: Sweet spot for 4-8×48GB GPUs, noticeable quality improvement
- **Math**: 560×1040×81 frames = ~47M pixels per sample
  - With gradient checkpointing: ~32-40GB per GPU
  - With 48GB GPUs: Comfortable margin
- **Face detail**: Each camera is 280×520 - much better for facial details
- **Status**: Documented for Phase 2 after 448×832 is validated

**Option 4: 320×640 → 640×1280 (Ambitious)**
- 120% more pixels than standard S2V
- **Why**: If you want near-720p quality in final output
- **Trade-off**: ~2× slower training, ~40-48GB per GPU (cutting it close)
- **Face detail**: Each camera is 320×640 - excellent quality

### Rationale for Asking:
Now that I knew GPU capacity (4-8 × 48GB), I could calculate safe ranges:

**Memory calculation**:
```
Model weights (bf16): ~28GB
Activations = resolution × num_frames × channels × batch_size

Option 2 (480×832):  ~26-34GB per GPU → Very safe
Option 3 (560×1040): ~32-40GB per GPU → Safe
Option 4 (640×1280): ~40-48GB per GPU → Risky (OOM possible)
```

Wanted you to choose the **quality vs speed vs safety** trade-off:
- **Conservative** (Option 2): Guaranteed to work, lower quality
- **Balanced** (Option 3): Best quality/safety ratio
- **Aggressive** (Option 4): Best quality, higher risk

### Decision Impact:
✅ **Chose Option 1 (448×832) for Phase 1** → Non-invasive, fastest to validate
📝 **Option 3 (560×1040)** is Phase 2 after 448×832 is validated

---

## Implementation Choice: On-the-Fly Loading vs Preprocessing

### Your Suggestion (not initially offered):
> "Can we process data while loading instead of preprocessing?"

### Analysis:

**Option A: On-the-Fly Loading** ✅ **CHOSEN**

**Advantages**:
- ✅ No storage overhead (keep only 440GB original dataset)
- ✅ Fast iteration - change resolution/cameras instantly without reprocessing
- ✅ Easy debugging - fix code and restart, no wasted preprocessing time
- ✅ Flexible experimentation with different configurations
- ✅ If preprocessing approach is wrong, haven't wasted hours/days

**Trade-offs**:
- ⚠️ Potential CPU bottleneck (mitigated with multi-worker DataLoader)
- ⚠️ Slightly more complex data operators (but more flexible)

**Storage savings**:
- Preprocessing would need: ~150-200GB
- On-the-fly needs: 0GB additional (use original 440GB)

**Option B: Preprocessing (Fallback)**

**Advantages**:
- ✅ Maximum training speed (no CPU overhead during training)
- ✅ Simpler data operators (just load preprocessed files)

**Trade-offs**:
- ❌ 150-200GB additional storage needed
- ❌ Hours of preprocessing time before training
- ❌ If resolution/camera layout wrong, must re-preprocess
- ❌ Hard to experiment with different configurations

### Rationale:
Your insight was **excellent** because:
1. **Storage constraint**: You have limited SSD space, preprocessing wastes it
2. **Iteration speed**: Changing grid layout requires re-preprocessing (hours wasted)
3. **Debugging**: If data looks wrong, on-the-fly lets you fix instantly
4. **Modern CPUs**: Multi-core CPUs + multi-worker DataLoader handles it fine
5. **Risk mitigation**: If approach is wrong, haven't wasted preprocessing time

**CPU bottleneck mitigation**:
```python
DataLoader(dataset, num_workers=8, ...)  # 8 CPU workers pre-load batches
```
While GPU trains on batch N, CPUs prepare batch N+1, N+2, etc.

### Decision Impact:
✅ **On-the-fly loading as primary approach** → Save storage, fast iteration
📝 **Preprocessing as fallback** → If CPU becomes bottleneck (unlikely)

---

## Final Configuration Summary

Based on all decisions above:

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Goal** | Single→Multi-view (2×2 grid from 1 image) | Aligned with mentor confirmation, end-user vision |
| **Approach** | Single→multi (Approach A) | Ambitious but directly achieves goal, fallback to multi→multi if needed |
| **Resolution** | 224×416 per cam (448×832 total) | Original S2V resolution, non-invasive fine-tuning, minimizes variables |
| **GPUs** | 4-8 × RTX 6000 Ada (48GB) | Sufficient for 448×832, comfortable margin |
| **Pose guidance** | Skip initially | Simplify first iteration |
| **Data loading** | On-the-fly | Save storage, fast iteration |
| **Training mode** | LoRA first, then full if needed | Faster feedback, validate approach quickly |

---

## Alternative Paths (Documented for Future or Fallback)

### Path 1: Multi→Multi (Approach C) - FALLBACK
- **When to use**: If single→multi quality is insufficient or training fails
- **How**: Input 2×2 grid → Output animated 2×2 grid
- **Advantage**: De-risks view synthesis problem, validates data pipeline separately
- **Timeline**: 1-2 days to working baseline if fallback needed

### Path 2: Higher Resolution (560×1040 or 640×1280)
- **When to attempt**: After 448×832 succeeds and quality is validated
- **Options**: 560×1040 (+56% pixels) or 640×1280 (+120% pixels)
- **Requirement**: Monitor VRAM carefully, may need to reduce batch size or switch to full GPUs

### Path 3: Add Pose Guidance
- **When to attempt**: After initial training, if head motion looks unrealistic
- **Requirement**: Preprocess keypoints3d → pose skeleton videos

### Path 4: Preprocessing Approach
- **When to use**: If on-the-fly loading creates CPU bottleneck
- **Indicator**: Training throughput < 1 batch per 5 seconds (likely won't happen)

---

## Key Lessons from Decision Process

1. **Start simple, iterate**: Single→multi first (with multi→multi as fallback)
2. **Validate dataset first**: Don't combine multiple unknowns
3. **Match resolution to GPU capacity**: Don't guess, calculate
4. **Prioritize iteration speed**: On-the-fly > preprocessing for experimentation
5. **Document alternatives**: Capture roads not taken for future

---

## Questions That Guided Us

Each question eliminated uncertainty and narrowed the solution space:

```
Unknown: What are we building?
    ↓ [Q1: Goal]
Known: Multi-view 2×2 grid output

Unknown: What quality level?
    ↓ [Q2: Resolution - blocked]
    ↓ [Q3: GPU capacity]
Known: 4-8 × 48GB GPUs → 448×832 for Phase 1 (non-invasive), can do 560×1040 in Phase 2

Unknown: How complex should data be?
    ↓ [Q4: Pose guidance]
Known: Skip pose initially

Unknown: What's the input/output relationship?
    ↓ [Q5: Input type]
    ↓ [Q7: Approach choice]
Known: Single→multi now, multi→multi fallback if needed

Unknown: How to prepare data?
    ↓ [User suggestion: On-the-fly]
Known: On-the-fly loading, preprocessing as fallback

Unknown: Final resolution choice?
    ↓ [Q8: Resolution decision]
Known: 448×832 (Option 1) for Phase 1; 560×1040 (Option 3) for Phase 2 if needed
```

**Total unknowns eliminated**: 8 major decisions
**Result**: Clear, de-risked implementation plan
