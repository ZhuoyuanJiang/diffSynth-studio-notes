# Understanding the Three Input Dicts: inputs_shared, inputs_posi, inputs_nega

## The Key Question: Why Three Dictionaries?

**Short Answer:** To support **Classifier-Free Guidance (CFG)** during inference.

**During Training:** We typically use `cfg_scale=1` (no CFG), so we only need positive conditioning.

**During Inference:** We use `cfg_scale>1` (e.g., 7.5) to improve generation quality by contrasting positive and negative conditioning.

---

## What Goes in Each Dictionary?

### 1. `inputs_shared` - CFG-Insensitive Inputs
**Rule:** Put data that is **the same for both positive and negative conditioning** during CFG.

**Examples:**
- `input_video`: The raw video frames (used by VAE encoder)
- `latents`: VAE-encoded video (after processing)
- `height`, `width`, `num_frames`: Dimensions
- `cfg_scale`: CFG scale parameter
- `use_gradient_checkpointing`: Training configs
- **NOT audio or prompt** - these affect conditioning!

**Why separate?**
- These inputs are processed ONCE (not duplicated for CFG)
- Saves computation and memory during inference

---

### 2. `inputs_posi` - Positive Conditioning (CFG-Sensitive)
**Rule:** Put data that **defines what you WANT to generate**.

**Examples:**
- `prompt`: "A person speaking with a smile" ✅
- `prompt_embeds`: CLIP-encoded positive prompt (after processing)
- `audio_embeds`: wav2vec2-encoded audio (for S2V) - **THIS IS KEY!**
- `reference_image`: Face reference for consistency

**Why in inputs_posi?**
- During CFG, the model generates TWO predictions:
  - One conditioned on positive prompt + audio
  - One conditioned on negative prompt (no audio or neutral audio)
- The difference guides the model toward your desired output

---

### 3. `inputs_nega` - Negative Conditioning (CFG-Sensitive)
**Rule:** Put data that **defines what you DON'T want to generate** (or neutral baseline).

**Examples:**
- `prompt`: "" (empty string) or "low quality, blurry" ❌
- `prompt_embeds`: CLIP-encoded negative prompt (after processing)
- **NO audio_embeds** (or silent audio) for S2V negative conditioning

**Why separate from inputs_posi?**
- CFG formula: `pred = pred_nega + cfg_scale * (pred_posi - pred_nega)`
- By subtracting negative prediction, we "push away" from unwanted outputs
- Higher cfg_scale = stronger guidance toward positive, away from negative

---

## Your Specific Questions Answered

### Q: "Is audio in inputs_posi and video in inputs_shared because all models require video but not all models require audio?"

**Almost correct!** More precisely:

- **Video → inputs_shared** because:
  - The raw video is needed by VAE (CFG-insensitive)
  - The latents (encoded video) are shared for both positive/negative predictions
  - DiT needs latents regardless of conditioning

- **Audio → inputs_posi** because:
  - Audio is **conditioning** (tells DiT WHAT lip movements to generate)
  - During CFG, we want to contrast "with audio" vs "without audio"
  - Only applies to S2V models (not all Wan Video modes)

### Q: "Can I understand that for inputs not shared across all models we put in inputs_posi?"

**Not quite!** Better rule:

- **If it affects WHAT is generated (conditioning)** → `inputs_posi` or `inputs_nega`
- **If it's structural/technical (not conditioning)** → `inputs_shared`

**Examples:**

| Input | Category | Why? |
|-------|----------|------|
| `video` frames | `inputs_shared` | Structural (latent source) |
| `latents` | `inputs_shared` | Structural (what we denoise) |
| `height`, `width` | `inputs_shared` | Structural (dimensions) |
| `prompt` | `inputs_posi` | **Conditioning** (what to generate) |
| `audio` | `inputs_posi` | **Conditioning** (lip sync) |
| `reference_image` | `inputs_posi` | **Conditioning** (face identity) |
| `cfg_scale` | `inputs_shared` | Technical parameter |

### Q: "What do we usually put in inputs_negative or inputs_positive then?"

**inputs_posi (Positive Conditioning):**
```python
inputs_posi = {
    "prompt": "A person speaking with a cheerful expression",
    "prompt_embeds": tensor([1, 77, 768]),  # After CLIP encoding
    "audio_embeds": tensor([1, T_audio, 1280]),  # For S2V only
    "reference_image": PIL.Image,  # For face consistency (optional)
}
```

**inputs_nega (Negative Conditioning):**
```python
# Training (cfg_scale=1): Usually empty or copy of inputs_posi
inputs_nega = {}

# Inference (cfg_scale>1): Typically minimal conditioning
inputs_nega = {
    "prompt": "",  # Empty or "low quality, blurry, distorted"
    "prompt_embeds": tensor([1, 77, 768]),  # Encoded empty prompt
    # NO audio_embeds for S2V (we want to contrast with/without audio)
}
```

### Q: "What's the motivation of having these two variables instead of just one input_shared?"

**Motivation: Classifier-Free Guidance (CFG) Quality Improvement**

**Without CFG (single forward pass):**
```python
# Model tries to satisfy prompt, but output may be:
# - Less sharp
# - Less prompt-adherent
# - More generic
pred = dit(latents, t, prompt_embeds, audio_embeds)
```

**With CFG (two forward passes):**
```python
# Pass 1: With conditioning (positive)
pred_posi = dit(latents, t, prompt_embeds=positive, audio_embeds=audio)

# Pass 2: Without conditioning (negative)
pred_nega = dit(latents, t, prompt_embeds=empty, audio_embeds=None)

# Combine: Amplify the difference (push toward positive, away from negative)
pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
#      ^^^^^^^^^^^   ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
#      Baseline      Amplifier   "Direction" toward positive
```

**Result:** More prompt-adherent, sharper, higher-quality outputs!

**Analogy:**
- `pred_nega`: "Vague generic video"
- `pred_posi`: "Video matching your prompt+audio"
- `cfg_scale * difference`: "Push harder toward your desired output"

---

## Concrete Example: S2V Training vs Inference

### Training (cfg_scale=1, no CFG):
```python
# forward_preprocess() creates:
inputs_shared = {
    "input_video": [PIL.Image, ...],  # 81 frames
    "latents": tensor([1, 16, 81, H/8, W/8]),  # After VAE encoding
    "height": 720, "width": 1280, "num_frames": 81,
    "cfg_scale": 1,  # No CFG
}

inputs_posi = {
    "prompt": "A person speaking",
    "prompt_embeds": tensor([1, 77, 768]),
    "audio_embeds": tensor([1, T_audio, 1280]),  # From input_audio
}

inputs_nega = {}  # Empty (not used in training)

# One forward pass:
loss = dit(latents, t, **inputs_posi)
```

### Inference (cfg_scale=7.5, with CFG):
```python
# Inference code creates:
inputs_shared = {
    "latents": tensor([1, 16, 81, H/8, W/8]),  # Random noise initially
    "height": 720, "width": 1280, "num_frames": 81,
    "cfg_scale": 7.5,  # Strong CFG
}

inputs_posi = {
    "prompt": "A person speaking with a smile",
    "prompt_embeds": tensor([1, 77, 768]),
    "audio_embeds": tensor([1, T_audio, 1280]),
}

inputs_nega = {
    "prompt": "",  # Empty prompt
    "prompt_embeds": tensor([1, 77, 768]),  # Encoded empty prompt
    # NO audio_embeds!
}

# Two forward passes (in denoising loop):
for t in timesteps:
    pred_posi = dit(latents, t, **inputs_posi)
    pred_nega = dit(latents, t, **inputs_nega)
    pred = pred_nega + 7.5 * (pred_posi - pred_nega)  # CFG
    latents = scheduler.step(pred, t, latents)  # Denoise
```

---

## Understanding lora_base_model vs trainable_models

### Do they have to be the same?

**No, they don't have to be the same!** They serve different purposes:

- **`trainable_models`**: Which models to enable gradients for (full training)
- **`lora_base_model`**: Which model to add LoRA adapters to (parameter-efficient training)

### Common scenarios:

```python
# Scenario 1: LoRA training (typical for large models like 14B DiT)
trainable_models = "dit"
lora_base_model = "dit"
# → DiT frozen, only LoRA adapters trainable (~few MB)

# Scenario 2: Full fine-tuning (for smaller models)
trainable_models = "dit"
lora_base_model = None
# → Entire DiT trainable (~14B params)

# Scenario 3: Mixed (rare but possible)
trainable_models = "dit,vae"
lora_base_model = "dit"
# → DiT has LoRA (adapters trainable), VAE fully trainable

# Scenario 4: Error case (inconsistent)
trainable_models = "dit"
lora_base_model = "vae"  # VAE not in trainable_models!
# → This would likely cause issues
```

**Key insight:** `lora_base_model` should typically be a subset (or equal to) `trainable_models`. If you add LoRA to a model, that model must be in `trainable_models`.

---

## Data Flow: UnifiedDataset → forward_preprocess() → training_loss()

### 📥 INPUT: Raw data from UnifiedDataset
```python
data = {
    "video": [PIL.Image, PIL.Image, ...],  # 81 frames as PIL Images
    "audio": "/path/to/dataset/audio/sample001.wav",  # Path to audio file (NOT loaded yet!)
    "prompt": "A person speaking with neutral expression"  # Text prompt
}
```

### 🔄 PROCESSING: forward_preprocess() creates three dicts
```python
# BEFORE pipeline units:
inputs_shared = {
    "input_video": data["video"],  # Raw frames
    "input_audio": data["audio"],  # Audio path (added via extra_inputs)
    "height": 720, "width": 1280, "num_frames": 81,
    "cfg_scale": 1, ...
}

inputs_posi = {
    "prompt": data["prompt"]
}

inputs_nega = {}
```

### ⚙️ Pipeline units process the inputs:
```python
# Unit 1: WanVideoUnit_S2V (take_over=True)
# - Loads audio from inputs_shared["input_audio"]
# - Encodes audio → audio_embeds
# - Adds to inputs_posi["audio_embeds"]

# Unit 2: WanVideoUnit_PromptEmbedder (seperate_cfg=True)
# - Encodes prompt → prompt_embeds
# - Adds to inputs_posi["prompt_embeds"]

# Unit 3: WanVideoUnit_VAEEncoder (normal mode)
# - Encodes video → latents
# - Adds to inputs_shared["latents"]

# Unit 4: WanVideoUnit_ShapeChecker (normal mode)
# - Validates shapes
```

### 📤 OUTPUT: Merged dict ready for DiT
```python
# After merging {**inputs_shared, **inputs_posi}:
{
    # From inputs_shared (processed by units):
    "latents": torch.Tensor,  # [1, 16, 81, H/8, W/8] - VAE-encoded video
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    ... (other configs)

    # From inputs_posi (processed by units):
    "prompt_embeds": torch.Tensor,  # [1, 77, 768] - CLIP text embeddings
    "audio_embeds": torch.Tensor,  # [1, T_audio, 1280] - wav2vec2 audio embeddings
}
```

---

## Summary Table

| Dict | Purpose | Examples | Used in Training? | Used in Inference? |
|------|---------|----------|-------------------|-------------------|
| `inputs_shared` | Structural, CFG-insensitive data | video, latents, dims, configs | ✅ Yes | ✅ Yes |
| `inputs_posi` | **What you WANT** (positive conditioning) | prompt, audio, reference | ✅ Yes (primary) | ✅ Yes (with CFG) |
| `inputs_nega` | **What you DON'T want** (negative conditioning) | empty prompt, no audio | ❌ Usually empty | ✅ Yes (with CFG) |

**Key Takeaway:** The three-dict design enables CFG during inference while keeping training simple (cfg_scale=1, only use inputs_posi).

---

## Understanding special_operator_map for S2V Training

### Q: Do I need to define anything in special_operator_map for S2V?

**Short Answer:** **NO, you typically DON'T need special_operator_map for S2V training!**

### Why not?

The `special_operator_map` is used when you need **DIFFERENT processing** for different data types. Here's how UnifiedDataset decides what operator to use:

```python
# In UnifiedDataset.__getitem__():
for key in self.data_file_keys:  # e.g., ["video", "audio"]
    if key in data:
        if key in self.special_operator_map:
            data[key] = self.special_operator_map[key](data[key])  # Use special operator
        else:
            data[key] = self.main_data_operator(data[key])  # Use main operator
```

### For S2V, here's what happens:

**Your dataset CSV might look like:**
```csv
video,audio,prompt
videos/clip001.mp4,audios/clip001.wav,"A person speaking"
videos/clip002.mp4,audios/clip002.wav,"A person talking"
```

**Your training config:**
```python
dataset = UnifiedDataset(
    base_path="/data/RenderMe360",
    metadata_path="/data/RenderMe360/train.csv",
    data_file_keys=["video", "audio"],  # ← Load both video and audio
    main_data_operator=UnifiedDataset.default_video_operator(...),  # ← Processes video files
    special_operator_map={}  # ← EMPTY! No special operators needed
)
```

**What happens when loading:**
1. **For "video" key:** Uses `main_data_operator` → loads video as `[PIL.Image, ...]`
2. **For "audio" key:** Uses `main_data_operator` → BUT it just **returns the path string**!

**Wait, why does audio just return a path?**

Because `default_video_operator` includes `RouteByExtensionName`:
```python
# Inside default_video_operator:
RouteByExtensionName({
    "image": ToAbsolutePath(...) >> LoadImage(...),
    "video": ToAbsolutePath(...) >> LoadVideo(...),
})
```

When it sees `"clip001.wav"`, it doesn't match "image" or "video", so it **falls back** to returning the absolute path as a string.

**Then, WanVideoUnit_S2V loads the audio:**
```python
# In forward_preprocess():
inputs_shared["input_audio"] = data["audio"]  # Just the path string!

# Later, WanVideoUnit_S2V.process():
audio_path = inputs_shared["input_audio"]  # "audios/clip001.wav"
audio_waveform = load_audio(audio_path)  # Load audio file
audio_embeds = self.audio_encoder(audio_waveform)  # Encode
inputs_posi["audio_embeds"] = audio_embeds  # Add to inputs_posi
```

### When DO you need special_operator_map?

**Use special_operator_map when you need CUSTOM processing for specific keys.**

**Example 1: Animate (Pose + Face videos)**
```python
# Animate has 3 video inputs: main video, pose video, face video
special_operator_map={
    # Face video needs fixed 512x512 resolution (different from main video)
    "animate_face_video": ToAbsolutePath(...) >> LoadVideo(..., frame_processor=ImageCropAndResize(512, 512, ...))
}
```
From the training script we saw:
```bash
--data_file_keys "video,animate_pose_video,animate_face_video"
--extra_inputs "input_image,animate_pose_video,animate_face_video"
```
And in commented_train.py:
```python
special_operator_map={
    "animate_face_video": ToAbsolutePath(...) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
}
```

**Example 2: Custom Audio Processing (if you wanted to preprocess audio)**
```python
# If you wanted to load and preprocess audio in the dataset (instead of in WanVideoUnit_S2V)
special_operator_map={
    "audio": ToAbsolutePath(...) >> LoadAudio() >> ResampleAudio(16000) >> NormalizeAudio()
}
```
But this is **NOT recommended** for S2V because:
- WanVideoUnit_S2V already handles audio loading
- Keeping audio as paths is more flexible (can use different audio encoders)

### S2V Training: Complete Example

**Dataset CSV (train.csv):**
```csv
video,audio,prompt
videos/person1_clip1.mp4,audios/person1_clip1.wav,"A person speaking calmly"
videos/person2_clip1.mp4,audios/person2_clip1.wav,"A person talking with expression"
```

**Training Script:**
```python
dataset = UnifiedDataset(
    base_path="/data/RenderMe360",
    metadata_path="/data/RenderMe360/train.csv",
    data_file_keys=["video", "audio"],  # Load video and audio columns
    main_data_operator=UnifiedDataset.default_video_operator(
        base_path="/data/RenderMe360",
        num_frames=81,
        max_pixels=1048576,
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
    ),
    special_operator_map={}  # ← EMPTY for S2V! No special processing needed
)

model = WanTrainingModule(
    ...,
    extra_inputs="input_audio",  # ← CRITICAL! Tells forward_preprocess to include audio
)
```

**What you get from dataset[0]:**
```python
{
    "video": [PIL.Image, PIL.Image, ...],  # 81 frames
    "audio": "/data/RenderMe360/audios/person1_clip1.wav",  # Just the path!
    "prompt": "A person speaking calmly"
}
```

**Then forward_preprocess() does:**
```python
inputs_shared["input_audio"] = data["audio"]  # Path string
# WanVideoUnit_S2V will load and encode it later
```

### Summary Table: When to Use special_operator_map

| Scenario | Need special_operator_map? | Why? |
|----------|---------------------------|------|
| **S2V (audio + video)** | ❌ NO | Audio path is fine, WanVideoUnit_S2V loads it |
| **T2V (text + video)** | ❌ NO | Only video, main_data_operator handles it |
| **I2V (image + video)** | ❌ NO | Image/video both handled by main_data_operator |
| **Animate (pose + face videos)** | ✅ YES | Face video needs different resolution (512x512) |
| **Custom preprocessing** | ✅ YES | If you need special handling for specific keys |

### Key Takeaway for S2V

**For S2V training on RenderMe360:**
```python
special_operator_map={}  # Leave empty!
# OR simply omit it (defaults to empty dict)
```

**Just make sure:**
1. CSV has "audio" column with audio file paths
2. `data_file_keys` includes "audio"
3. `extra_inputs` includes "input_audio"
4. WanVideoUnit_S2V will handle the rest!
