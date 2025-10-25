# Learning Notes for WanVideoPipeline

This document contains detailed explanations for key concepts in `diffsynth/pipelines/wan_video_new.py`.

---

## Q1: What is `self.units` and how are units executed?

**Short Answer:**
- `self.units` is a **list of 20 pipeline units** that execute **sequentially** (one after another)
- Each unit transforms the three input dictionaries: `inputs_shared`, `inputs_posi`, `inputs_nega`

**Sequential Execution Order:**

```
1. ShapeChecker()
   ↓ outputs: validated height, width, num_frames
2. NoiseInitializer()
   ↓ outputs: initial noise tensor
3. PromptEmbedder()
   ↓ outputs: prompt_embeds (text encoding)
4. S2V()
   ↓ outputs: audio_embeds (audio encoding) ⭐
5. ImageEmbedderVAE()
   ↓ outputs: image embeddings
... and so on (20 units total)
```

**Detailed Explanation:**

### The Unit List (from `__init__`)

```python
self.units = [
    WanVideoUnit_ShapeChecker(),         # 1. Validate dimensions
    WanVideoUnit_NoiseInitializer(),     # 2. Create initial noise
    WanVideoUnit_PromptEmbedder(),       # 3. Text → prompt_embeds
    WanVideoUnit_S2V(),                  # 4. Audio → audio_embeds ⭐
    WanVideoUnit_ImageEmbedderVAE(),     # 5. Encode image with VAE
    # ... 15 more units
]
```

### How Units Execute (from `__call__`)

```python
# Line 706-707 in __call__() method:
for unit in self.units:
    inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
        unit, self, inputs_shared, inputs_posi, inputs_nega
    )
```

**Execution Flow:**

```
Initial State:
  inputs_shared = {"input_audio": audio_waveform, "height": 480, ...}
  inputs_posi = {"prompt": "a person talking"}
  inputs_nega = {"negative_prompt": "blurry"}

↓ Unit 1: ShapeChecker
  - Validates height, width, num_frames
  - Updates: inputs_shared["height"] = 480, inputs_shared["width"] = 832

↓ Unit 2: NoiseInitializer
  - Creates random noise tensor
  - Updates: inputs_shared["noise"] = torch.randn(...)

↓ Unit 3: PromptEmbedder
  - Encodes text with T5
  - Updates: inputs_posi["prompt_embeds"] = tensor(...)

↓ Unit 4: S2V ⭐ (KEY FOR YOUR LEARNING)
  - Extracts: input_audio from inputs_shared
  - Encodes: audio → audio_embeds with wav2vec2
  - Updates: inputs_posi["audio_embeds"] = tensor(...)
  - Removes: inputs_shared["input_audio"] (no longer needed)

↓ Unit 5: ImageEmbedderVAE
  - If input_image provided, encodes it
  - Updates: inputs_shared["first_frame_latents"] = tensor(...)

... (continues for all 20 units)

Final State:
  inputs_shared = {"latents": noise, "height": 480, ...}
  inputs_posi = {"prompt_embeds": ..., "audio_embeds": ..., "image_embeds": ...}
  inputs_nega = {"prompt_embeds": ..., "audio_embeds": 0.0 * ..., ...}
```

**Key Takeaway:** Each unit adds new data to the dictionaries. By the time all units finish, `inputs_posi` contains all the conditioning needed for the DiT (text, audio, image embeddings).

---

## Q2: What is a `state_dict` in PyTorch?

**Short Answer:**
A `state_dict` is a **Python dictionary** that stores all the learnable parameters (weights and biases) of a model.

**Detailed Explanation:**

### What's Inside a State Dict?

```python
# Example: A simple linear layer
layer = torch.nn.Linear(in_features=512, out_features=256)

# Get its state_dict
state_dict = layer.state_dict()

print(state_dict)
# Output:
# {
#     'weight': tensor([[0.1, 0.2, ...], [...], ...]),  # Shape: (256, 512)
#     'bias': tensor([0.01, 0.02, ...])                 # Shape: (256,)
# }
```

### For a Full Model (e.g., DiT with LoRA):

```python
# A DiT model with LoRA has a state_dict like:
{
    # Original DiT weights
    'blocks.0.attn.q_proj.weight': tensor(...),  # Shape: (dim, dim)
    'blocks.0.attn.k_proj.weight': tensor(...),
    'blocks.0.attn.v_proj.weight': tensor(...),

    # LoRA weights (added during training)
    'blocks.0.attn.q_proj.lora_A.default.weight': tensor(...),  # Shape: (rank, dim)
    'blocks.0.attn.q_proj.lora_B.default.weight': tensor(...),  # Shape: (dim, rank)

    # More layers...
}
```

### How State Dicts are Saved and Loaded

```python
# Save model weights to file
torch.save(model.state_dict(), "model_checkpoint.pth")

# Load model weights from file
state_dict = torch.load("model_checkpoint.pth")
model.load_state_dict(state_dict)
```

**Key Takeaway:** A `state_dict` is just a dictionary mapping parameter names (strings) to their tensor values. When you train a model, you save its `state_dict` to a file. Later, you can load that file to restore the trained weights.

---

## Q3: What is `lora_config` and what files does it refer to?

**Short Answer:**
- `lora_config` points to a **trained LoRA checkpoint file** (e.g., `.pth`, `.safetensors`)
- This file contains **LoRA adapter weights** (lora_A and lora_B matrices)

**What's in a LoRA File?**

A LoRA checkpoint file contains only the trained LoRA adapter weights:

```python
# Contents of a LoRA file (e.g., my_lora.pth):
{
    'blocks.0.attn.q_proj.lora_A.default.weight': tensor(4096, 8),
    'blocks.0.attn.q_proj.lora_B.default.weight': tensor(8, 4096),
    'blocks.0.attn.k_proj.lora_A.default.weight': tensor(4096, 8),
    'blocks.0.attn.k_proj.lora_B.default.weight': tensor(8, 4096),
    # ... (only LoRA weights, NOT the original DiT weights)
}
```

**File size:** ~100 MB (vs 28 GB for full DiT)

**What does `lora_config` look like?**

`lora_config` can be **two types**:

```python
# Type 1: String (direct file path)
lora_config = "/path/to/my_trained_lora.pth"  # Path to LoRA checkpoint file

# Type 2: ModelConfig object (auto-download from internet)
lora_config = ModelConfig(
    model_id="your_username/my_lora_model",  # Repo ID on ModelScope/HuggingFace
    origin_file_pattern="*.safetensors"       # File pattern to download
)
```

**Usage example:**

```python
# Load LoRA from local file
pipe.load_lora(
    module=pipe.dit,
    lora_config="/path/to/my_trained_lora.pth"  # ← This is lora_config (string)
)

# OR load from ModelScope/HuggingFace
pipe.load_lora(
    module=pipe.dit,
    lora_config=ModelConfig(...)  # ← This is lora_config (ModelConfig object)
)
```

---

## Q4: What does "load LoRA into module with specified alpha" mean?

**Short Answer:**
Yes! It means **adding LoRA adapters to specific layers** (like `q_proj`, `k_proj`) with a **strength multiplier (alpha)**.

**How It Works:**

When you call `pipe.load_lora(module=pipe.dit, ...)`, the function:

1. **Reads the LoRA checkpoint file** and looks at the key names:
   ```python
   # LoRA checkpoint contains:
   {
       'blocks.0.attn.q_proj.lora_A.default.weight': tensor(...),
       'blocks.0.attn.q_proj.lora_B.default.weight': tensor(...),
       'blocks.0.attn.k_proj.lora_A.default.weight': tensor(...),
       # ... more layers
   }
   ```

2. **Parses the key names** to determine WHERE to add LoRA:
   - Key: `'blocks.0.attn.q_proj.lora_A.default.weight'`
   - Means: Add LoRA to layer `pipe.dit.blocks[0].attn.q_proj`

3. **Adds the LoRA matrices** to those layers:
   ```python
   # Before: layer only has original weight W
   layer = pipe.dit.blocks[0].attn.q_proj  # Has only W

   # After: layer now has W + LoRA adapters
   layer.lora_A = lora_A_from_checkpoint  # Added
   layer.lora_B = lora_B_from_checkpoint  # Added
   ```

**Where LoRA is Added:**

```
DiT Model Structure:
├─ blocks[0]
│  ├─ attn
│  │  ├─ q_proj (Linear)  ← LoRA added here (if in checkpoint)
│  │  ├─ k_proj (Linear)  ← LoRA added here (if in checkpoint)
│  │  ├─ v_proj (Linear)  ← LoRA added here (if in checkpoint)
│  │  └─ out_proj (Linear) ← LoRA added here (if in checkpoint)
│  ├─ mlp
│  │  ├─ fc1 (Linear)     ← LoRA added here (if in checkpoint)
│  │  └─ fc2 (Linear)     ← LoRA added here (if in checkpoint)
├─ blocks[1]
│  └─ ... (same structure)
... (40 blocks total)
```

**Example:**

```python
# Load trained LoRA into DiT
pipe.load_lora(
    module=pipe.dit,                              # Target: DiT model
    lora_config="./outputs/lora_audio_style.pth", # Path to trained LoRA checkpoint
    alpha=1.0                                     # Strength multiplier
)

# Now when you run inference:
video = pipe(
    prompt="a person talking",
    input_audio=my_audio
)
# The generated video will have the learned style from your LoRA training!
```

**Key Takeaway:** The layers don't already have LoRA. The checkpoint **tells the function which layers to add LoRA to** via the key names, then the function adds `lora_A` and `lora_B` matrices to those layers.

---

## Q5: What is Multi-LoRA?

**Short Answer:**
Multi-LoRA means loading **multiple LoRA checkpoints simultaneously** into the same model. Each LoRA can represent a different style or concept.

**Detailed Explanation:**

### How Multi-LoRA Works

Instead of just one LoRA, you can stack multiple:

```python
# Forward pass with SINGLE LoRA:
output = input @ W + (input @ lora_A @ lora_B) * alpha

# Forward pass with MULTI-LoRA (3 LoRAs stacked):
output = input @ W
       + (input @ lora_A_1 @ lora_B_1) * alpha_1  # LoRA 1: "anime style"
       + (input @ lora_A_2 @ lora_B_2) * alpha_2  # LoRA 2: "realistic faces"
       + (input @ lora_A_3 @ lora_B_3) * alpha_3  # LoRA 3: "fast motion"
```

### Example Use Case

```python
# Train 3 different LoRAs for different styles:
# 1. lora_anime.pth - trained on anime videos
# 2. lora_realistic.pth - trained on realistic videos
# 3. lora_fast_motion.pth - trained on fast motion videos

# Load all 3 into the model:
pipe.load_lora(pipe.dit, "lora_anime.pth", alpha=0.5, hotload=True)
pipe.load_lora(pipe.dit, "lora_realistic.pth", alpha=0.3, hotload=True)
pipe.load_lora(pipe.dit, "lora_fast_motion.pth", alpha=0.2, hotload=True)

# Generate video with ALL 3 styles blended:
video = pipe(prompt="a person talking", input_audio=audio)
# Result: 50% anime + 30% realistic + 20% fast motion
```

### `hotload=True` vs `hotload=False`

```python
# hotload=False (default): REPLACE existing LoRA
pipe.load_lora(pipe.dit, "lora1.pth", hotload=False)  # Loads LoRA 1
pipe.load_lora(pipe.dit, "lora2.pth", hotload=False)  # Replaces LoRA 1 with LoRA 2

# hotload=True: APPEND to existing LoRA (multi-LoRA)
pipe.load_lora(pipe.dit, "lora1.pth", hotload=True)   # Loads LoRA 1
pipe.load_lora(pipe.dit, "lora2.pth", hotload=True)   # Adds LoRA 2 (now both active)
pipe.load_lora(pipe.dit, "lora3.pth", hotload=True)   # Adds LoRA 3 (now all 3 active)
```

**Key Takeaway:** Multi-LoRA lets you combine multiple trained LoRAs to blend different styles/concepts in one generation. Use `hotload=True` to append instead of replace.

---

## Summary: The `load_lora()` Method Flow

```python
def load_lora(self, module, lora_config, alpha=1.0, hotload=False, state_dict=None):
```

**Step-by-step:**

1. **Get the LoRA weights** (load FROM file or use provided state_dict)
   - See detailed comments in `commented_wan_video_new.py` lines 196-205

2. **Inject LoRA into module**:
   - If `hotload=True`: Append to existing LoRA lists (multi-LoRA)
   - If `hotload=False`: Use `GeneralLoRALoader` to replace existing LoRA

3. **Result**: The module (e.g., `pipe.dit`) now has LoRA adapters added to its layers, scaled by `alpha`.

**Where does the trained LoRA file come from?**
- Created during **training** (e.g., by running `train.py`)
- Training script saves checkpoints like: `./outputs/checkpoint_step_1000.pth`
- Later, you load FROM that saved checkpoint for inference

---

## Practical Example: Training and Using LoRA for S2V

### Training Phase:

```bash
# Train LoRA on your custom audio dataset
accelerate launch train.py \
    --dataset_base_path /data/my_audio_videos \
    --lora_base_model "dit" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir ./outputs

# After training, you get: ./outputs/lora_checkpoint.pth
```

### Inference Phase:

```python
# Load base model
pipe = WanVideoPipeline.from_pretrained(model_configs=[...])

# Load your trained LoRA
pipe.load_lora(
    module=pipe.dit,
    lora_config="./outputs/lora_checkpoint.pth",
    alpha=1.0
)

# Generate video with your custom LoRA style
video = pipe(
    prompt="a person talking about AI",
    input_audio=my_audio,
    height=480,
    width=832,
    num_frames=81
)
```

**What happens:** The DiT now has your trained LoRA adapters, so the generated video will follow the patterns learned from your custom dataset!

---

## Understanding `training_loss()` - How Diffusion Models Are Trained

### Q: What's the difference between `timestep_id` and `timestep`?

```python
timestep_id = torch.randint(0, 1000, (1,))  # Random integer: e.g., 347
timestep = scheduler.timesteps[timestep_id]  # Actual timestep value: e.g., 0.347
```

**Answer:**
- `timestep_id`: **Index** into the scheduler's timestep array (integer 0-999)
- `timestep`: **Actual noise level value** used in computation (float, often normalized 0.0-1.0)

**Example:**
```python
scheduler.timesteps = [0.0, 0.001, 0.002, ..., 0.998, 0.999, 1.0]  # 1000 values
                       ↑                           ↑
                    timestep_id=0              timestep_id=347
                    timestep=0.0               timestep=0.347
```

**Why the conversion?**
Neural networks work better with **continuous values** (0.347) than discrete indices (347). The actual timestep value gets embedded and fed to the model.

---

### Q: Why sample random timesteps during training?

**Answer:**
During training, we need to teach the model to denoise at **any noise level**:
- Timestep 0 = completely clean image
- Timestep 999 = completely noisy image (pure noise)

**Example training batches:**
```
Batch 1: Sample timestep = 100  (slightly noisy)
Batch 2: Sample timestep = 500  (medium noisy)
Batch 3: Sample timestep = 900  (very noisy)
```

By randomly sampling, the model learns to denoise from any starting point. This is crucial for inference, where we start from pure noise (timestep 1000) and iteratively denoise to clean (timestep 0).

---

### Q: If timestep = 5, do we add 5 noises sequentially?

**Answer: NO!** We add noise **in one step** directly to the target noise level.

**The formula inside `scheduler.add_noise()`:**
```python
noisy_latents = sqrt(alpha_t) * clean_latents + sqrt(1 - alpha_t) * noise
```

**Example with timestep = 5:**
```python
# Suppose alpha_5 = 0.99 (still mostly clean at timestep 5)
noisy_latents = 0.995 * clean_latents + 0.1 * random_noise
# Result: slightly noisy version of clean_latents
```

**Key insight:** It's **one mathematical operation**, not 5 sequential noise additions!

---

### Q: Where does `alpha_t` come from? Higher timestep = lower alpha?

**Answer:**
`alpha_t` is defined by the **noise schedule** in the scheduler (e.g., FlowMatchScheduler):

```python
# Simplified example:
class Scheduler:
    def __init__(self):
        self.betas = torch.linspace(0.0001, 0.02, 1000)  # Noise increase rate
        self.alphas = 1 - self.betas  # Signal retention rate
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

# Example values:
# timestep=0:   alpha_t = 0.9999  (99.99% clean signal)
# timestep=500: alpha_t = 0.5     (50% clean signal)
# timestep=999: alpha_t = 0.0001  (0.01% clean signal, almost pure noise)
```

**Yes! Higher timestep = lower alpha_t:**
- Timestep 0: alpha_t ≈ 1.0 → noisy_latents ≈ clean_latents (no noise)
- Timestep 500: alpha_t ≈ 0.5 → noisy_latents = 50% clean + 50% noise
- Timestep 999: alpha_t ≈ 0.0 → noisy_latents ≈ pure noise

---

### Q: What is `training_target`? Is it predicted or ground truth?

**Answer: `training_target` is GROUND TRUTH, NOT predicted!**

```python
# What we KNOW (ground truth):
input_latents = [clean video]  # Ground truth
noise = torch.randn(...)        # Ground truth (we generated it ourselves)
training_target = noise         # Ground truth (same as the noise we added)

# What the MODEL predicts:
noise_pred = model(noisy_latents, ...)  # Model's PREDICTION of the noise

# Loss compares:
loss = MSE(noise_pred, training_target)
     = MSE(predicted_noise, actual_noise)
     = MSE(what_model_guessed, what_we_actually_added)
```

**We know exactly how much noise we added because we added it ourselves during training!**

**Training flow:**
1. We have clean video (ground truth)
2. We generate random noise (ground truth)
3. We add noise to clean video → noisy video (we know the formula)
4. We ask model: "Given this noisy video, what noise was added?"
5. Model predicts noise (this is the prediction)
6. We compare model's prediction to the actual noise we added
7. Compute loss and update model weights

---

### Q: During inference, does model output noise or clean video?

**Answer: Model outputs NOISE, then we subtract it to get cleaner video!**

```python
# ===== INFERENCE: Starting from Pure Noise =====

# Initial state (timestep 1000):
latents = torch.randn(...)  # Pure random noise

# Denoising Loop (50 steps):
for timestep in [1000, 980, 960, ..., 20, 0]:

    # STEP 1: Model predicts noise
    noise_pred = model(
        latents=latents,           # Current noisy latents
        audio_embeds=audio_embeds,
        prompt_embeds=prompt_embeds,
        timestep=timestep          # Tell model noise level
    )
    # ↑ MODEL OUTPUT: predicted noise tensor

    # STEP 2: Subtract predicted noise to get cleaner latents
    latents = scheduler.step(noise_pred, timestep, latents)
    # ↑ This does: latents = latents - scale * noise_pred
    # Result: slightly cleaner latents

# After 50 iterations:
latents ≈ clean video latents (at timestep 0)

# STEP 3: VAE decodes latents to video frames
video_frames = vae.decode(latents)
# ↑ INFERENCE FINAL OUTPUT: clean video frames
```

**Key insight:**
- **Model output (each step):** Predicted noise
- **Inference final output:** Clean video (after removing all predicted noise + VAE decoding)

**Analogy:**
Think of it like peeling an onion:
- Model says: "Remove this layer (noise)"
- You remove that layer → slightly cleaner
- Model says: "Remove this next layer (noise)"
- You remove that layer → even cleaner
- ... repeat 50 times ...
- Final result: clean core (video)

---

### Complete Training Flow Example

```python
# ===== INPUTS =====
input_latents = [clean video latent]  # Shape: (1, 16, 21, 30, 52)
noise = torch.randn_like(input_latents)  # Random Gaussian noise
audio_embeds = [audio features from wav2vec2]
prompt_embeds = [text features from T5]

# ===== STEP 1: Sample Random Timestep =====
timestep_id = torch.randint(0, 1000, (1,))  # Say we get: 347
timestep = scheduler.timesteps[347]  # Get actual value (e.g., 0.347)

# ===== STEP 2: Add Noise (ONE STEP, NOT SEQUENTIAL) =====
# Suppose alpha_347 = 0.65
noisy_latents = sqrt(0.65) * input_latents + sqrt(0.35) * noise
# Result: 65% clean + 35% noise

inputs["latents"] = noisy_latents  # This goes to the model

# ===== STEP 3: What Should Model Predict? =====
training_target = noise  # The original pure noise we added

# ===== STEP 4: Model Forward Pass =====
noise_pred = model(
    latents=noisy_latents,       # Input: partially noisy
    audio_embeds=audio_embeds,   # Conditioning
    prompt_embeds=prompt_embeds, # Conditioning
    timestep=timestep            # Tell model noise level
)

# ===== STEP 5: Compute Loss =====
loss = MSE(noise_pred, training_target)
     = MSE(predicted_noise, actual_noise)

# If loss is low: Model correctly identified the noise
# Backprop → model learns to denoise better
```

---

### Summary Table

| Variable | What It Is | Example Value |
|----------|-----------|---------------|
| `timestep_id` | Index into timestep array | 347 (integer) |
| `timestep` | Actual noise level value | 0.347 (float) |
| `alpha_t` | Signal retention rate | 0.65 (from noise schedule) |
| `input_latents` | Clean video latent (ground truth) | Clean tensor |
| `noise` | Random Gaussian noise (ground truth) | Random tensor |
| `inputs["latents"]` | Noisy version at timestep t | 65% clean + 35% noise |
| `training_target` | What model should predict (ground truth) | Original `noise` tensor |
| `noise_pred` | What model actually predicted | Model's guess of noise |
| `loss` | Prediction error | MSE(noise_pred, training_target) |

---

### Key Takeaways

1. **timestep_id** = index (integer), **timestep** = actual noise level (float)
2. **alpha_t** comes from noise schedule, **higher timestep = lower alpha_t = more noise**
3. **Noise is added in ONE STEP**, not sequentially
4. **training_target is GROUND TRUTH** (the noise we added), not a prediction
5. **Model outputs NOISE**, not clean video directly
6. **Inference = iteratively removing predicted noise** over 50 steps
7. **Final video = VAE decode(denoised latents)**

---

## Understanding `from_pretrained()` Parameters

### Q1: What do these 3 variables look like?

**Example: Loading Wan2.2-S2V-14B for inference**

```python
from diffsynth import WanVideoPipeline, ModelConfig

# ===== 1. model_configs: List of ALL models needed =====
model_configs = [
    # Model 1: DiT (Diffusion Transformer - the main model)
    ModelConfig(
        model_id="Wan-AI/Wan2.2-S2V-14B",
        origin_file_pattern="diffusion_pytorch_model*.safetensors"
    ),

    # Model 2: VAE (Video Autoencoder)
    ModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        origin_file_pattern="Wan2.1_VAE.pth"
    ),

    # Model 3: Text Encoder (T5)
    ModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
    ),

    # Model 4: Image Encoder (CLIP)
    ModelConfig(
        model_id="Wan-AI/Wan2.1-I2V-14B-480P",
        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ),

    # Model 5: Audio Encoder (wav2vec2) ⭐ FOR S2V
    ModelConfig(
        model_id="Wan-AI/Wan2.2-S2V-14B",
        origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"
    ),
]
# So model_configs is a list of 5 ModelConfig objects (not 20!)

# ===== 2. tokenizer_config: T5 tokenizer (for text encoding) =====
tokenizer_config = ModelConfig(
    model_id="Wan-AI/Wan2.1-T2V-1.3B",
    origin_file_pattern="google/*"  # All files in google/ folder
)

# ===== 3. audio_processor_config: wav2vec2 processor (for audio preprocessing) =====
audio_processor_config = ModelConfig(
    model_id="Wan-AI/Wan2.2-S2V-14B",
    origin_file_pattern="wav2vec2-large-xlsr-53-english"
)

# ===== Now create pipeline =====
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,              # List of 5 models
    tokenizer_config=tokenizer_config,        # 1 tokenizer
    audio_processor_config=audio_processor_config  # 1 audio processor
)
```

**Why not 20 models?**

You're thinking of the **20 pipeline units** (`self.units`), but those are **processing steps**, not models!

**Models (5-6):**
- DiT (main model)
- VAE
- Text Encoder (T5)
- Image Encoder (CLIP)
- Audio Encoder (wav2vec2)
- Motion Controller (optional)

**Pipeline Units (20):**
- Processing steps that USE the models
- Example: `WanVideoUnit_S2V` uses the `audio_encoder` model
- Example: `WanVideoUnit_PromptEmbedder` uses the `text_encoder` model

---

### Q2: What does "map filename to canonical repository" mean?

**The Problem:**

Multiple model versions share the same files:

```
Wan2.1-T2V-1.3B repo contains:
├─ Wan2.1_VAE.pth  ← VAE model
├─ models_t5_umt5-xxl-enc-bf16.pth  ← T5 encoder

Wan2.2-S2V-14B repo ALSO needs:
├─ Wan2.1_VAE.pth  ← Same VAE!
├─ models_t5_umt5-xxl-enc-bf16.pth  ← Same T5!
```

Without redirection, you'd download the same file TWICE from different repos.

**The Solution (Redirection):**

```python
redirect_dict = {
    # Map: filename → canonical (main) repository
    "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
    "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
}
```

**Example:**

```python
# You ask to load from Wan2.2-S2V-14B:
ModelConfig(
    model_id="Wan-AI/Wan2.2-S2V-14B",
    origin_file_pattern="Wan2.1_VAE.pth"
)

# Redirection logic:
if "Wan2.1_VAE.pth" in redirect_dict:
    # Redirect to canonical repo
    model_id = "Wan-AI/Wan2.1-T2V-1.3B"  # Use this instead!

# Result: Downloads from Wan2.1-T2V-1.3B (the original source)
# Saves to: ./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

# Later, when Wan2.2-S2V-14B needs it:
# Already exists at ./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
# No duplicate download!
```

**Analogy:**
It's like having multiple classes that need the same textbook. Instead of each student buying their own copy, they all borrow from the library (canonical source).

---

### Q3: What is `pipe`?

**`pipe` is an INSTANCE of `WanVideoPipeline` class.**

Think of it like this:

```python
# Class = Blueprint (the code definition)
class WanVideoPipeline(BasePipeline):
    def __init__(self):
        self.dit = None
        self.vae = None
        # ...

    def __call__(self, prompt, input_audio):
        # ... generate video

# Instance = Actual object created from blueprint
pipe = WanVideoPipeline.from_pretrained(...)
# ↑ pipe is now a REAL pipeline with loaded models
```

**What's inside `pipe`?**

```python
pipe = WanVideoPipeline.from_pretrained(...)

# Now pipe contains:
pipe.dit              # ← The 14B DiT model (loaded)
pipe.vae              # ← The VAE model (loaded)
pipe.text_encoder     # ← The T5 encoder (loaded)
pipe.audio_encoder    # ← The wav2vec2 encoder (loaded) ⭐
pipe.scheduler        # ← The noise scheduler
pipe.units            # ← List of 20 processing units
pipe.device           # ← "cuda"
pipe.torch_dtype      # ← torch.bfloat16

# And many methods:
pipe.__call__()       # Generate video
pipe.training_loss()  # Compute training loss
pipe.load_lora()      # Load LoRA weights
```

**Analogy:**

Think of `pipe` as a **factory**:

```
WanVideoPipeline class = Factory blueprint (instructions)
pipe = The actual factory building (constructed and ready)

Inside the factory:
- dit = Main assembly line (14B parameters)
- vae = Packaging department
- audio_encoder = Audio processing station ⭐
- text_encoder = Text processing station
- units = 20 conveyor belts (sequential processing)

When you call pipe(...):
- Raw materials (audio, text) enter
- Go through 20 conveyor belts (units)
- Main assembly line (dit) builds the product
- Packaging (vae) prepares final output
- Clean video comes out!
```

**Visual Representation:**

```
pipe (WanVideoPipeline instance)
│
├─ Models (the workers):
│  ├─ dit: WanModel (14B params)
│  ├─ vae: WanVideoVAE
│  ├─ text_encoder: WanTextEncoder
│  ├─ audio_encoder: Wav2Vec2Model ⭐
│  └─ image_encoder: WanImageEncoder
│
├─ Processing units (the assembly line):
│  ├─ units[0]: ShapeChecker
│  ├─ units[1]: NoiseInitializer
│  ├─ units[2]: PromptEmbedder
│  ├─ units[3]: S2V (uses audio_encoder) ⭐
│  ├─ units[4-19]: Other units...
│  └─ post_units[0]: S2V post-processing
│
├─ Configuration:
│  ├─ device: "cuda"
│  ├─ torch_dtype: torch.bfloat16
│  └─ scheduler: FlowMatchScheduler
│
└─ Methods (the operations):
   ├─ __call__(): Generate video (inference)
   ├─ training_loss(): Compute loss (training)
   ├─ load_lora(): Load LoRA weights
   └─ from_pretrained(): Create pipeline (class method)
```

### How `pipe` is Created and Used

**Step 1: Create the pipeline (load models)**

```python
pipe = WanVideoPipeline.from_pretrained(
    model_configs=[
        ModelConfig(...),  # DiT
        ModelConfig(...),  # VAE
        ModelConfig(...),  # Text encoder
        ModelConfig(...),  # Audio encoder ⭐
    ]
)
```

**What happened:**
- Downloaded 4 model files (or loaded from cache)
- Created `pipe` object
- Loaded models into `pipe.dit`, `pipe.vae`, etc.
- `pipe` is now ready to use!

**Step 2: Use the pipeline (generate video)**

```python
video = pipe(
    prompt="a person talking",
    input_audio=my_audio_waveform,
    height=480,
    width=832,
    num_frames=81
)
```

**What happened inside `pipe.__call__()`:**
1. Organized inputs into 3 dicts
2. Ran 20 pipeline units sequentially
   - Unit 3 (`S2V`): `input_audio` → `audio_embeds` (using `pipe.audio_encoder`)
3. Denoising loop: `pipe.dit` predicts noise 50 times
4. VAE decode: `pipe.vae` converts latents to frames
5. Returns: list of PIL.Image frames

**Step 3: Train the pipeline (fine-tune)**

```python
loss = pipe.training_loss(
    input_latents=clean_video,
    noise=random_noise,
    audio_embeds=audio_features,
    prompt_embeds=text_features
)

loss.backward()  # Update pipe.dit weights
```

---

## Understanding Inference Arguments

### What are these arguments in `__call__()`?

```python
# Boundary
switch_DiT_boundary: Optional[float] = 0.875,
# Scheduler
num_inference_steps: Optional[int] = 50,
sigma_shift: Optional[float] = 5.0,
# Speed control
motion_bucket_id: Optional[int] = None,
```

**`switch_DiT_boundary: 0.875`**

Switches from DiT1 to DiT2 partway through generation (for Animate mode only). For S2V with only 1 DiT, this doesn't apply. Keep default 0.875.

**`num_inference_steps: 50`**

Number of denoising iterations. Trade-off:
- More steps (100): Better quality, slower
- Fewer steps (25): Faster, lower quality
- Sweet spot (50): Good balance

**`sigma_shift: 5.0`**

Shifts the noise schedule. This is the trained default - **don't change unless you know what you're doing.**

**`motion_bucket_id: None`**

Controls motion intensity in generated video (50=low, 127=medium, 200=high). For S2V, usually not used (keep `None`) since audio drives the motion naturally.

---

## Why only 1 tokenizer for 5 models?

**Answer:** Only the **text_encoder (T5)** needs a tokenizer. The other models don't process text!

```
text_encoder (T5) → needs tokenizer ← tokenizer_config
dit → takes embeddings (no tokenizer needed)
vae → processes images/latents (no tokenizer needed)
audio_encoder (wav2vec2) → takes audio waveform (no tokenizer needed)
image_encoder (CLIP) → processes images (no tokenizer needed)
```

**Flow:**
```
Text → [tokenizer] → token IDs → [text_encoder] → embeddings → [dit]
Audio → [audio_processor] → normalized audio → [audio_encoder] → embeddings → [dit]
```

Only the T5 text encoder needs text tokenization!

---

## Why separate `audio_processor_config`?

**Answer:** Because `audio_processor` is **NOT a neural network** - it's a preprocessing tool (like tokenizer).

**The split:**
```python
# Neural network weights (in model_configs):
audio_encoder weights (.safetensors)  # The 300M parameter model

# Preprocessing config (separate):
audio_processor config (.json files)  # Normalization mean/std, sample rate
```

**Why separate?**
- `model_configs` = actual model weights to load into GPU
- `audio_processor_config` = just config files (mean=0.0, std=1.0, etc.)
- Different loading mechanisms (torch.load vs json.load)

**Other models DON'T have separate processor configs because:**
- VAE: No preprocessing needed (takes normalized tensors directly)
- Text encoder: Uses `tokenizer_config` (same pattern!)
- Image encoder: Preprocessing hardcoded in pipeline units

---

## Understanding How DiT Processes Text and Audio Conditioning

### How Does DiT "Understand" Text and Audio?

**Short answer:** Text and audio are converted to **embeddings** (numerical vectors) that the DiT transformer can process through attention mechanisms.

**Why This Works (One Sentence Each):**

1. **Text encoder (T5/CLIP):** Pre-trained on millions of text-image pairs, learned to convert words into semantic vectors that capture meaning
2. **Audio encoder (wav2vec2):** Pre-trained on speech data, learned to convert sound into acoustic feature vectors that capture phonemes, pitch, and rhythm
3. **DiT transformer:** Uses **cross-attention** to "look at" text/audio embeddings and generate latents that match their meaning
4. **Training:** DiT was trained on (video, text, audio) triplets, learned to associate embeddings with visual patterns

**Key insight:** DiT doesn't "understand" text/audio like humans do—it learned statistical correlations: "These text embeddings + these audio embeddings → these visual patterns"

**The Conversion Process:**

```python
# HUMAN WORLD:
prompt = "a person talking"           # Text (words)
audio = [0.1, -0.3, 0.5, ...]        # Audio waveform (sound)

# CONVERTED TO EMBEDDINGS (numbers DiT can process):
text_embeds = T5_encoder(prompt)      # [1, 512, 4096] - semantic vectors
audio_embeds = wav2vec2(audio)        # [1, 80, 768] - acoustic vectors

# DiT PROCESSES THEM TOGETHER:
# "Given text_embeds + audio_embeds, denoise latents"
```

**Inside DiT (from `model_fn_wans2v`):**

```python
# Text processing:
context = dit.text_embedding(text_embeds)  # [1, 512, 4096]

# Audio processing:
audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)
# audio_emb_global: [1, 768] - overall audio characteristics
# merged_audio_emb: [1, 80, 768] - frame-by-frame audio features

# Transformer blocks:
for block in dit.blocks:
    # Cross-attention: latents ATTEND to text embeddings
    x = block(x, context, ...)  # "Look at text, understand what to generate"

    # Audio injection: Add audio features to latents
    x = dit.after_transformer_block(..., audio_emb_global, merged_audio_emb, ...)
    # "Align video frames with audio features"
```

---

## Understanding STEP 4: The Denoising Loop

### What is STEP 4 For?

**Purpose:** Transform random noise into a coherent video by **gradually removing noise** over many iterations (typically 50 steps).

### Before STEP 4:

```python
# After STEP 3 (pipeline units), we have:
inputs_shared["latents"] = random_noise  # [1, 16, 81, 30, 52] - Pure random noise
inputs_posi["audio_embeds"] = [1, 80, 768]  # Audio conditioning
inputs_posi["context"] = [1, 512, 4096]     # Text conditioning
inputs_nega["audio_embeds"] = zeros         # No audio (for CFG)

scheduler.timesteps = [999, 979, 959, ..., 19, 0]  # 50 denoising steps
```

**Visual:** Latents = TV static (no structure, pure randomness)

### After STEP 4:

```python
inputs_shared["latents"] = clean_latents  # [1, 16, 81, 30, 52] - Clean structured patterns
# Still in VAE latent space (not RGB yet)
# Represents video content (a person talking)
```

**Visual:** Latents = Structured patterns ready for VAE decoding

### Goal of STEP 4:

Denoise latents from **pure noise → clean representation** over 50 iterations:

```
Iteration 0:  [100% noise, 0% structure]  ████████████████████
Iteration 10: [80% noise, 20% structure]  ████████████████░░░░
Iteration 25: [50% noise, 50% structure]  ██████████░░░░░░░░░░
Iteration 40: [20% noise, 80% structure]  ████░░░░░░░░░░░░░░░░
Iteration 50: [0% noise, 100% clean]      ░░░░░░░░░░░░░░░░░░░░
```

### What is the Scheduler?

**Scheduler = Noise removal strategy**

The scheduler controls **how much noise to remove at each step** through a noise schedule.

**Noise schedule example:**

```python
scheduler.sigmas = [0.98, 0.96, 0.94, ..., 0.02, 0.0]  # Noise levels
# timestep 999: sigma=0.98 (98% noise)
# timestep 500: sigma=0.50 (50% noise)
# timestep 0:   sigma=0.0  (0% noise, clean)
```

### What is `scheduler.step()`?

**Purpose:** Remove predicted noise from current latents to get the next (cleaner) latents.

**The math (simplified):**

```python
# From flow_match.py line 81:
next_latents = current_latents + predicted_noise * (sigma_next - sigma_current)

# Example at iteration 0:
sigma_current = 0.98   # 98% noise at timestep 999
sigma_next = 0.96      # 96% noise at next timestep
delta = 0.96 - 0.98 = -0.02  # Reduce noise by 2%

next_latents = current_latents + noise_pred * (-0.02)
# Move latents opposite to predicted noise direction
# Result: 96% noisy (2% cleaner!)
```

**Key insight:** Each `scheduler.step()` outputs **slightly cleaner latents** by removing a small amount of predicted noise.

### STEP 4 Process (Concrete Example)

**Iteration 0 (timestep=999, very noisy):**

```python
# 1. DiT predicts noise with audio+text conditioning
noise_pred_posi = dit(
    latents=[1, 16, 81, 30, 52],    # Current noisy latents
    audio_embeds=[1, 80, 768],      # ⭐ Audio: "person talking"
    context=[1, 512, 4096],         # Text: "person talking"
    timestep=999                    # "We're at 98% noise level"
)
# DiT thinks: "Given this audio+text, the noise looks like [...]"

# 2. CFG: Blend conditional and unconditional predictions
noise_pred_nega = dit(..., audio_embeds=zeros, ...)  # No audio/text
noise_pred = noise_pred_nega + 5.0 * (noise_pred_posi - noise_pred_nega)
# Push AWAY from unconditional, TOWARD conditional

# 3. Remove predicted noise
latents_new = scheduler.step(noise_pred, timestep=999, latents)
# latents_new is now 96% noisy (cleaner!)

# 4. Lock first frame (if I2V mode)
if "first_frame_latents" in inputs_shared:
    latents_new[:, :, 0:1] = first_frame_latents  # Frame 0 stays fixed

inputs_shared["latents"] = latents_new  # Update for next iteration
```

**After iteration 0:** Latents ≈ 96% noise, 4% structure

**Iteration 25 (halfway, timestep=499):**
- Latents now 50% noisy, DiT makes better predictions
- Remove 2% more noise → 48% noisy

**Iteration 49 (final, timestep=0):**
- Latents now 2% noisy, almost clean
- Remove last 2% noise → 0% noisy (clean!)

### Visual Summary: What STEP 4 Does

```
BEFORE STEP 4:
┌─────────────────────────────────────┐
│  latents = [random noise]           │
│  [[2.1, -0.8, 1.5, ...]]            │  ← TV static
│  Shape: [1, 16, 81, 30, 52]         │
│  Noise level: 100%                  │
└─────────────────────────────────────┘

STEP 4: Denoising Loop (50 iterations)
┌─────────────────────────────────────────────┐
│  FOR EACH TIMESTEP:                         │
│  1. DiT predicts noise (using audio+text)   │
│  2. scheduler.step() removes predicted noise│
│  3. Latents become 2% cleaner               │
│  4. Repeat                                  │
│                                             │
│  Progress:                                  │
│  Step 0:  98% ████████████████████░░        │
│  Step 25: 50% ██████████░░░░░░░░░░░░        │
│  Step 50:  0% ░░░░░░░░░░░░░░░░░░░░░░        │
└─────────────────────────────────────────────┘

AFTER STEP 4:
┌─────────────────────────────────────┐
│  latents = [clean patterns]         │
│  [[0.31, -0.12, 0.79, ...]]         │  ← Structured
│  Shape: [1, 16, 81, 30, 52]         │
│  Noise level: 0%                    │
│                                      │
│  (One tensor containing entire      │
│   video, ready for VAE decode)      │
└─────────────────────────────────────┘
```

**Important clarification:**
- **ONE latent tensor** `[1, 16, 81, 30, 52]` represents the **entire 81-frame video**
- This single tensor will be decoded into 81 individual RGB frames by the VAE

---

## Understanding STEP 5: Post Units

### What Are Post Units?

**Post units** are **optional processing steps** that modify the clean latents **after denoising** but **before VAE decoding**.

```
STEP 4 → Clean latents [1, 16, 81, 30, 52]
  ↓
STEP 5 → Post-process latents (optional modifications)
  ↓
STEP 6 → VAE decode → RGB video
```

### Current Post Units in WanVideoPipeline:

```python
# Line 171-173 in commented_wan_video_new.py:
self.post_units = [
    WanVideoPostUnit_S2V(),  # Only one post unit
]
```

### What Does `WanVideoPostUnit_S2V` Do?

**Purpose:** Adds **motion context** for multi-clip long video generation.

**Problem:** DiT can only generate 81 frames (5 seconds @ 16fps) at a time. How to generate 10+ seconds?

**Solution:** Generate multiple clips, use motion context from previous clip for smooth transitions.

### Example: Generating a 10-Second S2V Video

```
LONG VIDEO GENERATION (10 seconds = 160 frames)

┌────────────────────────────────────────────────────────────┐
│ CLIP 1 (Frames 0-80)                                       │
│   - Generate 81 frames from scratch                        │
│   - No motion context (first clip)                         │
│   - AFTER STEP 4: latents [1, 16, 81, 30, 52]             │
│   - STEP 5: Skip (drop_motion_frames=True)                 │
│   - STEP 6: Decode → Frames 0-80                           │
└────────────────────────────────────────────────────────────┘
              │
              ├─ Save last 73 frames as motion_latents
              ▼
┌────────────────────────────────────────────────────────────┐
│ CLIP 2 (Frames 81-160)                                     │
│   - Generate 81 new frames                                 │
│   - Use last 73 frames from Clip 1 as motion context       │
│   - AFTER STEP 4: new_latents [1, 16, 81, 30, 52]         │
│   - STEP 5: Prepend motion context ⭐                       │
│       latents = [motion_latents (73), new_latents[1:] (80)]│
│       Result: [1, 16, 153, 30, 52]                         │
│   - STEP 6: Decode → Take last 80 frames → Frames 81-160  │
└────────────────────────────────────────────────────────────┘
```

**Code example (CLIP 2 post-processing):**

```python
# AFTER STEP 4:
latents_clip2 = [1, 16, 81, 30, 52]  # Newly generated 81 frames

# STEP 5 (WanVideoPostUnit_S2V, line 1500):
drop_motion_frames = False  # Use motion context
latents_with_motion = torch.cat([
    motion_latents,         # [1, 16, 73, 30, 52] - from Clip 1
    latents_clip2[:,:,1:]   # [1, 16, 80, 30, 52] - skip first frame
], dim=2)
# Result: [1, 16, 153, 30, 52]

# Why skip first frame?
# Avoid duplication: last motion frame overlaps with first new frame
# Creates smooth temporal transition
```

### Summary: STEP 5

**Q: What is STEP 5?**
- Optional post-processing of clean latents before VAE decoding
- Only one post unit: `WanVideoPostUnit_S2V`
- Purpose: Add motion context for multi-clip generation (smooth transitions)
- For single-clip inference: Does nothing (skipped)

**Q: What are post units?**
- Pipeline units that run AFTER denoising (between STEP 4 and STEP 6)
- Modify clean latents before decoding
- Currently only used for multi-clip S2V motion context

---

## Understanding `WanVideoUnit_S2V` Integration and `take_over=True`

### Summary: Key Concepts

**Q: What does `take_over=True` mean?**
- Unit receives ALL 3 dicts as parameters: `inputs_shared`, `inputs_posi`, `inputs_nega`
- Can read and modify ALL 3 dicts
- Must return ALL 3 dicts

**Q: What does `take_over=False` mean (default)?**
- Unit only receives parameters extracted from ONE dict (usually `inputs_shared`)
- Can only modify that ONE dict
- Returns a simple dict that gets merged back

**Q: Why does S2V need `take_over=True`?**
- Must extract `input_audio` from `inputs_shared` (and remove it)
- Must add `audio_embeds` to `inputs_posi` (positive conditioning for CFG)
- Must add zeroed `audio_embeds` to `inputs_nega` (negative conditioning for CFG)
- Can't do this with `take_over=False` (would only see one dict)

---

### What Are `inputs_posi` and `inputs_nega`?

**They're conditioning dictionaries for Classifier-Free Guidance (CFG).**

**CFG Formula:**
```python
final_output = negative_output + cfg_scale * (positive_output - negative_output)
#              ↑                               ↑
#        Unconditional                    Conditional
#     (no guidance)                    (with audio/text)
```

**The 3 Dictionaries:**

```python
# inputs_posi: Positive conditioning (what we WANT)
inputs_posi = {
    "prompt_embeds": encode("a person talking"),  # Text guidance
    "audio_embeds": wav2vec2(audio)               # Audio guidance ⭐
}
# DiT generates video that MATCHES audio+text

# inputs_nega: Negative conditioning (unconditional baseline)
inputs_nega = {
    "prompt_embeds": encode("blurry, distorted"),  # Anti-guidance
    "audio_embeds": zeros                           # NO audio ⭐
}
# DiT generates baseline video (no audio guidance)

# inputs_shared: CFG-insensitive (same for both passes)
inputs_shared = {
    "latents": noise,  # Noisy latents being denoised
    "height": 480,
    "width": 832,
    # ... other params that don't change between positive/negative
}
```

**How CFG Uses Them (in denoising loop):**

```python
for timestep in scheduler.timesteps:
    # POSITIVE PASS: DiT with audio+text guidance
    noise_pred_posi = dit(**inputs_shared, **inputs_posi, timestep=timestep)
    # Result: "Video with this specific audio"

    # NEGATIVE PASS: DiT WITHOUT audio guidance
    noise_pred_nega = dit(**inputs_shared, **inputs_nega, timestep=timestep)
    # Result: "Generic video (no audio)"

    # CFG: Blend predictions (push TOWARD conditional, AWAY from unconditional)
    noise_pred = noise_pred_nega + 5.0 * (noise_pred_posi - noise_pred_nega)

    # Remove predicted noise
    latents = scheduler.step(noise_pred, timestep, latents)
```

**Why S2V Must Modify Both Dicts:**
- `inputs_posi["audio_embeds"]`: Real audio embeddings → guide generation toward audio
- `inputs_nega["audio_embeds"]`: Zeros → unconditional baseline
- CFG blends them → final video strongly matches audio

---

### Comparing `take_over=True` vs `take_over=False`

**Example: `take_over=False` (ShapeChecker)**

```python
class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=False,  # Default mode
            input_params=["height", "width", "num_frames"]
        )

    def process(self, pipe, height, width, num_frames):
        # Only receives 3 params (from inputs_shared)
        # CAN'T see inputs_posi or inputs_nega!

        height = (height // 16) * 16  # Validate
        return {"height": height, "width": width, "num_frames": num_frames}

# unit_runner extracts params from inputs_shared only:
processor_inputs = {"height": 480, "width": 832, "num_frames": 81}
processor_outputs = unit.process(pipe, **processor_inputs)
inputs_shared.update(processor_outputs)  # Only updates inputs_shared
```

**Example: `take_over=True` (S2V)**

```python
class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)  # Full control!

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        # Receives ALL 3 dicts!

        # 1. Extract audio from inputs_shared (remove it)
        input_audio = inputs_shared.pop("input_audio")

        # 2. Process audio → embeddings
        audio_embeds = self.process_audio(pipe, input_audio, ...)

        # 3. Add to inputs_posi (positive conditioning)
        inputs_posi["audio_embeds"] = audio_embeds["audio_embeds"]

        # 4. Add to inputs_nega (negative conditioning)
        inputs_nega["audio_embeds"] = 0.0 * audio_embeds["audio_embeds"]

        # 5. Return ALL 3 (modified)
        return inputs_shared, inputs_posi, inputs_nega

# unit_runner gives full control:
inputs_shared, inputs_posi, inputs_nega = unit.process(
    pipe, inputs_shared, inputs_posi, inputs_nega
)
```

---

### How `WanVideoUnit_S2V` Is Integrated Into Pipeline

**Step 1: Unit is registered in `__init__`**

```python
class WanVideoPipeline(BasePipeline):
    def __init__(self):
        self.units = [
            WanVideoUnit_ShapeChecker(),         # 1
            WanVideoUnit_NoiseInitializer(),     # 2
            WanVideoUnit_PromptEmbedder(),       # 3
            WanVideoUnit_S2V(),                  # 4 ⭐ Registered here
            WanVideoUnit_ImageEmbedderVAE(),     # 5
            # ... 15 more units
        ]
```

**Step 2: Unit is executed in `__call__()`**

```python
def __call__(self, prompt, input_audio, ...):
    # Create 3 dicts
    inputs_shared = {"input_audio": input_audio, "height": 480, ...}
    inputs_posi = {"prompt": "person talking"}
    inputs_nega = {"negative_prompt": "blurry"}

    # Run all units sequentially
    for unit in self.units:  # Iterates through all 20 units
        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
            unit, self, inputs_shared, inputs_posi, inputs_nega
        )
```

**Step 3: `unit_runner` detects `take_over=True`**

```python
def __call__(self, unit, pipe, inputs_shared, inputs_posi, inputs_nega):
    if unit.take_over:  # ← S2V has this!
        # Give unit full control of all 3 dicts
        inputs_shared, inputs_posi, inputs_nega = unit.process(
            pipe, inputs_shared, inputs_posi, inputs_nega
        )

    return inputs_shared, inputs_posi, inputs_nega
```

**Step 4: S2V's `process()` executes**

```python
def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
    # Extract audio
    input_audio = inputs_shared.pop("input_audio")

    # Convert to embeddings
    audio_embeds = self.process_audio(pipe, input_audio, ...)
    # → {"audio_embeds": [1, 80, 768]}

    # Add to positive conditioning
    inputs_posi["audio_embeds"] = audio_embeds["audio_embeds"]

    # Add to negative conditioning (zeros)
    inputs_nega["audio_embeds"] = 0.0 * audio_embeds["audio_embeds"]

    return inputs_shared, inputs_posi, inputs_nega
```

**Step 5: Updated dicts flow to next units**

```python
# After S2V completes:
inputs_shared = {"height": 480, ...}  # NO input_audio (removed)
inputs_posi = {"prompt_embeds": ..., "audio_embeds": [1, 80, 768]}  # ⭐
inputs_nega = {"negative_prompt_embeds": ..., "audio_embeds": zeros}  # ⭐

# Continue through remaining 16 units...
```

**Step 6: Dicts used in denoising loop**

```python
for timestep in scheduler.timesteps:
    noise_pred_posi = self.model_fn(**inputs_shared, **inputs_posi, timestep=timestep)
    noise_pred_nega = self.model_fn(**inputs_shared, **inputs_nega, timestep=timestep)
    # CFG blending...
```

---

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ WanVideoPipeline.__init__()                             │
│   self.units = [                                        │
│       ...,                                              │
│       WanVideoUnit_S2V(),  ← Unit created and stored   │
│       ...,                                              │
│   ]                                                     │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ WanVideoPipeline.__call__(input_audio=audio, ...)      │
│                                                         │
│   inputs_shared = {"input_audio": audio, ...}          │
│   inputs_posi = {"prompt": "..."}                      │
│   inputs_nega = {"negative_prompt": "..."}             │
│                                                         │
│   for unit in self.units:  ← Loop through all units    │
│       ↓                                                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ When unit = WanVideoUnit_S2V:                           │
│                                                         │
│   unit_runner(unit, ...) checks:                       │
│   if unit.take_over == True:  ← S2V has this!          │
│       ↓                                                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Call unit.process() with ALL 3 dicts:                  │
│                                                         │
│   inputs_shared, inputs_posi, inputs_nega =            │
│       unit.process(pipe, inputs_shared,                │
│                    inputs_posi, inputs_nega)           │
│       ↓                                                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Inside WanVideoUnit_S2V.process():                     │
│                                                         │
│   1. input_audio = inputs_shared.pop("input_audio")    │
│   2. audio_embeds = wav2vec2(input_audio)              │
│   3. inputs_posi["audio_embeds"] = audio_embeds        │
│   4. inputs_nega["audio_embeds"] = zeros               │
│   5. return inputs_shared, inputs_posi, inputs_nega    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Back to __call__(), continue with next unit:           │
│                                                         │
│   Updated dicts flow to remaining 16 units...          │
│   Then used in denoising loop (STEP 4)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Where Does `get_audio_feats_per_inference` Come From?

**Location:** `diffsynth/models/wav2vec.py` (line 186)

**It's a method defined in the audio encoder model class:**

```python
# In wav2vec.py (simplified):
class AudioEncoder:
    def get_audio_feats_per_inference(self, input_audio, sample_rate, processor, ...):
        # 1. Extract raw audio features from wav2vec2
        audio_feat = self.extract_audio_feat(input_audio, ...)

        # 2. Bucket audio features to match video FPS
        audio_embed_bucket = self.get_audio_embed_bucket_fps(audio_feat, fps=16, ...)

        # 3. Split into clips (for multi-clip generation)
        audio_embeds = [audio_embed_bucket[..., i*80:(i+1)*80] for i in range(num_clips)]

        return audio_embeds
```

**How it's used:**

```python
# In process_audio() line 1238:
audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(...)
#              ↑                  ↑
#          pipe.audio_encoder   Method defined in AudioEncoder class
#          (loaded from HF)
```

**Where's the model from?**
- Downloaded from HuggingFace: `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors`
- Loaded into `pipe.audio_encoder` during `from_pretrained()`
- The model class is `AudioEncoder` from `diffsynth/models/wav2vec.py`
- `get_audio_feats_per_inference()` is a **custom method** written by DiffSynth authors, wrapping wav2vec2
