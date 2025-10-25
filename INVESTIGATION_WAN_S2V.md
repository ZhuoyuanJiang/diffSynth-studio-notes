# Investigation: Fine-tuning Wan S2V on RenderMe360 Dataset

**Author**: Zhuoyuan
**Date**: 2025-10-02
**Purpose**: Document the investigation process for understanding how to fine-tune Wan2.2-S2V-14B using DiffSynth-Studio

---

## 🚀 Training Flow: Entry Point to Execution

### The Complete Execution Flow

```
1. USER RUNS SHELL SCRIPT
   ↓
   bash examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh

2. SHELL SCRIPT EXECUTES ACCELERATE
   ↓
   accelerate launch --config_file accelerate_config_14B.yaml \
     examples/wanvideo/model_training/train.py \
     --dataset_base_path ... \
     --extra_inputs "input_image,input_audio,s2v_pose_video" \
     ... (more args)

3. PYTHON ENTRY POINT
   ↓
   examples/wanvideo/model_training/train.py:92
   if __name__ == "__main__":

4. PARSE ARGUMENTS
   ↓
   diffsynth/trainers/utils.py:594
   wan_parser() → defines all CLI arguments

5. CREATE DATASET
   ↓
   train.py:95-114
   UnifiedDataset(
     metadata_path="metadata.csv",
     data_file_keys=["video", "audio", "prompt"],
     ...
   )
   → Reads CSV, loads data via operators

6. CREATE TRAINING MODULE
   ↓
   train.py:115-127
   WanTrainingModule(
     model_id_with_origin_paths=...,
     extra_inputs=args.extra_inputs,  # ← "input_audio,s2v_pose_video"
     lora_base_model="dit",
     ...
   )
   → Loads pipeline, switches to training mode

7. START TRAINING LOOP
   ↓
   diffsynth/trainers/utils.py:521
   launch_training_task(dataset, model, model_logger, args)
   → Accelerate training loop begins

8. TRAINING ITERATION (per batch)
   ↓
   a. DataLoader yields batch from dataset
      - Loads: data["video"], data["audio"], data["prompt"]

   b. model.forward(data) called
      ↓
      train.py:42-82
      forward_preprocess(data):
        - Creates inputs_shared dict
        - Adds extra_inputs from data to inputs_shared
        - inputs_shared["input_audio"] = data["audio"]
        - inputs_shared["s2v_pose_video"] = data["s2v_pose_video"]

   c. Pipeline units process inputs
      ↓
      train.py:79-81
      for unit in self.pipe.units:
        inputs_shared, inputs_posi, inputs_nega = unit.process(...)

      - WanVideoUnit_S2V sees input_audio
      ↓
      diffsynth/pipelines/wan_video_new.py:1027-1040
      WanVideoUnit_S2V.process():
        - Extracts audio from inputs_shared
        - Calls process_audio() → audio_embeds
        - Adds audio_embeds to inputs_posi

   d. Compute loss
      ↓
      train.py:85-89
      loss = self.pipe.training_loss(**models, **inputs)
      - DiT forward with audio conditioning
      - Diffusion loss computed

   e. Backprop & optimizer step
      ↓
      Accelerate handles: loss.backward(), optimizer.step()

9. SAVE CHECKPOINT
   ↓
   Periodically saves LoRA weights to output_path

10. TRAINING COMPLETE
```

### Key Files in Execution Order

| Step | File | Lines | Purpose |
|------|------|-------|---------|
| **1. Entry** | `examples/wanvideo/model_training/lora/*.sh` | All | Shell command with all arguments |
| **2. Main** | `examples/wanvideo/model_training/train.py` | 92-132 | Python entry point, setup |
| **3. Parser** | `diffsynth/trainers/utils.py` | 594+ | Argument definitions |
| **4. Dataset** | `diffsynth/trainers/unified_dataset.py` | 230-249 | Load data from CSV |
| **5. Training Module** | `examples/wanvideo/model_training/train.py` | 10-89 | Model wrapper with forward() |
| **6. Pipeline Units** | `diffsynth/pipelines/wan_video_new.py` | 972-1052 | S2V audio processing |
| **7. Training Loop** | `diffsynth/trainers/utils.py` | 521+ | Accelerate training loop |

### Data Flow Diagram

```
CSV File (metadata.csv)
  ├─ video: "path/to/clip.mp4"
  ├─ audio: "path/to/audio.mp3"
  ├─ prompt: "A person speaking"
  └─ s2v_pose_video: "path/to/pose.mp4"
        ↓
UnifiedDataset.__getitem__(idx)
  ├─ Loads video → List[PIL.Image] (81 frames)
  ├─ Loads audio → waveform array
  ├─ Loads prompt → str
  └─ Loads pose video → List[PIL.Image]
        ↓
Returns: data = {
  "video": [img1, img2, ...],  # 81 frames
  "audio": audio_waveform,      # numpy array
  "prompt": "A person speaking",
  "s2v_pose_video": [pose1, pose2, ...]
}
        ↓
WanTrainingModule.forward(data)
        ↓
forward_preprocess(data)
  ├─ inputs_posi = {"prompt": data["prompt"]}
  ├─ inputs_shared = {"input_video": data["video"], ...}
  └─ Loop over extra_inputs:
      ├─ inputs_shared["input_image"] = data["video"][0]
      ├─ inputs_shared["input_audio"] = data["audio"]
      └─ inputs_shared["s2v_pose_video"] = data["s2v_pose_video"]
        ↓
Pipeline Units Process
  ├─ WanVideoUnit_PromptEmbedder
  │   └─ Encodes prompt with T5
  ├─ WanVideoUnit_S2V  ← KEY!
  │   ├─ Sees input_audio in inputs_shared
  │   ├─ Loads audio_encoder
  │   ├─ audio_embeds = audio_encoder(input_audio)
  │   ├─ inputs_posi["audio_embeds"] = audio_embeds
  │   └─ inputs_nega["audio_embeds"] = 0.0 * audio_embeds
  ├─ WanVideoUnit_InputVideoEmbedder
  │   └─ Encodes video with VAE
  └─ ... other units
        ↓
inputs = {**inputs_shared, **inputs_posi}
  = {
      "latents": video_latents,
      "prompt_emb": text_embeddings,
      "audio_embeds": audio_embeddings,  ← From S2V unit
      "s2v_pose_latents": pose_latents,
      ...
    }
        ↓
self.pipe.training_loss(**models, **inputs)
  ├─ DiT forward pass with audio conditioning
  ├─ Add noise to latents
  ├─ Predict noise with DiT(latents, audio_embeds, ...)
  └─ Compute MSE loss
        ↓
loss.backward()
        ↓
optimizer.step()
        ↓
Next batch...
```

### The Critical `--extra_inputs` Mechanism

**How it works**:
1. **Shell script** passes: `--extra_inputs "input_image,input_audio,s2v_pose_video"`
2. **Argument parser** stores: `args.extra_inputs = "input_image,input_audio,s2v_pose_video"`
3. **Training module** splits: `self.extra_inputs = ["input_image", "input_audio", "s2v_pose_video"]`
4. **Dataset** loads these columns from CSV into `data` dict
5. **forward_preprocess()** loops over `self.extra_inputs`:
   ```python
   for extra_input in self.extra_inputs:  # ["input_image", "input_audio", ...]
       if extra_input == "input_image":
           inputs_shared["input_image"] = data["video"][0]
       else:
           inputs_shared[extra_input] = data[extra_input]
   ```
6. **Pipeline units** check `inputs_shared` for their required inputs
7. **S2V unit** sees `input_audio` → processes it → adds embeddings to flow

**This is why no code modification needed!** The framework is designed to generically pass any data through the pipeline via `--extra_inputs`.

---

## 📚 How I Got These Solutions - Learning Path

This document outlines the **step-by-step investigation process** used to understand the DiffSynth-Studio codebase and develop a training strategy for Wan S2V on the RenderMe360 dataset.

---

## Phase 1: Understanding the Model (30 min)

### Script 1: Inference Example
**File**: `examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py`

**What I learned**:
- **Line 9-23**: Function signature shows required inputs:
  - `prompt` (text description)
  - `input_image` (reference image)
  - `audio_path` (speech audio)
  - `pose_video_path` (optional pose guidance)
  - `num_frames` (number of frames to generate)

- **Line 26**: Audio processing requirements:
  ```python
  input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
  ```
  Audio needs 16kHz sample rate

- **Line 30-39**: Audio preprocessing:
  ```python
  audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
      pipe=pipe,
      input_audio=input_audio,
      audio_sample_rate=sample_rate,
      s2v_pose_video=pose_video,
      num_frames=infer_frames + 1,
      ...
  )
  ```
  Audio is converted to embeddings before inference

- **Line 46-58**: Video generation loop:
  - Generates video in clips
  - Each clip uses corresponding audio segment
  - Motion context from previous frames

- **Line 70-79**: Model components required:
  ```python
  ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),  # DiT
  ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),  # T5 Text Encoder
  ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),  # Audio Encoder
  ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),  # VAE
  ```

- **Line 88**: Frame constraint:
  ```python
  infer_frames = 80  # 4n, because num_frames = infer_frames + 1 = 81 = 4*20 + 1
  ```
  Must satisfy: `num_frames % 4 == 1`

**Key insight**: S2V takes audio + reference image → generates speaking video synchronized with audio

---

## Phase 2: Pipeline Architecture (45 min)

### Script 2: Pipeline Implementation
**File**: `diffsynth/pipelines/wan_video_new.py`

**What I learned**:

#### Overall Architecture (Lines 32-78)
```python
class WanVideoPipeline(BasePipeline):
    def __init__(self, ...):
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),  # ← S2V-specific unit
            WanVideoUnit_InputVideoEmbedder(),
            ...
        ]
```
- Pipeline composed of "units" that process inputs sequentially
- Each unit adds specific functionality
- Units communicate via shared dictionaries: `inputs_shared`, `inputs_posi`, `inputs_nega`

#### S2V Unit Implementation (Lines 972-1052) - **CRITICAL**

**Class Definition (Line 972-977)**:
```python
class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )
```
This unit loads audio encoder and VAE when needed

**Audio Processing (Lines 979-987)**:
```python
def process_audio(self, pipe, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
    if audio_embeds is not None:
        return {"audio_embeds": audio_embeds}
    pipe.load_models_to_device(["audio_encoder"])
    audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
        input_audio, audio_sample_rate, pipe.audio_processor,
        fps=fps, batch_frames=num_frames-1, ...
    )
```
Converts audio waveform to embeddings using wav2vec2 audio encoder

**Motion Latents Processing (Lines 989-1002)**:
```python
def process_motion_latents(self, pipe, height, width, tiled, tile_size, tile_stride, motion_video=None):
    motion_frames = 73  # Fixed context size
    if motion_video is not None and len(motion_video) > 0:
        motion_latents = pipe.preprocess_video(motion_video)
        kwargs["drop_motion_frames"] = False
    else:
        motion_latents = torch.zeros([1, 3, motion_frames, height, width], ...)
        kwargs["drop_motion_frames"] = True
```
Uses 73 frames of motion context for temporal consistency

**Pose Conditioning (Lines 1004-1025)**:
```python
def process_pose_cond(self, pipe, s2v_pose_video, num_frames, height, width, ...):
    if s2v_pose_video is None:
        return {"s2v_pose_latents": None}
    # Extract pose features from video
    input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
    # Encode with VAE
    cond_latents = pipe.vae.encode(cond, ...)
```
Optional pose guidance from video (can be None)

**Main Processing (Lines 1027-1040)**:
```python
def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
    if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or ...:
        return inputs_shared, inputs_posi, inputs_nega

    # Extract inputs
    input_audio = inputs_shared.pop("input_audio")
    s2v_pose_video = inputs_shared.pop("s2v_pose_video")

    # Process audio (positive)
    audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames)
    inputs_posi.update(audio_input_positive)

    # Process audio (negative - zero embeddings for CFG)
    inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

    # Process motion and pose
    inputs_shared.update(self.process_motion_latents(...))
    inputs_shared.update(self.process_pose_cond(...))
```

**Key insight**: Pipeline uses modular "units" - S2V unit extracts audio embeddings and pose features, adds them to the processing flow

---

## Phase 3: Training Framework (1 hour)

### Script 3: Training Script
**File**: `examples/wanvideo/model_training/train.py`

**What I learned**:

#### Training Module Structure (Lines 10-40)
```python
class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        use_gradient_checkpointing=True,
        extra_inputs=None,
        ...
    ):
        # Load models
        model_configs = self.parse_model_configs(...)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu",
            model_configs=model_configs
        )

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, ...
        )

        # Store configs
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
```

Key observations:
- Inherits from `DiffusionTrainingModule`
- Loads same pipeline as inference
- `switch_pipe_to_training_mode()` enables training (LoRA or full)
- **`extra_inputs` stored as list** - this is crucial!

#### Forward Preprocessing (Lines 42-82) - **KEY METHOD**
```python
def forward_preprocess(self, data):
    # CFG-sensitive parameters (affected by classifier-free guidance)
    inputs_posi = {"prompt": data["prompt"]}
    inputs_nega = {}

    # CFG-unsensitive parameters (shared between positive/negative)
    inputs_shared = {
        "input_video": data["video"],
        "height": data["video"][0].size[1],
        "width": data["video"][0].size[0],
        "num_frames": len(data["video"]),
        "cfg_scale": 1,
        "tiled": False,
        ...
    }

    # Extra inputs - THIS IS THE KEY!
    for extra_input in self.extra_inputs:
        if extra_input == "input_image":
            inputs_shared["input_image"] = data["video"][0]
        elif extra_input == "end_image":
            inputs_shared["end_image"] = data["video"][-1]
        elif extra_input == "reference_image" or extra_input == "vace_reference_image":
            inputs_shared[extra_input] = data[extra_input][0]
        else:
            inputs_shared[extra_input] = data[extra_input]

    # Pipeline units process the inputs
    for unit in self.pipe.units:
        inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
            unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
        )
    return {**inputs_shared, **inputs_posi}
```

**Critical understanding**:
1. `data` comes from dataset (CSV columns)
2. `extra_inputs` specifies which additional columns to load
3. Extra inputs added to `inputs_shared` dictionary
4. Pipeline units (including S2V unit) process these inputs
5. If `input_audio` in `inputs_shared`, S2V unit will process it!

#### Training Loop Setup (Lines 92-132)
```python
if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # Dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            ...
        ),
    )

    # Training module
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        extra_inputs=args.extra_inputs,  # ← Passed from command line
        ...
    )
```

**Key insight**: Training uses same pipeline as inference, but:
1. Switches models to `.train()` mode
2. Loads data from CSV via `UnifiedDataset`
3. Uses `extra_inputs` to pass additional data to pipeline
4. Computes loss instead of generating images

---

## Phase 4: Dataset Format (30 min)

### Script 4: Dataset Implementation
**File**: `diffsynth/trainers/unified_dataset.py`

**What I learned**:

#### UnifiedDataset Class (Lines 230-249)
```python
class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,              # Base directory for data files
        metadata_path=None,           # Path to CSV file
        repeat=1,                     # Repeat dataset N times per epoch
        data_file_keys=tuple(),       # Column names from CSV
        main_data_operator=lambda x: x,  # Processing pipeline for main data
        special_operator_map=None,    # Custom operators for specific columns
    ):
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.load_metadata(metadata_path)
```

**How it works**:
1. Reads CSV with columns matching `data_file_keys`
2. Each column processed by an operator (function/pipeline)
3. Main data (usually video) uses `main_data_operator`
4. Special columns use `special_operator_map`

#### Data Operators

**LoadImage (Lines 62-69)**:
```python
class LoadImage(DataProcessingOperator):
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image
```

**LoadVideo (Lines 117-143)**:
```python
class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, ...):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            # Ensure divisibility constraint
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
```
**Critical**: For Wan videos, must satisfy `num_frames % 4 == 1` (e.g., 81, 85, 89)

**ToAbsolutePath (Lines 221-226)**:
```python
class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        return os.path.join(self.base_path, data)
```
Prepends `base_path` to relative paths in CSV

#### Example Usage Pattern
```python
dataset = UnifiedDataset(
    base_path="/path/to/data",
    metadata_path="/path/to/metadata.csv",
    data_file_keys=["video", "audio", "prompt"],  # CSV columns
    main_data_operator=LoadVideo(num_frames=81),  # For "video" column
    special_operator_map={
        "audio": ToAbsolutePath("/path/to/data") >> LoadAudio(),
        "prompt": ToStr(),
    }
)
```

**Key insight**: Dataset expects CSV with columns matching `data_file_keys`, applies operator pipeline to load/process each column

---

## Phase 5: LoRA Training Example (20 min)

### Script 5: Training Configuration
**File**: `examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh`

**What I learned**:

```bash
# Line 1-2: GPU requirements
# 1*80G GPU cannot train Wan2.2-Animate-14B LoRA
# We tested on 8*80G GPUs

# Line 3: Distributed training setup
accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \

  # Lines 4-7: Dataset configuration
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata_animate.csv \
  --data_file_keys "video,animate_pose_video,animate_face_video" \

  # Lines 8-10: Video specifications
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \

  # Line 12: Model loading (colon-separated model_id:file_pattern pairs)
  --model_id_with_origin_paths "Wan-AI/Wan2.2-Animate-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-Animate-14B:models_t5_umt5-xxl-enc-bf16.pth,..." \

  # Lines 13-15: Training hyperparameters
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-Animate-14B_lora" \

  # Lines 16-18: LoRA configuration
  --lora_base_model "dit" \  # Only train DiT with LoRA
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \  # Attention Q,K,V,O + FFN layers
  --lora_rank 32 \

  # Line 19: THE KEY - Additional inputs!
  --extra_inputs "input_image,animate_pose_video,animate_face_video" \

  # Line 20: Memory optimization
  --use_gradient_checkpointing_offload
```

**Key insights**:
1. **`--extra_inputs`** is how we pass additional data to pipeline!
   - Wan2.2-Animate uses: `input_image,animate_pose_video,animate_face_video`
   - For S2V, we need: `input_image,input_audio,audio_sample_rate,s2v_pose_video`

2. **LoRA configuration**:
   - `--lora_base_model "dit"` → Only DiT is trainable with LoRA
   - `--lora_target_modules` → Which layers get LoRA adapters
   - Results in ~200MB checkpoint vs full 14B model

3. **GPU requirements**:
   - 8×80GB GPUs for 14B model LoRA training
   - Gradient checkpointing offload for memory efficiency

---

## Phase 6: Connecting to S2V (Critical Realization)

### The "Aha!" Moment

**Going back to S2V Unit (Line 1027-1040)**:
```python
def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
    # Line 1028: Check if audio inputs exist
    if (inputs_shared.get("input_audio") is None and
        inputs_shared.get("audio_embeds") is None) or
        pipe.audio_encoder is None:
        return inputs_shared, inputs_posi, inputs_nega  # Skip if no audio

    # Line 1031: Extract S2V-specific inputs from inputs_shared
    input_audio = inputs_shared.pop("input_audio")
    audio_embeds = inputs_shared.pop("audio_embeds")
    s2v_pose_video = inputs_shared.pop("s2v_pose_video")

    # Line 1034: Process audio
    audio_input_positive = self.process_audio(pipe, input_audio, ...)
    inputs_posi.update(audio_input_positive)

    # Line 1036: Negative (CFG)
    inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})
```

**Connection to Training Script (Line 69-77)**:
```python
# In WanTrainingModule.forward_preprocess():
for extra_input in self.extra_inputs:
    if extra_input == "input_image":
        inputs_shared["input_image"] = data["video"][0]
    elif extra_input == "end_image":
        inputs_shared["end_image"] = data["video"][-1]
    else:
        inputs_shared[extra_input] = data[extra_input]  # ← Generic passthrough!
```

**THE REALIZATION**:

If we pass `--extra_inputs "input_image,input_audio,s2v_pose_video"`:

1. **Training script** loads these columns from CSV (`data["input_audio"]`, `data["s2v_pose_video"]`)
2. **Adds them to `inputs_shared`** dict via the `else` clause
3. **Pipeline processes inputs** through all units in sequence
4. **S2V unit** sees `input_audio` in `inputs_shared`
5. **S2V unit processes audio** → extracts embeddings → adds to `inputs_posi`
6. **DiT receives audio embeddings** during training forward pass
7. **Loss computed** on audio-conditioned generation

**This is exactly how Wan2.2-Animate works!**
- Animate: `--extra_inputs "input_image,animate_pose_video,animate_face_video"`
- S2V: `--extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video"`

Same mechanism, different inputs!

---

## The Complete Solution Path

### Investigation Flow:
1. ✅ **Started with inference example** → Learned what inputs S2V needs
2. ✅ **Traced pipeline implementation** → Understood how S2V unit processes audio
3. ✅ **Studied training script** → Found `extra_inputs` mechanism
4. ✅ **Analyzed dataset format** → Learned CSV structure requirements
5. ✅ **Examined LoRA example** → Saw how Animate uses `extra_inputs`
6. ✅ **Connected the dots** → Realized S2V training = Wan training + audio in `extra_inputs`

### The Solution:
**S2V training requires**:
1. CSV with columns: `video,audio,prompt,input_image,s2v_pose_video`
2. Training flag: `--extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video"`
3. Load S2V model components: DiT, T5, wav2vec2, VAE
4. Same training script as other Wan models!

**No custom training script needed** - the existing framework handles everything through:
- `extra_inputs` for data passing
- Pipeline units for processing
- Modular architecture

---

## Key Scripts to Study (Recommended Order)

### For Understanding (1-2 hours):

1. **`examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py`** (10 min)
   - **Purpose**: Understand S2V inputs/outputs
   - **Key lines**: 9-23 (function signature), 26 (audio loading), 70-79 (model components)
   - **Takeaway**: S2V needs audio (16kHz) + reference image → speaking video

2. **`diffsynth/pipelines/wan_video_new.py`** lines 972-1052 (20 min)
   - **Purpose**: How S2V processes audio during inference/training
   - **Key sections**:
     - 979-987: Audio encoding
     - 989-1002: Motion latents
     - 1004-1025: Pose conditioning
     - 1027-1040: Main processing logic
   - **Takeaway**: S2V unit extracts audio embeddings and adds to pipeline flow

3. **`examples/wanvideo/model_training/train.py`** (30 min)
   - **Purpose**: Training framework architecture
   - **Key sections**:
     - 10-40: Training module initialization
     - 42-82: Forward preprocessing (**critical!**)
     - 92-132: Training loop setup
   - **Takeaway**: `extra_inputs` mechanism passes data from CSV to pipeline

### For Implementation (2-3 hours):

4. **`diffsynth/trainers/unified_dataset.py`** (30 min)
   - **Purpose**: Dataset format requirements
   - **Key sections**:
     - 230-249: UnifiedDataset class
     - 62-69: LoadImage operator
     - 117-143: LoadVideo operator (note frame constraints)
     - 221-226: ToAbsolutePath operator
   - **Takeaway**: CSV → Operators → Processed data dictionary

5. **`examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh`** (10 min)
   - **Purpose**: LoRA training configuration
   - **Key lines**:
     - 3: Accelerate config
     - 4-7: Dataset paths and keys
     - 16-18: LoRA configuration
     - 19: `--extra_inputs` flag (**THE KEY!**)
   - **Takeaway**: How to configure training for additional inputs

6. **`TRAINING_PLAN_WAN_S2V_RENDERME360.md`** (Your implementation plan)
   - **Purpose**: Specific steps for RenderMe360 dataset
   - **Covers**: Data preprocessing, training config, evaluation

### Quick Reference Cheat Sheet:

```bash
# What S2V needs:
Input: audio (16kHz) + reference image + prompt + (optional) pose video
Output: Speaking video synchronized with audio

# How to train:
1. Create CSV: video,audio,prompt,input_image,s2v_pose_video
2. Add flag: --extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video"
3. Load models: DiT + T5 + wav2vec2 + VAE
4. Run training with existing train.py script

# Key constraint:
num_frames % 4 == 1  (e.g., 81, 85, 89 frames)
```

---

## Critical Insights Summary

### 1. Architecture Insight
> "DiffSynth uses a modular pipeline system where 'units' process inputs sequentially. Each unit is responsible for a specific task (text encoding, audio encoding, etc.). The S2V unit specifically handles audio encoding (via wav2vec2) and optional pose conditioning. Training reuses the same pipeline but switches models to training mode and computes loss instead of generating outputs."

### 2. Dataset Insight
> "The UnifiedDataset accepts CSV metadata with customizable columns specified by `data_file_keys`. Each column is processed by an operator (function or pipeline of functions). The `main_data_operator` handles the primary data (usually video), while `special_operator_map` defines custom processing for specific columns. For S2V, we need columns: video, audio, prompt, and optionally s2v_pose_video."

### 3. Training Insight
> "The `--extra_inputs` flag is the key mechanism that enables training with additional modalities. It tells the training script to:
> 1. Load specified columns from the CSV dataset
> 2. Add them to the `inputs_shared` dictionary
> 3. Pass them through the pipeline units
>
> Pipeline units check for their required inputs in `inputs_shared`. If present, they process them. This allows the same training script to support different models (T2V, I2V, S2V, etc.) just by changing the `--extra_inputs` flag and loading appropriate model components."

### 4. S2V-Specific Insight
> "S2V training works exactly like Wan2.2-Animate training, but with different extra inputs:
> - Animate: `--extra_inputs 'input_image,animate_pose_video,animate_face_video'`
> - S2V: `--extra_inputs 'input_image,input_audio,audio_sample_rate,s2v_pose_video'`
>
> The S2V unit (WanVideoUnit_S2V) processes audio into embeddings using wav2vec2, which are then passed to the DiT for audio-conditioned video generation. No custom training code needed!"

### 5. Adaptation Strategy for RenderMe360
> "Our RenderMe360 dataset contains all required modalities:
> - ✅ Synchronized videos (20 camera views)
> - ✅ Audio tracks (MP3, can be converted to 16kHz)
> - ✅ 3D keypoints (can generate pose videos)
> - ✅ High-resolution images (2448x2048, can resize)
>
> Main preprocessing needed:
> 1. Extract 81-frame video clips (satisfying num_frames % 4 == 1)
> 2. Align and resample audio segments to 16kHz
> 3. Create metadata CSV with required columns
> 4. Add text prompts (generic or auto-generated)
> 5. (Optional) Render pose videos from 3D keypoints"

---

## Presentation Script for Leader

### Opening
> "I analyzed the DiffSynth-Studio codebase to understand how to fine-tune Wan S2V on our RenderMe360 dataset. The investigation revealed that the framework already supports everything we need through its modular pipeline architecture."

### Step-by-Step Findings

**Step 1: Model Requirements**
> "I examined the S2V inference example (`Wan2.2-S2V-14B_multi_clips.py`) and discovered:
> - S2V requires audio at 16kHz sample rate, processed by wav2vec2 audio encoder
> - Takes a reference image (first frame) and generates synchronized speech video
> - Must generate videos with frame count satisfying: `num_frames % 4 == 1` (e.g., 81 frames)
> - Uses 73 frames of motion context for temporal consistency"

**Step 2: Pipeline Architecture**
> "I traced the pipeline implementation in `wan_video_new.py` and found:
> - The pipeline is composed of modular 'units' that process inputs sequentially
> - The S2V unit (`WanVideoUnit_S2V`, lines 972-1052) handles audio processing
> - It converts audio waveforms to embeddings using wav2vec2
> - These embeddings are added to the pipeline flow and passed to the DiT
> - Same architecture used for inference and training"

**Step 3: Training Mechanism**
> "I studied the training framework (`train.py`) and identified the key mechanism:
> - The `--extra_inputs` flag specifies which additional data columns to load from CSV
> - Training script loads these columns and adds them to `inputs_shared` dictionary
> - Pipeline units check for their required inputs in this dictionary
> - If S2V unit finds `input_audio`, it processes it into embeddings
> - These embeddings condition the DiT during training - no custom code needed!"

**Step 4: Validation via Similar Model**
> "I analyzed Wan2.2-Animate training configuration (`Wan2.2-Animate-14B.sh`) and confirmed:
> - Uses same training script with different `--extra_inputs` flag
> - Animate: `--extra_inputs 'input_image,animate_pose_video,animate_face_video'`
> - For S2V, we need: `--extra_inputs 'input_image,input_audio,s2v_pose_video'`
> - Same pattern, different inputs - validates our approach"

**Step 5: Dataset Mapping**
> "I mapped our RenderMe360 dataset to training requirements:
> - ✅ We have: Multi-view videos (20 cameras), synchronized audio, 3D keypoints
> - ⚠️ We need: Text prompts (can use generic like 'A person speaking')
> - 📋 Preprocessing: Extract 81-frame clips, resample audio to 16kHz, create CSV
> - 📊 Result: ~78,000 training samples (all cameras) or ~55,000 (frontal cameras only)"

### Conclusion
> "The solution is straightforward:
> 1. **Preprocess data**: Extract 81-frame clips with aligned 16kHz audio segments
> 2. **Create CSV**: `video,audio,prompt,input_image,s2v_pose_video`
> 3. **Use existing training script** with: `--extra_inputs 'input_image,input_audio,s2v_pose_video'`
> 4. **LoRA fine-tuning** on 8×80GB GPUs for ~3-5 days
>
> I've documented the complete implementation plan in `TRAINING_PLAN_WAN_S2V_RENDERME360.md` with preprocessing scripts, training configuration, and evaluation strategy."

### Technical Depth (if asked)
> "The beauty of this framework is its modularity:
> - Pipeline units are self-contained processors
> - They declare what inputs they need
> - Training script generically passes data through the pipeline
> - Each unit processes only if its inputs are present
> - This allows one training script to support multiple model types
>
> For S2V specifically:
> - The S2V unit checks for `input_audio` in the shared inputs dictionary
> - If present, it loads the audio encoder and processes audio → embeddings
> - These embeddings are added to positive conditioning (`inputs_posi`)
> - Zero embeddings added to negative conditioning for CFG
> - DiT receives audio-conditioned inputs during forward pass
> - Standard diffusion loss is computed
>
> This is why no custom training code is needed - the architecture handles it!"

---

## Next Steps Checklist

Based on this investigation, here are the immediate next steps:

### Data Preparation
- [ ] Verify audio format (check if already 16kHz or needs resampling)
- [ ] Write video clip extraction script (81 frames per clip)
- [ ] Write audio segment alignment script
- [ ] Create metadata CSV generator
- [ ] Test data loading with one sample

### Training Setup
- [ ] Modify training script to load audio_processor
- [ ] Create training configuration file
- [ ] Test on small subset (1 subject, 1 camera)
- [ ] Validate memory usage and adjust batch size

### Execution
- [ ] Launch full training run
- [ ] Monitor loss curves and VRAM usage
- [ ] Generate intermediate samples for validation
- [ ] Evaluate on held-out subjects

---

## Additional Resources

### Official Documentation
- DiffSynth-Studio README: `/home/zhuoyuan/projects/DiffSynth-Studio/README.md`
- Wan Video Examples: `/home/zhuoyuan/projects/DiffSynth-Studio/examples/wanvideo/`
- Project CLAUDE.md: `/home/zhuoyuan/projects/DiffSynth-Studio/CLAUDE.md`

### Implementation Plan
- Training Plan: `/home/zhuoyuan/projects/DiffSynth-Studio/TRAINING_PLAN_WAN_S2V_RENDERME360.md`

### Code References
- S2V Pipeline Unit: `diffsynth/pipelines/wan_video_new.py:972-1052`
- Training Module: `examples/wanvideo/model_training/train.py:42-82`
- UnifiedDataset: `diffsynth/trainers/unified_dataset.py:230-249`

---

**End of Investigation Document**
