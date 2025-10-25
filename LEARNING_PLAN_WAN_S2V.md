# Learning Plan: Understanding Wan S2V Training in DiffSynth-Studio

**Goal**: Understand how to fine-tune Wan2.2-S2V-14B on RenderMe360 dataset
**Estimated Time**: 4-5 hours
**Strategy**: Bottom-Up (understand components first, then see how they're orchestrated)

---

## 📋 Learning Progress Tracker

```
Phase 1: Understand S2V Inputs/Outputs       [ ] 30 min
Phase 2: Understand S2V Pipeline Unit        [ ] 45 min
Phase 3: Understand Dataset Format           [ ] 30 min
Phase 4: Understand Training Script          [ ] 60 min
Phase 5: See Working Example                 [ ] 30 min
Phase 6: Understand Arguments (Optional)     [ ] 20 min
──────────────────────────────────────────────────────
Total:                                       [ ] 3-4 hours
```

---

## Phase 1: Understand S2V Inputs/Outputs (30 min)

### **Goal**: What does S2V need and produce?

### **File to Study**
📄 **`examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py`**

### **What to Focus On**

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 9-23 | `speech_to_video()` function signature | Required inputs: prompt, input_image, audio_path, pose_video_path |
| 26 | `librosa.load()` | Audio must be 16kHz sample rate |
| 30-39 | `WanVideoUnit_S2V.pre_calculate_audio_pose()` | Audio is preprocessed into embeddings before inference |
| 46-66 | Video generation loop | Generates video in clips, uses motion context from previous frames |
| 70-79 | Model components | 4 components needed: DiT, T5, wav2vec2, VAE |
| 88-90 | Frame constraint | `infer_frames = 80` → `num_frames = 81` (must satisfy `num_frames % 4 == 1`) |

### **Questions to Answer**
- [ ] What are the required inputs for S2V?
- [ ] What sample rate does audio need to be?
- [ ] What is the frame count constraint?
- [ ] What model components are required?
- [ ] Is pose video optional or required?

### **Key Takeaway**
```
S2V Input:  audio (16kHz) + reference image + prompt + [optional] pose video
S2V Output: Speaking video synchronized with audio
Constraint: num_frames % 4 == 1 (e.g., 81, 85, 89 frames)
```

### **Action Items**
- [ ] Read the entire file once quickly
- [ ] Re-read focusing on the table sections above
- [ ] Write down the key inputs in your notes
- [ ] Answer all 5 questions above

---

## Phase 2: Understand S2V Pipeline Unit (45 min) ⭐⭐⭐ CRITICAL

### **Goal**: How does S2V process audio internally?

### **File to Study**
📄 **`diffsynth/pipelines/wan_video_new.py`** (Lines 972-1052 ONLY)

### **What to Focus On**

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 972-977 | `WanVideoUnit_S2V` class definition | Unit loads `audio_encoder` and `vae` when needed |
| 979-987 | `process_audio()` method | Converts audio waveform → embeddings using wav2vec2 |
| 989-1002 | `process_motion_latents()` method | Uses 73 frames of motion context for temporal consistency |
| 1004-1025 | `process_pose_cond()` method | Processes optional pose video into latents |
| 1027-1040 | `process()` main method | **KEY**: Checks for `input_audio` in `inputs_shared`, processes it, adds to `inputs_posi` |

### **Questions to Answer**
- [ ] What does `process_audio()` do?
- [ ] Where does the unit expect to find `input_audio`? (Hint: line 1028)
- [ ] What does the unit add to `inputs_posi`? (Hint: line 1035)
- [ ] What does the unit add to `inputs_nega`? (Hint: line 1036)
- [ ] Is pose video required? (Hint: line 1028-1029)

### **Key Takeaway**
```
S2V Unit's Logic:
1. Check if `input_audio` exists in inputs_shared
2. If YES:
   - Load audio_encoder
   - Process audio → audio_embeds
   - Add audio_embeds to inputs_posi (positive conditioning)
   - Add 0.0 * audio_embeds to inputs_nega (CFG)
3. If NO:
   - Skip audio processing
```

### **Critical Understanding**
```python
# This is THE KEY mechanism:
def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
    if inputs_shared.get("input_audio") is None:
        return  # Skip if no audio

    # Extract audio from inputs_shared
    input_audio = inputs_shared.pop("input_audio")

    # Process into embeddings
    audio_embeds = process_audio(input_audio, ...)

    # Add to positive/negative conditioning
    inputs_posi["audio_embeds"] = audio_embeds
    inputs_nega["audio_embeds"] = 0.0 * audio_embeds
```

### **Action Items**
- [ ] Read lines 972-1052 carefully
- [ ] Trace the flow: `input_audio` → `process_audio()` → `audio_embeds`
- [ ] Understand what gets added to `inputs_posi` and `inputs_nega`
- [ ] Answer all 5 questions above
- [ ] **Draw a simple flowchart** of the `process()` method

---

## Phase 3: Understand Dataset Format (30 min)

### **Goal**: How to prepare CSV and load data?

### **File to Study**
📄 **`diffsynth/trainers/unified_dataset.py`**

### **What to Focus On**

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 230-249 | `UnifiedDataset` class `__init__()` | How dataset is configured: base_path, metadata_path, data_file_keys, operators |
| 62-69 | `LoadImage` operator | How images are loaded from paths |
| 117-143 | `LoadVideo` operator | How videos are loaded, **frame count constraint** |
| 221-226 | `ToAbsolutePath` operator | How relative paths are converted to absolute paths |

### **Questions to Answer**
- [ ] What is `data_file_keys`? (Hint: CSV column names)
- [ ] What is `main_data_operator`? (Hint: processing pipeline for main data)
- [ ] What is `special_operator_map`? (Hint: custom operators for specific columns)
- [ ] How does `LoadVideo` ensure `num_frames % 4 == 1`? (Hint: lines 125-131)
- [ ] How are CSV paths converted to absolute paths?

### **Key Takeaway**
```
CSV Structure:
  Column Name (data_file_keys)
      ↓
  Operator (loads and processes)
      ↓
  data[column_name] (Python dict)

Example:
CSV: video,audio,prompt
  ↓
data = {
  "video": [PIL.Image, ...],    # LoadVideo operator
  "audio": audio_waveform,       # LoadAudio operator
  "prompt": "A person speaking"  # ToStr operator
}
```

### **Expected CSV Format for S2V**
```csv
video,audio,prompt,input_image,s2v_pose_video
path/to/clip.mp4,path/to/audio.mp3,"A person speaking",path/to/frame0.jpg,path/to/pose.mp4
```

### **Action Items**
- [ ] Read lines 230-249 to understand UnifiedDataset
- [ ] Read LoadVideo operator to understand frame constraints
- [ ] Understand how operators transform CSV columns to data dict
- [ ] Answer all 5 questions above
- [ ] **Write down** the CSV format you'll need for S2V

---

## Phase 4: Understand Training Script (60 min) ⭐⭐⭐ CRITICAL

### **Goal**: How does the training script orchestrate everything?

### **File to Study**
📄 **`examples/wanvideo/model_training/train.py`**

### **Part 1: The Key Method (30 min)**

**Focus on Lines 42-82: `forward_preprocess()` method**

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 44-45 | CFG-sensitive vs unsensitive inputs | `inputs_posi` vs `inputs_shared` |
| 48-65 | Building `inputs_shared` dict | How video, height, width, etc. are added |
| 69-77 | **Extra inputs handling** | **THE MECHANISM!** How `extra_inputs` are added to `inputs_shared` |
| 79-81 | Pipeline unit processing | How units transform inputs |

### **Questions to Answer (Part 1)**
- [ ] What's the difference between `inputs_posi` and `inputs_shared`?
- [ ] What does line 70-71 do? (Hint: handles `input_image`)
- [ ] What does line 76-77 do? (Hint: generic passthrough for other inputs)
- [ ] How do pipeline units get the inputs? (Hint: line 79-81)
- [ ] Where does `self.extra_inputs` come from? (Hint: line 37)

### **THE CRITICAL UNDERSTANDING**
```python
# Lines 69-77: This is HOW audio gets to S2V unit!

for extra_input in self.extra_inputs:  # e.g., ["input_image", "input_audio", ...]
    if extra_input == "input_image":
        inputs_shared["input_image"] = data["video"][0]
    elif extra_input == "end_image":
        inputs_shared["end_image"] = data["video"][-1]
    else:
        # GENERIC PASSTHROUGH!
        inputs_shared[extra_input] = data[extra_input]
        # This means: inputs_shared["input_audio"] = data["audio"]

# Then pipeline units process inputs_shared
for unit in self.pipe.units:
    inputs_shared, inputs_posi, inputs_nega = unit.process(...)
    # S2V unit sees input_audio in inputs_shared!
```

### **Part 2: The Main Entry Point (30 min)**

**Focus on Lines 92-132: `if __name__ == "__main__":` **

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 93-94 | Argument parsing | Where command-line args come from |
| 95-114 | Dataset creation | How UnifiedDataset is configured |
| 115-127 | Training module creation | How WanTrainingModule is initialized |
| 124 | **`extra_inputs` argument** | This comes from `--extra_inputs` CLI flag! |
| 128-132 | Training loop launch | How training actually starts |

### **Questions to Answer (Part 2)**
- [ ] Where does `args.extra_inputs` come from? (Hint: CLI argument)
- [ ] How is `data_file_keys` used? (Hint: CSV columns to load)
- [ ] What is `main_data_operator`? (Hint: video loading pipeline)
- [ ] How is the training module initialized? (Hint: lines 115-127)
- [ ] What does `launch_training_task()` do? (Hint: starts training loop)

### **Action Items**
- [ ] Read lines 42-82 very carefully (this is the core!)
- [ ] Trace how `extra_inputs` flows: CLI → init → forward_preprocess
- [ ] Read lines 92-132 to see the big picture
- [ ] Answer all 10 questions above
- [ ] **Draw a flowchart**: CSV → dataset → forward_preprocess → units → loss

---

## Phase 5: See Working Example (30 min)

### **Goal**: See how another model uses `extra_inputs`

### **File to Study**
📄 **`examples/wanvideo/model_training/lora/Wan2.2-Animate-14B.sh`**

### **What to Focus On**

| Lines | Content | What to Learn |
|-------|---------|---------------|
| 1-2 | GPU requirements | 8×80GB GPUs needed for 14B model LoRA training |
| 3 | `accelerate launch` command | How distributed training is started |
| 4-7 | Dataset configuration | `dataset_base_path`, `dataset_metadata_path`, `data_file_keys` |
| 8-11 | Video specifications | `height`, `width`, `num_frames`, `dataset_repeat` |
| 12 | Model loading | Colon-separated `model_id:file_pattern` pairs |
| 16-18 | LoRA configuration | `lora_base_model="dit"`, `lora_target_modules`, `lora_rank=32` |
| 19 | **`--extra_inputs` flag** | **THE KEY!** `"input_image,animate_pose_video,animate_face_video"` |
| 20 | Memory optimization | `--use_gradient_checkpointing_offload` |

### **Questions to Answer**
- [ ] What extra_inputs does Animate use? (Hint: line 19)
- [ ] What extra_inputs should S2V use? (Think about Phase 2!)
- [ ] What's the pattern for `model_id_with_origin_paths`? (Hint: line 12)
- [ ] What does `--lora_base_model "dit"` mean? (Hint: only DiT is trainable)
- [ ] What GPU resources are needed?

### **Key Comparison**
```bash
# Wan2.2-Animate-14B:
--extra_inputs "input_image,animate_pose_video,animate_face_video"

# Wan2.2-S2V-14B (what you need to create):
--extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video"
```

### **What You Need to Change for S2V**
```bash
# 1. Model files:
--model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,\
Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,\
Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,\  # ← Add this!
Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth"

# 2. Data keys:
--data_file_keys "video,audio,prompt,s2v_pose_video"  # Changed

# 3. Extra inputs:
--extra_inputs "input_image,input_audio,audio_sample_rate,s2v_pose_video"  # Changed

# Everything else: SAME!
```

### **Action Items**
- [ ] Read the entire .sh script
- [ ] Identify all arguments and their purposes
- [ ] Compare Animate's extra_inputs with what S2V needs
- [ ] Answer all 5 questions above
- [ ] **Write down** what parameters you need to change for S2V

---

## Phase 6: Understand Arguments (Optional, 20 min)

### **Goal**: What arguments are available?

### **File to Study**
📄 **`diffsynth/trainers/utils.py`** (Lines 594-650)

### **What to Focus On**

**Key Arguments for Training:**
- `--dataset_base_path`: Base directory for dataset
- `--dataset_metadata_path`: Path to CSV file
- `--data_file_keys`: CSV column names (comma-separated)
- `--model_id_with_origin_paths`: Model loading specification
- `--extra_inputs`: Additional inputs to pass to pipeline
- `--lora_base_model`: Which component to add LoRA to
- `--lora_target_modules`: Which layers get LoRA adapters
- `--lora_rank`: LoRA rank (default: 32)
- `--height`, `--width`, `--num_frames`: Video dimensions
- `--learning_rate`: Learning rate
- `--num_epochs`: Number of training epochs

### **Action Items**
- [ ] Skim through the parser to see available arguments
- [ ] Understand the purpose of each key argument
- [ ] Reference this when creating your training script

---

## 🎯 Final Verification: The Clarity Test

**After completing all phases, verify your understanding:**

### **Test 1: Can you draw the data flow diagram?**
```
[ ] CSV → UnifiedDataset → data dict
[ ] data dict → forward_preprocess → inputs_shared
[ ] inputs_shared → S2V unit → audio_embeds
[ ] audio_embeds → DiT → loss
```

### **Test 2: Can you explain to someone?**
```
[ ] How does --extra_inputs work?
[ ] Why does S2V training use the same script as Animate?
[ ] What's the difference between S2V and Animate training configs?
[ ] How does audio get from CSV to the DiT model?
```

### **Test 3: Can you create the training script?**
```
[ ] Know which .sh script to copy as template
[ ] Know which parameters to change
[ ] Know what CSV format is needed
[ ] Know what data preprocessing is required
```

**All checked?** → You're ready to implement! 🚀

---

## 📝 Key Insights Summary

After completing this learning plan, you should understand:

### **1. The S2V Unit Mechanism**
```
input_audio in inputs_shared
    ↓
S2V Unit processes
    ↓
audio_embeds added to inputs_posi
    ↓
DiT receives audio conditioning
```

### **2. The extra_inputs Mechanism**
```
--extra_inputs "input_image,input_audio,..."  # CLI
    ↓
self.extra_inputs = ["input_image", "input_audio", ...]  # Training module
    ↓
for extra_input in self.extra_inputs:
    inputs_shared[extra_input] = data[extra_input]  # forward_preprocess
    ↓
Pipeline units see inputs_shared  # S2V unit processes input_audio
```

### **3. The Complete Data Flow**
```
CSV (metadata.csv)
  ↓
UnifiedDataset loads → data["video"], data["audio"], data["prompt"]
  ↓
forward_preprocess() → inputs_shared["input_audio"] = data["audio"]
  ↓
S2V Unit → audio_embeds
  ↓
DiT → loss
```

### **4. What You Need to Create**
```
1. Data Preprocessing:
   - Extract 81-frame video clips
   - Extract/resample audio to 16kHz
   - Create metadata CSV

2. Training Script:
   - Copy Wan2.2-Animate-14B.sh
   - Change model_id_with_origin_paths (add wav2vec2)
   - Change extra_inputs (use audio instead of animate videos)
   - Change data_file_keys (match CSV columns)

3. No Python code changes needed!
```

---

## 🎓 Next Steps After Learning

Once you complete this learning plan:

1. **Create data preprocessing scripts** (see `TRAINING_PLAN_WAN_S2V_RENDERME360.md`)
2. **Create training shell script** (based on Wan2.2-Animate-14B.sh)
3. **Test on small subset** (1 subject, 1 camera)
4. **Launch full training** (8×80GB GPUs)

---

## 📚 Reference Documents

- **Training Plan**: `TRAINING_PLAN_WAN_S2V_RENDERME360.md` - What to build
- **Investigation**: `INVESTIGATION_WAN_S2V.md` - How we figured it out
- **Methodology**: `LEARNING_METHODOLOGY_TOP_DOWN_VS_BOTTOM_UP.md` - Learning framework
- **Project Guide**: `CLAUDE.md` - DiffSynth-Studio overview

---

## ✅ Completion Checklist

```
Phase 1: Understand S2V Inputs/Outputs       [ ]
  └─ Answered all questions                  [ ]
  └─ Noted key inputs in notes               [ ]

Phase 2: Understand S2V Pipeline Unit        [ ]
  └─ Answered all questions                  [ ]
  └─ Drew flowchart of process() method      [ ]

Phase 3: Understand Dataset Format           [ ]
  └─ Answered all questions                  [ ]
  └─ Wrote down CSV format needed            [ ]

Phase 4: Understand Training Script          [ ]
  └─ Answered all questions (Part 1)         [ ]
  └─ Answered all questions (Part 2)         [ ]
  └─ Drew complete data flow diagram         [ ]

Phase 5: See Working Example                 [ ]
  └─ Answered all questions                  [ ]
  └─ Listed parameters to change for S2V     [ ]

Phase 6: Understand Arguments (Optional)     [ ]
  └─ Reviewed available arguments            [ ]

Final Verification                           [ ]
  └─ Passed Test 1 (can draw diagram)        [ ]
  └─ Passed Test 2 (can explain)             [ ]
  └─ Passed Test 3 (can implement)           [ ]
```

---

**Ready to start? Begin with Phase 1!** 🚀

Good luck! Remember: understanding > speed. Take your time in each phase.
