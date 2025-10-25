# Wan2.2-S2V-14B Training on RenderMe360 - Session Log

**Date:** 2025-10-25
**Server:** vllab9
**Task:** Launch validation test and fix bugs to get training running

---

## What We Accomplished

### 1. ✅ Dataset Testing (PASSED)
**File:** `scripts/test_dataset.py`
- Verified RenderMe360S2VDataset loads correctly
- Confirmed 1,048 training samples ready
- All data shapes and types validated:
  - Video: 81 frames @ 448×832 (2×2 grid)
  - Input image: 224×416 (cam_54 frontal view)
  - Audio: 80,000 samples @ 16kHz (5.0s)

### 2. ✅ Fixed Library Dependencies
**Challenge:** Missing Python packages causing import errors

**Actions Taken:**
```bash
pip install soxr                    # Audio resampling
pip install 'numpy>=1.22.4,<1.25.0' # Pandas compatibility
pip install 'numba>=0.56'           # NumPy 1.24 compatibility
pip install deepspeed               # Distributed training
```

### 3. ✅ Fixed collate_fn Bug
**Challenge:** Training code tried to pass custom `collate_fn` but `launch_training_task()` didn't accept it

**File Modified:** `diffsynth/trainers/utils.py:521-550`

**Fix:**
```python
# Added collate_fn parameter
def launch_training_task(
    dataset, model, model_logger, ...,
    collate_fn=None,  # NEW PARAMETER
    args=None
):
    # Use provided collate_fn or default
    if collate_fn is None:
        collate_fn = lambda x: x[0]

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True,
        collate_fn=collate_fn,  # Now uses custom function
        num_workers=num_workers
    )
```

**Why Needed:** Default PyTorch collate can't handle PIL Images. Our custom `passthrough_collate()` returns raw batch[0] for batch_size=1.

### 4. ✅ Fixed Disk Space Issues
**Challenge 1:** Codebase was 20GB (should be ~65MB)

**Root Cause:** Models directory (20GB) was in codebase, filling NAS quota

**Solution:**
```bash
# Move models to /ssd2 (282GB available)
mv models /ssd2/zhuoyuan/diffsynth_models/
ln -s /ssd2/zhuoyuan/diffsynth_models/models ./models

# Result: Codebase now 65MB ✅
```

**Challenge 2:** DeepSpeed cache filling home directory (100GB NAS quota)

**Error:** `OSError: [Errno 122] Disk quota exceeded` when DeepSpeed tried to write autotune cache

**Solution:** Redirect DeepSpeed cache to /ssd2
```bash
export TRITON_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
export DEEPSPEED_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
```

**Files Modified:**
- `examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360-TEST.sh`
- `examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360.sh` (full training)

Also updated `--output_path` to save checkpoints to `/ssd2/zhuoyuan/diffsynth_training/` instead of codebase.

---

## 🚨 Current Blocking Issue: Audio Processor Config Files Missing

### The Problem

**Error:**
```
OSError: Can't load feature extractor for './models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/'.
If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local
directory with the same name. Otherwise, make sure './models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/'
is the correct path to a directory containing a preprocessor_config.json file
```

### Root Cause Analysis

**Investigation Steps:**
1. ✅ Checked what files ModelScope downloaded
2. ✅ Searched HuggingFace repo for file list
3. ✅ Compared with upstream training code
4. ✅ Tested downloader behavior

**Findings:**

| Location | Files Available |
|----------|----------------|
| **HuggingFace** (Wan-AI/Wan2.2-S2V-14B) | ✅ model.safetensors<br>✅ config.json<br>✅ preprocessor_config.json<br>✅ tokenizer_config.json<br>✅ special_tokens_map.json<br>✅ vocab.json |
| **ModelScope Download** (our server) | ✅ model.safetensors<br>❌ config.json<br>❌ preprocessor_config.json<br>❌ tokenizer_config.json<br>❌ special_tokens_map.json<br>❌ vocab.json |

**Diagnosis:** This is a **ModelScope downloader bug**. When downloading folders (pattern ends with `/`), it only downloads certain files (model weights) but skips config files.

**Evidence:**
```bash
# What we have:
$ ls models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/
model.safetensors  # 1.26 GB - Downloaded ✅

# What's missing (needed by Wav2Vec2Processor.from_pretrained()):
# preprocessor_config.json  ❌
# config.json               ❌
# tokenizer_config.json     ❌
# special_tokens_map.json   ❌
# vocab.json                ❌
```

**Why Upstream Works:** They likely manually downloaded the config files once or have them cached. After initial setup, training runs fine.

### The Solution

**Option 1 (RECOMMENDED):** Download missing config files from HuggingFace once
- Simple, one-time fix
- Config files are tiny (~10 KB total)
- Won't need re-downloading

**Option 2:** Switch to HuggingFace downloader for audio processor
- Requires code changes
- More complex

**Option 3:** Wait for ModelScope to fix their downloader
- Could take weeks/months
- Blocks training now

---

## 📋 Commands to Resume Training

### Step 1: Download Missing Audio Processor Config Files

```bash
# Activate environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Download wav2vec2 processor with all config files from HuggingFace
cd /tmp
cat > download_wav2vec.py << 'EOF'
from transformers import Wav2Vec2Processor

# Download complete processor (includes all config files)
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53')

# Save to our models directory
save_path = '/home/zhuoyuan/projects/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english'
processor.save_pretrained(save_path)
print(f"✅ Audio processor configs saved to {save_path}")

# Verify all files present
import os
required_files = ['preprocessor_config.json', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json']
for file in required_files:
    path = os.path.join(save_path, file)
    if os.path.exists(path):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} MISSING")
EOF

python download_wav2vec.py
```

**Expected Output:**
```
✅ Audio processor configs saved to .../wav2vec2-large-xlsr-53-english
✅ preprocessor_config.json
✅ config.json
✅ tokenizer_config.json
✅ special_tokens_map.json
✅ vocab.json
```

### Step 2: Resume Validation Test

```bash
cd ~/projects/DiffSynth-Studio
bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360-TEST.sh
```

**What to Monitor:**
1. LoRA modules matched: Should print "Found X potential target modules" (X > 0)
2. Training starts: Look for loss values appearing
3. Memory usage: VRAM < 45GB per GPU
4. No errors: Should run for at least 50 steps

### Step 3: Launch Full Training (After Test Passes)

```bash
cd ~/projects/DiffSynth-Studio
bash examples/wanvideo/model_training/lora/Wan2.2-S2V-14B-RenderMe360.sh
```

**Training Configuration:**
- Samples: 1,048 (21 subjects × ~50 clips each)
- Epochs: 500
- Total steps: ~524,000
- Checkpoints: Every 10,000 steps → `/ssd2/zhuoyuan/diffsynth_training/Wan2.2-S2V-14B_RenderMe360_lora/`
- Expected duration: ~2 days on 8× RTX 6000 Ada

---

## 🗂️ File Structure Summary

### Custom Dataset Implementation
```
diffsynth/trainers/renderme360_dataset.py (301 lines)
├── RenderMe360S2VDataset class
│   ├── Loads 2×2 grid video (4 cameras)
│   ├── Nearest-frame sampling: 30fps → 16fps
│   ├── Audio loading with librosa (80k samples)
│   └── Returns: {video, input_image, input_audio, audio_sample_rate, prompt}
└── passthrough_collate() - Custom collate for PIL Images
```

### Metadata Generation
```
scripts/generate_metadata_single2multi.py (113 lines)
└── Output: /ssd2/zhuoyuan/renderme360_4cam/metadata_single2multi.csv
    ├── 1,048 training samples
    ├── Stride: 150 frames (0% overlap)
    └── Columns: subject, performance, start_frame_30fps, num_frames, audio_path, input_camera, prompt
```

### Training Scripts
```
examples/wanvideo/model_training/train_s2v.py (180 lines)
├── WanS2VTrainingModule
│   ├── Audio processor config handling
│   ├── LoRA verification
│   └── Custom forward pass
└── Uses RenderMe360S2VDataset with passthrough_collate

examples/wanvideo/model_training/lora/
├── Wan2.2-S2V-14B-RenderMe360-TEST.sh  # Validation (50 samples, 1 epoch)
└── Wan2.2-S2V-14B-RenderMe360.sh       # Full training (1048 samples, 500 epochs)
```

### Storage Locations
```
/home/zhuoyuan/projects/DiffSynth-Studio/  # 65 MB (code only)
└── models -> /ssd2/zhuoyuan/diffsynth_models/models  # Symlink

/ssd2/zhuoyuan/
├── renderme360_4cam/                    # Dataset (raw images + audio)
│   ├── metadata_single2multi.csv        # Full training metadata
│   └── metadata_test.csv                # Test metadata (50 samples)
├── diffsynth_models/models/             # Model weights (20 GB)
│   └── Wan-AI/Wan2.2-S2V-14B/
│       ├── diffusion_pytorch_model-*.safetensors (4 parts, 32 GB total)
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       ├── Wan2.1_VAE.pth
│       └── wav2vec2-large-xlsr-53-english/
│           ├── model.safetensors ✅
│           └── [config files] ❌ <- NEED TO DOWNLOAD
├── diffsynth_training/                  # Training checkpoints
│   ├── Wan2.2-S2V-14B_RenderMe360_lora_TEST/
│   └── Wan2.2-S2V-14B_RenderMe360_lora/
└── deepspeed_cache/                     # DeepSpeed autotune cache

/ssd1/zhuoyuan/hf_cache/                 # HuggingFace model cache (41 GB)
```

---

## 🔧 Environment Configuration

### Conda Environment
```bash
Environment: diffsynth-s2v
Python: 3.10
Key packages:
- torch 2.9.0
- transformers (with wav2vec2)
- deepspeed 0.18.1
- accelerate
- librosa
- pandas, numpy 1.24.4, numba 0.62.1
```

### Environment Variables (Set in Training Scripts)
```bash
HF_HOME=/ssd1/zhuoyuan/hf_cache              # HuggingFace cache
TRITON_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
DEEPSPEED_CACHE_DIR=/ssd2/zhuoyuan/deepspeed_cache
TOKENIZERS_PARALLELISM=false
```

### Hardware
```
Server: vllab9
GPUs: 8× NVIDIA RTX 6000 Ada (49GB VRAM each)
Training: DeepSpeed ZeRO-2 with CPU offloading
Batch size: 1 (per GPU)
```

---

## 📝 Key Technical Details

### Frame Sampling Algorithm
```python
# Converts 30fps source to 16fps output using integer arithmetic
for k in range(81):  # 81 frames = 5.0 seconds @ 16fps
    idx_30fps = start_frame + ((k * 30 + 8) // 16)
    # Example: k=0 → 0, k=1 → 1, k=2 → 3, k=3 → 5, ...
```

### Grid Layout
```
┌──────────────┬──────────────┐
│   cam_28     │   cam_37     │  Each: 224×416 pixels
│ (Top-Left)   │ (Top-Right)  │
├──────────────┼──────────────┤
│   cam_49     │   cam_54     │  Total: 448×832 pixels
│ (Bottom-Left)│ (Bottom-Right)│
└──────────────┴──────────────┘
```

### LoRA Configuration
```
Base model: dit (Diffusion Transformer)
Target modules: q,k,v,o,ffn.0,ffn.2
Rank: 32
Learning rate: 1e-4
```

---

## 🎯 Next Actions

1. **IMMEDIATE:** Run the download command above to get wav2vec2 config files
2. **TEST:** Run validation script (50 samples, ~2-4 hours)
3. **VERIFY:** Check LoRA matched, loss decreases, no OOM
4. **LAUNCH:** Start full 500-epoch training (~2 days)

---

## 💡 Lessons Learned

1. **ModelScope Downloader:** Has bugs with folder downloads - missing config files
2. **Home Directory Quota:** Always check `/home` quota (100GB NAS limit) - use local SSDs
3. **DeepSpeed Cache:** Needs explicit directory or fills home
4. **PIL Images:** Require custom collate_fn, can't use default PyTorch collate
5. **Dependency Hell:** numpy/numba/pandas versions must align carefully

---

## 📚 References

- Training Plan: `Training_Plan_S2V_4view.md`
- Dataset Summary: `RenderMe360_4cam_dataset_summary.md`
- Rationale: `Training_Plan_S2V_4view_Rationale.md`
- Upstream S2V update: commit `30292d9` (2025-10-21)
- HuggingFace model: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
