# Setup Guide: RenderMe360 Training on Wan2.2-S2V-14B

## Overview

**Goal:** Fine-tune Wan2.2-S2V-14B on RenderMe360 dataset for speech-to-video with novel view synthesis

**Prerequisite:** Server with 8× GPUs (preferably A100 80GB or better)

**GitHub Repository:** https://github.com/ZhuoyuanJiang/diffSynth-studio-notes

This training pipeline generates talking head videos with novel view synthesis:
- **Input:** Single frontal view (cam_28) + 5-second audio
- **Output:** 2×2 grid (4 camera views) × 81 frames with lip-synced speech

**Known issue:** OOM on 48GB GPUs (RTX 6000 Ada). 14B model requires ~42GB + training overhead. This is why we need higher VRAM GPUs.

---

## Important: Path Setup - Two Options

### Option A: Easy Setup (No Script Editing Required!) 🌟

**Recommended if you want the simplest setup.**

Just create matching directories on YOUR server:
```bash
# Check if /ssd1/ exists on your server
ls /ssd1/

# If /ssd1/ exists, create this directory structure:
sudo mkdir -p /ssd1/zhuoyuan
sudo chown $USER:$USER /ssd1/zhuoyuan  # Give yourself ownership

# If /ssd1/ doesn't exist but you have /ssd2/ or other storage:
# Create symlink: sudo ln -s /ssd2/ /ssd1/
# Then: mkdir -p /ssd1/zhuoyuan
```

**That's it!** If you use `/ssd1/zhuoyuan/` as your base path, you won't need to edit ANY training scripts. Just transfer data/models to:
- Dataset: `/ssd1/zhuoyuan/renderme360_4cam/`
- Models: `/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/`
- Outputs will go to: `/ssd1/zhuoyuan/diffsynth_training/`

**Skip to Section 1** and use `/ssd1/zhuoyuan/` everywhere.

---

### Option B: Custom Paths (Requires Script Editing)

**Use this if:**
- You can't create `/ssd1/zhuoyuan/` for permission reasons
- You prefer using your own username/paths
- Your server doesn't have `/ssd1/` and you don't want symlinks

**Key directories you'll create:**
- Dataset storage: `/ssd1/<your-username>/renderme360_4cam/` (or your equivalent local storage)
- Model storage: `/ssd1/<your-username>/diffsynth_models/models/Wan-AI/`
- Training output: `/ssd1/<your-username>/diffsynth_training/`

**⚠️ If choosing custom paths:**
- Check available storage: `df -h`
- Replace `<your-username>` with YOUR actual username throughout this guide
- **CRITICAL:** You must update paths in:
  1. Symlink creation commands (Section 4)
  2. Training scripts: `run_renderme360_test.sh` and `run_renderme360_full.sh` (Section 6)
  3. Environment variables: `HF_HOME` (Section 5)

---

## Quick Start Checklist

```bash
☐ 1. Clone repository and checkout branch
☐ 2. Set up conda environment from environment_renderme360.yml
☐ 3. Download/transfer RenderMe360 dataset
☐ 4. Download/transfer Wan2.2-S2V-14B model weights
☐ 5. Create symlinks for models
☐ 6. Configure environment variables (HF_HOME, CUDA_HOME)
☐ 7. Run smoke test (100 samples)
☐ 8. Monitor and report results
```

---

## 1. Repository Setup

### Clone and Checkout
```bash
# Clone the repository
git clone https://github.com/ZhuoyuanJiang/diffSynth-studio-notes.git DiffSynth-Studio
cd DiffSynth-Studio

# Checkout the training branch
git checkout try-s2v-upstream-custom-integration
```

### Verify Files Present
```bash
# Training files
ls examples/wanvideo/model_training/train_renderme360.py
ls examples/wanvideo/model_training/lora/run_renderme360_test.sh
ls examples/wanvideo/model_training/lora/run_renderme360_full.sh
ls examples/wanvideo/model_training/lora/accelerate_config_renderme360.yaml

# Dataset integration
ls diffsynth/trainers/renderme360_unified_dataset.py
ls diffsynth/trainers/renderme360_operators.py

# Documentation
ls SESSION_TRAINING_DEBUG.md
```

---

## 2. Environment Setup

### Option A: Conda Environment (Recommended)
```bash
# Create environment from exported file
conda env create -f environment_renderme360.yml

# Activate environment
conda activate diffsynth-s2v

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# Expected: PyTorch: 2.4.0+cu124, CUDA: 12.4
```

### Option B: Pip Requirements (Alternative)
```bash
# Create fresh conda environment
conda create -n diffsynth-s2v python=3.10 -y
conda activate diffsynth-s2v

# Install from requirements
pip install -r requirements_renderme360.txt
```

### Install DiffSynth-Studio
```bash
# Install in editable mode
pip install -e .
```

### Verify Installation
```bash
python -c "import diffsynth; print('DiffSynth installed successfully')"
python -c "import accelerate; print('Accelerate version:', accelerate.__version__)"
python -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"
```

---

## 3. Dataset Setup

### Source Location
**Server:** vllab15
**Path:** `/ssd4/zhuoyuan/renderme360_4cam`
**Size:** 443G

### Transfer Dataset

**Step 1:** Create local directory
```bash
# Check available storage on your server first
df -h

# Create dataset directory (choose your storage path)
mkdir -p /ssd1/zhuoyuan/renderme360_4cam  # Option A: Use this path (easiest)
# OR
# mkdir -p /ssd1/<your-username>/renderme360_4cam  # Option B: Use your username
```

**Step 2:** Transfer data (choose one method)

**Using rsync (recommended - can resume if interrupted):**
```bash
# Replace <your-username> with YOUR username for SSH connection
rsync -avz --progress \
  <your-username>@vllab15:/ssd4/zhuoyuan/renderme360_4cam/ \
  /ssd1/zhuoyuan/renderme360_4cam/
```

**Using scp (alternative):**
```bash
# Replace <your-username> with YOUR username for SSH connection
scp -r <your-username>@vllab15:/ssd4/zhuoyuan/renderme360_4cam/ \
  /ssd1/zhuoyuan/renderme360_4cam/
```

**Note:** Transfer takes 30-60 minutes depending on network speed.

### Verify Dataset Transfer

**Check folder exists:**
```bash
ls /ssd1/zhuoyuan/renderme360_4cam/
# Should show: metadata_single2multi.csv, 0001/, 0002/, ..., 0026/
```

**Check size (most reliable verification):**
```bash
du -sh /ssd1/zhuoyuan/renderme360_4cam/
# Should show: 443G (or close to it)
```

If the size is correct, transfer was successful!

### Dataset Structure
After transfer, you should have:
```
/ssd1/zhuoyuan/renderme360_4cam/
├── metadata_single2multi.csv          # Metadata (1,048 samples)
├── 0001/                               # Subject folders
│   ├── cam_28/                         # Input camera (frontal view)
│   │   └── 0000_img/ (frames)
│   ├── cam_37/, cam_49/, cam_54/      # Output cameras
│   │   └── 0000_img/ (frames)
│   └── audio_16khz/                    # Audio files
│       └── 0000.wav
├── 0002/
│   └── ...
└── 0026/
```

### If Metadata CSV is Missing

If `metadata_single2multi.csv` wasn't transferred:

**Option 1: Transfer just the metadata file:**
```bash
scp <your-username>@vllab15:/ssd4/zhuoyuan/renderme360_4cam/metadata_single2multi.csv \
  /ssd1/zhuoyuan/renderme360_4cam/
```

**Option 2: Generate it yourself:**
```bash
python scripts/generate_metadata_single2multi.py
```

**📝 Note:**
- **Option A users** (using `/ssd1/zhuoyuan/` path): You're done! No training script editing needed.
- **Option B users** (using `/ssd1/<your-username>/` path): Remember your custom path. You'll need to update training scripts in Section 6.
- **Replace `<your-username>@vllab15`** in all scp/rsync commands with YOUR actual username for SSH connection.

---

## 4. Model Setup

### Source Location
**Server:** vllab15
**Path:** `/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B`
**Size:** 43G

### Transfer Models

**Step 1:** Create local directory
```bash
# Create model directory
mkdir -p /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI  # Option A: Use this path (easiest)
# OR
# mkdir -p /ssd1/<your-username>/diffsynth_models/models/Wan-AI  # Option B: Use your username
```

**Step 2:** Transfer models (choose one method)

**Using rsync (recommended - can resume if interrupted):**
```bash
# Replace <your-username> with YOUR username for SSH connection
rsync -avz --progress \
  <your-username>@vllab15:/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/ \
  /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/
```

**Using scp (alternative):**
```bash
# Replace <your-username> with YOUR username for SSH connection
scp -r <your-username>@vllab15:/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/ \
  /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/
```

**Note:** Transfer takes 30-60 minutes depending on network speed.

### Verify Model Transfer

**Check folder exists:**
```bash
ls /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/
# Should show: diffusion_pytorch_model-*.safetensors, text_encoder/, vae/, wav2vec2-large-xlsr-53-english/
```

**Check size (most reliable verification):**
```bash
du -sh /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/
# Should show: 43G (or close to it)
```

If the size is correct, transfer was successful!

### Model Structure
After transfer, you should have:
```
/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/
├── diffusion_pytorch_model-00001-of-00004.safetensors  (7.8GB)
├── diffusion_pytorch_model-00002-of-00004.safetensors  (7.8GB)
├── diffusion_pytorch_model-00003-of-00004.safetensors  (7.8GB)
├── diffusion_pytorch_model-00004-of-00004.safetensors  (7.2GB)
├── text_encoder/
│   └── model.safetensors                                (11GB)
├── vae/
│   └── diffusion_pytorch_model.safetensors             (485MB)
└── wav2vec2-large-xlsr-53-english/
    └── pytorch_model.bin                                (1.2GB)
```

### Create Symlinks

**⚠️ CRITICAL: This symlink allows the training scripts to find models at `models/Wan-AI/`**

```bash
# Navigate to DiffSynth-Studio repo (wherever YOU cloned it)
cd ~/projects/DiffSynth-Studio  # Or your actual clone path

# Create symlink from repo's models/ to YOUR actual model storage
# Option A (easy - no script editing needed):
ln -s /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI

# Option B (custom paths):
# ln -s /ssd1/<your-username>/diffsynth_models/models/Wan-AI models/Wan-AI

# Verify symlink works
ls -lh models/Wan-AI/Wan2.2-S2V-14B/
# Should show all model files (diffusion_pytorch_model-*.safetensors, etc.)

# Check symlink target
readlink -f models/Wan-AI
# Should print: /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI
#    (or your custom path if using Option B)
```

**📝 Note:**
- **Option A users** (using `/ssd1/zhuoyuan/` path): You're done! Symlink is set up correctly.
- **Option B users** (using `/ssd1/<your-username>/` path): Make sure the symlink points to your actual storage path.

### Verify Setup (Important!)

**Check codebase size:**
```bash
cd ~/projects/DiffSynth-Studio  # Or wherever you cloned
du -sh .
# Should show: ~70M (or less than 1G)
```

**⚠️ If codebase is >1G:** Models were downloaded to the project directory instead of the symlink location! This means:
- Symlink might be broken
- HF_HOME might not be set correctly
- Models are wasting space in your home directory

**Fix:** Delete the models from project directory and verify symlink:
```bash
rm -rf models/Wan-AI/Wan2.2-S2V-14B/
ln -sf /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI  # Recreate symlink
```

### Optional: Test Inference Before Training

Before running training, you can test if inference works to verify your setup:

**File:** `test_inference_with_renderme360.py` (in project root)

**Edit these lines based on where YOU downloaded data:**

```python
# Line 29: Model path (where YOU downloaded models)
LOCAL_MODEL_PATH = "/ssd1/zhuoyuan/diffsynth_models/models"

# Line 30: Dataset path (where YOU downloaded dataset)
BASE_PATH = "/ssd1/zhuoyuan/renderme360_4cam"  # Change to YOUR dataset location

# Line 32: GPU device (change if needed)
DEVICE = "cuda:0"  # or "cuda:1", "cuda:2", etc.
```

**If you used default paths** (`/ssd1/zhuoyuan/`), the paths above should work as-is!

**Run inference test:**
```bash
python test_inference_with_renderme360.py
```

If inference works, your setup is correct!

---

## 5. Environment Variables

### Critical Configuration

**Add to YOUR `~/.bashrc` or set before each run:**

```bash
# HuggingFace cache location (IMPORTANT: Use local SSD, not home directory!)
# Option A (easy):
export HF_HOME=/ssd1/zhuoyuan/hf_cache

# Option B (custom):
# export HF_HOME=/ssd1/<your-username>/hf_cache

# CUDA home (needed for DeepSpeed, if using conda cuda-toolkit)
export CUDA_HOME=$CONDA_PREFIX

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false
```

**How to edit ~/.bashrc:**
```bash
# Open your bashrc in an editor
nano ~/.bashrc  # or vim, emacs, etc.

# Add the above export lines at the end (choose Option A or B)
# Save and exit

# Create the HF cache directory
mkdir -p /ssd1/zhuoyuan/hf_cache  # Option A
# OR
# mkdir -p /ssd1/<your-username>/hf_cache  # Option B
```

### Apply Changes
```bash
# Source bashrc to apply in current shell
source ~/.bashrc

# Verify environment variables are set
echo $HF_HOME
# Should print: /ssd1/<your-username>/hf_cache (or YOUR actual path)

echo $CUDA_HOME
# Should print: /home/<your-username>/miniconda3/envs/diffsynth-s2v (or YOUR conda path)
```

### Check Storage Quota
```bash
# Check home directory quota (if applicable)
quota -s

# Check available space on local SSD
df -h /ssd1/<your-username>/
```

---

## 6. Running Training

### Update Paths in Scripts

**⚠️ IF YOU USED OPTION A (EASY SETUP):** You can **SKIP THIS ENTIRE SECTION** and go directly to "Smoke Test" below! The scripts already have the correct paths.

**⚠️ IF YOU USED OPTION B (CUSTOM PATHS):** You MUST edit these scripts with YOUR actual paths before running.

These scripts currently use `/ssd1/zhuoyuan/` paths. If you used Option B with different paths, you need to update them.

#### Edit Script 1: Smoke Test (OPTION B ONLY)

**File:** `examples/wanvideo/model_training/lora/run_renderme360_test.sh`

Open the file:
```bash
cd ~/projects/DiffSynth-Studio  # Or wherever you cloned the repo
nano examples/wanvideo/model_training/lora/run_renderme360_test.sh
```

**Find and replace these variables (near the top of the file):**

1. **Line 17 - Dataset path:**
   ```bash
   # Find this line:
   DATASET_BASE="/ssd4/zhuoyuan/renderme360_4cam"

   # Replace with YOUR dataset path:
   DATASET_BASE="/ssd1/<your-username>/renderme360_4cam"
   # (or whatever storage path you chose in Section 3)
   ```

2. **Line 19 - Output path:**
   ```bash
   # Find this line:
   OUTPUT_DIR="/ssd1/zhuoyuan/diffsynth_training/renderme360_test"

   # Replace with YOUR output path:
   OUTPUT_DIR="/ssd1/<your-username>/diffsynth_training/renderme360_test"
   # (use same storage location as dataset for consistency)
   ```

**Save and exit** (Ctrl+O, Enter, Ctrl+X in nano)

#### Edit Script 2: Full Training

**File:** `examples/wanvideo/model_training/lora/run_renderme360_full.sh`

**Make the same two changes (near the top of the file):**

1. **Line 21 - Dataset path:**
   ```bash
   # Find this line:
   DATASET_BASE="/ssd4/zhuoyuan/renderme360_4cam"

   # Replace with YOUR dataset path:
   DATASET_BASE="/ssd1/<your-username>/renderme360_4cam"
   ```

2. **Line 23 - Output path:**
   ```bash
   # Find this line:
   OUTPUT_DIR="/ssd1/zhuoyuan/diffsynth_training/renderme360_lora"

   # Replace with YOUR output path:
   OUTPUT_DIR="/ssd1/<your-username>/diffsynth_training/renderme360_lora"
   ```

**📝 Summary of paths to update:**
- Dataset path: Use where YOU stored the dataset (Section 3)
- Output path: Use same storage location (not home directory!)
- Model path: Already handled by symlink (Section 4), no script changes needed

### Smoke Test (CRITICAL: Run This First!)

**Purpose:** Test with 100 samples to verify everything works before full training

```bash
cd ~/projects/DiffSynth-Studio

# Activate environment
conda activate diffsynth-s2v

# Check GPUs available
nvidia-smi
# Expected: All GPUs visible with most VRAM free

# Run smoke test
bash examples/wanvideo/model_training/lora/run_renderme360_test.sh
```

**Expected output:**
```
Loading models from: ['./models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00001-of-00004.safetensors', ...]
Dataset loaded: 100 samples
Training started...
Step 1: Loss = X.XXXX
Step 2: Loss = X.XXXX
...
Checkpoint saved: /ssd1/<your-username>/diffsynth_training/renderme360_test/checkpoint-100/
```

**What to watch for:**
- ✅ Models load successfully from symlinked paths
- ✅ Dataset loads 100 samples
- ✅ Training starts without OOM
- ✅ Loss values print (even if high initially)
- ✅ GPU memory usage stable (check with `watch -n 1 nvidia-smi`)
- ❌ Out of Memory → Report GPU specs and error
- ❌ NCCL timeout → Report network/GPU topology
- ❌ Loss = NaN → Report immediately

### Monitor Training
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f /ssd1/<your-username>/diffsynth_training/renderme360_test/train.log  # If logs are saved
```

### If Smoke Test Passes: Full Training

**Only run this if smoke test succeeds!**

```bash
bash examples/wanvideo/model_training/lora/run_renderme360_full.sh
```

**Specs:**
- 1,048 samples × 10 repeats = 10,480 samples/epoch
- 50 epochs
- Learning rate: 1e-4
- LoRA rank: 32
- Checkpoints every 10,000 steps
- Expected time: 24-48 hours (depending on GPU speed)

**Output location:**
```
/ssd1/<your-username>/diffsynth_training/renderme360_lora/
├── checkpoint-10000/
│   └── dit_lora.safetensors
├── checkpoint-20000/
│   └── dit_lora.safetensors
└── ...
```

---

## 7. What to Report Back

### If Successful ✅
```
✅ Smoke test passed!

**Environment:**
- Server: <hostname>
- GPUs: <count>× <model> (<VRAM>GB each)
- CUDA version: <version>

**Memory usage during training:**
- GPU 0: <XX>GB / <total>GB
- GPU 1: <XX>GB / <total>GB
- ...

**Training metrics:**
- Step 100 loss: <value>
- Training speed: <steps/sec or samples/sec>

**Checkpoints:**
- Location: <path>
- Size: <size>

Proceeding with full training...
```

### If Failed ❌
```
❌ Smoke test failed with <error type>

**Environment:**
- Server: <hostname>
- GPUs: <count>× <model> (<VRAM>GB each)
- CUDA version: <version>

**Error:**
<paste full error traceback>

**Memory usage before crash:**
<nvidia-smi output or memory stats>

**What was happening when it crashed:**
- Model loading ✓/✗
- Dataset loading ✓/✗
- First forward pass ✓/✗
- Backward pass ✓/✗
```

---

## 8. Troubleshooting

### OOM Even on 80GB GPUs
Try reducing LoRA rank in both scripts:
```bash
# Edit run_renderme360_test.sh and run_renderme360_full.sh
# Change line with --lora_rank:
--lora_rank 8  # Instead of 32
```

### NCCL Timeout
Check GPU topology and network:
```bash
nvidia-smi topo -m
```

### Models Not Found
Verify symlink:
```bash
ls -lh models/Wan-AI
readlink -f models/Wan-AI
```

### DeepSpeed CUDA Mismatch
Check PyTorch CUDA matches system CUDA:
```bash
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
nvcc --version  # System CUDA
```

Should match (both 12.4 or both 11.8).

### Dataset Loading Errors
Verify metadata CSV and folder structure:
```bash
head /ssd1/<your-username>/renderme360_4cam/metadata_single2multi.csv
ls /ssd1/<your-username>/renderme360_4cam/0001/
```

---

## 9. Alternative Configurations (If OOM Persists)

If training still OOMs on your GPUs, try these configurations:

### Lower LoRA Rank
```bash
# In both .sh scripts, change:
--lora_rank 4  # Reduces trainable params by 87.5%
```

### Fewer Frames (Smoke Test Only)
```bash
# In run_renderme360_test.sh, change:
--num_frames 9  # Instead of 17 (4×2+1)
```

### Smaller Resolution (Not Recommended)
```bash
# In run_renderme360_test.sh, change:
--height 224 --width 416  # Already at minimum reasonable size
```

---

## 10. Expected Timeline

**Smoke Test:**
- Setup: 1-2 hours (dataset + model transfer)
- Execution: 10-30 minutes (100 samples)

**Full Training:**
- 50 epochs × 10,480 samples
- Estimate: 24-48 hours on A100 80GB
- Checkpoint every 10,000 steps (~1 epoch)

---

## 11. After Training: Validation

After training completes, validate the LoRA:

```bash
# Copy validation script from inference examples
# (We'll need to create this - TBD)

# Expected: Generate test videos with trained LoRA
# Check: Lip sync quality, novel view quality
```

---

## Files Checklist for Transfer

**Code (from Git):**
- ✓ Already in repository after checkout

**Dataset:**
- [ ] `/ssd4/zhuoyuan/renderme360_4cam/` → Transfer to your `/ssd1/<your-username>/renderme360_4cam/`

**Models:**
- [ ] `/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/` → Transfer to your `/ssd1/<your-username>/diffsynth_models/models/Wan-AI/Wan2.2-S2V-14B/`

**Environment:**
- ✓ `environment_renderme360.yml` (in repo)
- ✓ `requirements_renderme360.txt` (in repo)

---

## Contact & Data Transfer

### Getting Dataset/Model Access

The dataset and models are hosted on a private server. Contact the original developer privately (NOT via GitHub issues) for:

1. **Server access details:**
   - Server hostname or IP address
   - Your SSH credentials (if needed)
   - Dataset source path
   - Model source path

2. **Alternative transfer methods:**
   - If direct rsync isn't possible, can arrange:
     - Shared cloud storage link
     - Physical drive transfer
     - Other institutional transfer methods

**Security Note:** Server details are NOT in this public GitHub repo for security reasons. You'll receive them via email/Slack/direct message.

### If You Encounter Issues

Provide the following information:
1. Full error traceback
2. `nvidia-smi` output 
3. Environment details (GPU model, CUDA version, PyTorch version)
4. What step failed (model loading, dataset loading, first forward pass, etc.)
5. Which option you used (Option A easy setup or Option B custom paths)

