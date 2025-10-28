# Server Change Checklist

Quick checklist when switching between vllab servers (vllab9, vllab14, etc.)

## 1. Verify Current Server

```bash
hostname  # Should show vllab9, vllab14, etc.
nvidia-smi  # Check GPU availability
```

## 2. Check Dataset Location

**Known locations:**
- vllab9: `/ssd2/zhuoyuan/renderme360_4cam/`
- vllab14: `/ssd4/zhuoyuan/renderme360_4cam/`
- vllab15: `/ssd4/zhuoyuan/renderme360_4cam/`

```bash
# Check which SSD the dataset is on
ls -la /ssd2/zhuoyuan/renderme360_4cam/ 2>/dev/null || \
ls -la /ssd4/zhuoyuan/renderme360_4cam/ 2>/dev/null

# Set variable for this session
export DATASET_BASE=/ssd4/zhuoyuan/renderme360_4cam  # Adjust as needed
```

## 3. Verify Model Storage & Symlinks

```bash
# Check HF_HOME (should be on local SSD, NOT home directory)
echo $HF_HOME  # Should be /ssd1/zhuoyuan/hf_cache or similar

# Check model symlink
ls -lh models/Wan-AI  # Should point to local SSD

# If symlink is broken or points to wrong server's storage:
# Find where models are stored on this server
ls -la /ssd*/zhuoyuan/diffsynth_models/models/Wan-AI 2>/dev/null

# Update symlink (replace /ssd2 with correct path)
ln -sf /ssd2/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI
```

## 4. Generate Metadata CSV (First Time on New Server)

```bash
# Check if metadata exists
ls -la $DATASET_BASE/metadata*.csv

# If metadata doesn't exist, generate it
# (We'll create this script later)
python scripts/generate_metadata_upstream.py --base_path $DATASET_BASE
```

## 5. Verify Conda Environment

```bash
# Activate environment
source ~/miniconda3/bin/activate diffsynth-s2v

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 6. Test Quick Sanity Check

```bash
# Run a quick dataset test (we'll create this script)
python scripts/test_dataset_loading.py --base_path $DATASET_BASE --num_samples 1
```

## Server-Specific Paths Summary

| Server | Dataset | Models | HF Cache |
|--------|---------|--------|----------|
| vllab9 | `/ssd2/zhuoyuan/renderme360_4cam` | `/ssd2/zhuoyuan/diffsynth_models` | `/ssd1/zhuoyuan/hf_cache` |
| vllab14 | `/ssd4/zhuoyuan/renderme360_4cam` | TBD | `/ssd1/zhuoyuan/hf_cache` (verify) |

## What's Shared (No Action Needed)

- ✅ Code in `~/projects/DiffSynth-Studio` (on NAS, shared across servers)
- ✅ Conda environments in `~/miniconda3` (on NAS, shared)
- ✅ Git branches (shared)
- ✅ All Python scripts (shared)

## What's Server-Specific (Needs Setup)

- ⚠️ Dataset location (`/ssd2` vs `/ssd4`)
- ⚠️ Model symlinks
- ⚠️ Generated metadata CSV files
- ⚠️ Training output directories

## Quick Setup Script (Copy-Paste Template)

```bash
# Set dataset path for this server
export DATASET_BASE=/ssd4/zhuoyuan/renderme360_4cam  # CHANGE THIS

# Verify dataset exists
ls $DATASET_BASE/0026/s1_all/images/cam_54/frame_000000.jpg

# Check/update model symlink
readlink models/Wan-AI  # Should point to this server's SSD

# Generate metadata if needed
if [ ! -f "$DATASET_BASE/metadata_upstream.csv" ]; then
    echo "Metadata missing - need to generate"
    # python scripts/generate_metadata_upstream.py --base_path $DATASET_BASE
fi

# Verify HF_HOME
echo "HF_HOME: $HF_HOME"  # Should NOT be in /home/

# Test GPU
nvidia-smi

echo "✅ Ready to run on $(hostname)"
```

## Notes

- **Always use `--dataset_base_path` argument** in training commands (never hardcode paths)
- **Check GPU availability** with `nvidia-smi` before running
- **Home directory quota:** 100GB limit - keep only code there!
- **Large files go on local SSDs:** `/ssd1/`, `/ssd2/`, `/ssd4/`, etc.



--- 
---

Example we used before:

### 2. Models Setup

**Location (vllab14):** `/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI/`

**Symlinked:** `models/Wan-AI` → `/ssd1/zhuoyuan/diffsynth_models/models/Wan-AI`

**Contents (43GB total):**
```
Wan-AI/
├── Wan2.2-S2V-14B/                      # S2V-specific (32GB)
│   ├── diffusion_pytorch_model-00001-of-00004.safetensors  # 9.3GB
│   ├── diffusion_pytorch_model-00002-of-00004.safetensors  # 9.3GB
│   ├── diffusion_pytorch_model-00003-of-00004.safetensors  # 9.3GB
│   ├── diffusion_pytorch_model-00004-of-00004.safetensors  # 2.6GB
│   └── wav2vec2-large-xlsr-53-english/
│       └── model.safetensors                                # 1.2GB
│
└── Wan2.1-T2V-1.3B/                     # Shared components (11GB)
    ├── models_t5_umt5-xxl-enc-bf16.pth                      # 11GB (T5 text encoder)
    └── Wan2.1_VAE.pth                                        # 485MB (VAE)
```

**How it was set up on vllab14:**
```bash
# Copied from vllab9
scp -r <vllab9-ip>:/ssd2/zhuoyuan/diffsynth_models /ssd1/zhuoyuan/

# Find IP address
hostname -I 

# Created symlink
cd ~/projects/DiffSynth-Studio
ln -sf /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI
```

**Note:** On vllab9, models were at `/ssd2/zhuoyuan/diffsynth_models/`. On vllab14, they are at `/ssd1/zhuoyuan/diffsynth_models/`. Update paths when switching servers!

---

### 3. Dataset Structure

**Location:** `/ssd4/zhuoyuan/renderme360_4cam/`

**Directory Structure:**
```
/ssd4/zhuoyuan/renderme360_4cam/
├── 0026/                       # Subject ID (21 subjects total)
│   ├── s1_all/                 # Performance (6 per subject)
│   │   ├── images/
│   │   │   ├── cam_28/         # Camera views (4 cameras)
│   │   │   │   ├── frame_000000.jpg
│   │   │   │   ├── frame_000001.jpg
│   │   │   │   └── ... (2529 frames @ 30fps)
│   │   │   ├── cam_37/
│   │   │   ├── cam_49/
│   │   │   └── cam_54/         # Input camera (frontal view)
│   │   ├── audio/
│   │   │   └── audio.mp3       # Full performance audio @ 16kHz
│   │   ├── masks/
│   │   ├── keypoints3d/
│   │   └── metadata/
│   ├── s2_all/
│   └── ... (s3_all through s6_all)
├── 0041/, 0048/, ..., 0297/    # Other subjects
└── metadata_single2multi.csv   # ✅ Generated (1048 training samples)
```

**Metadata CSV Format:**
```csv
subject,performance,start_frame_30fps,num_frames,audio_path,input_camera,prompt
0026,s1_all,0,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
0026,s1_all,150,81,0026/s1_all/audio/audio.mp3,cam_54,a person speaking
...
```

**Statistics:**
- **21 subjects**
- **126 performances** (6 per subject)
- **1,048 training clips**
- **Each clip:** 81 frames @ 16fps = 5.0625s
- **Frame sampling:** Uses nearest-frame rounding from 30fps source to 16fps output
- **No overlap** between clips (stride=150 frames @ 30fps)

**How metadata was generated:**
```bash
python -c "
from scripts.generate_metadata_single2multi import generate_metadata
generate_metadata(
    base_path='/ssd4/zhuoyuan/renderme360_4cam/',
    output_csv='/ssd4/zhuoyuan/renderme360_4cam/metadata_single2multi.csv',
    num_frames=81,
    stride=150
)
"
```

---