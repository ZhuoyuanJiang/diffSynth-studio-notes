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
