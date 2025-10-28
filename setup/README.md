# RenderMe360 Training Setup - Quick Start Guide

**Purpose:** Fine-tune Wan2.2-S2V-14B on RenderMe360 dataset for speech-to-video with novel view synthesis

**Challenge:** Out of Memory (OOM) on 48GB GPUs. We need to test on higher VRAM GPUs (A100 80GB recommended).

**Time:** ~2 hours setup (mostly data transfer)

---

## 📋 Two Setup Options (10/27/2025: For now, just go to option 2, option 1 is not well tested, no need to look at Overview, just go to MANUAL_SETUP.md)

### Option 1: Automated Setup (Recommended) ⭐
Follow this guide - edit 1 config file, run 3 scripts, done!

### Option 2: Manual Setup (Fallback)
If the automated scripts don't work, see **[MANUAL_SETUP.md](MANUAL_SETUP.md)** for step-by-step manual instructions.

---

## 📋 Overview (Automated Setup)

This setup involves **4 simple steps**:

1. **Edit config** (1 file, 2 lines) ← YOU ONLY EDIT ONCE HERE
2. **Setup environment** (run 1 script)
3. **Transfer data** (run 1 script, takes ~1 hour)
4. **Run training** (run existing scripts)

---

## Step 1: Edit Configuration (5 minutes)

### 1.1 Get Server Info from Developer

Contact the original developer and ask for:
- **Source server IP/hostname** (e.g., `192.168.1.100` or `vllab15.example.com`)

That's it! Just one piece of info.

### 1.2 Edit config_renderme360.sh

Open the config file:
```bash
cd setup/
nano config_renderme360.sh  # or vim, code, etc.
```

**Edit line 14:** Paste the server IP/hostname you got from developer
```bash
SOURCE_SERVER="PASTE_SERVER_IP_HERE"  # ← Change this
```

**Edit line 38 (optional):** Choose your storage location
```bash
LOCAL_STORAGE_BASE="/ssd1/zhuoyuan"  # ← Keep as-is (recommended) or change
```

**Recommended:** Keep `LOCAL_STORAGE_BASE="/ssd1/zhuoyuan"` as-is. This way you don't need to edit any other files!

**If /ssd1 doesn't exist on your server:**
- Check what you have: `df -h`
- Update `LOCAL_STORAGE_BASE` to your actual storage (e.g., `/ssd2/zhuoyuan`, `/tmp2/zhuoyuan`, `/data/zhuoyuan`)
- The config file has detailed comments explaining this

**Save and exit.**

---

## Step 2: Setup Environment (5 minutes)

Run the setup script to create directories and symlinks:

```bash
cd setup/
bash setup_renderme360.sh
```

This will:
- ✓ Create all necessary directories
- ✓ Create model symlink
- ✓ Add environment variables to ~/.bashrc

After it finishes, apply environment variables:
```bash
source ~/.bashrc
```

---

## Step 3: Transfer Data (~1-2 hours)

### Transfer Dataset and Models

```bash
cd setup/
bash transfer_renderme360_data.sh
```

This will:
- Transfer dataset (~50GB, 30-60 minutes)
- Transfer models (~43GB, 30-60 minutes)
- Verify all files are complete

**Note:** The script will connect to the source server and transfer files via rsync. You may be prompted to confirm the SSH connection on first connection.

---

## Step 4: Install Conda Environment (10 minutes)

```bash
cd ..  # Back to project root

# Create conda environment
conda env create -f environment_renderme360.yml

# Activate environment
conda activate diffsynth-s2v

# Install DiffSynth-Studio
pip install -e .

# Verify installation
python -c "import diffsynth; print('✓ DiffSynth installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

### Verify Setup (Important!)

**Check codebase size:**
```bash
du -sh .
# Should show: ~70M (or less than 1G)
```

**⚠️ If codebase is >1G:** Models were downloaded to the project directory instead of the symlink location!

**Fix:**
```bash
rm -rf models/Wan-AI/Wan2.2-S2V-14B/
ln -sf /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI  # Recreate symlink
```

### Optional: Test Inference Before Training

You can verify your setup by testing inference:

**File:** `test_inference_with_renderme360.py`

**Edit these lines based on where YOU downloaded data:**
```python
# Line 29: Model path (where YOU downloaded models)
LOCAL_MODEL_PATH = "/ssd1/zhuoyuan/diffsynth_models/models"

# Line 30: Dataset path (where YOU downloaded dataset)
BASE_PATH = "/ssd1/zhuoyuan/renderme360_4cam"  # Change to YOUR dataset location

# Line 32: GPU device (change if needed)
DEVICE = "cuda:0"  # or "cuda:1", "cuda:2", etc.
```

**Run:**
```bash
python test_inference_with_renderme360.py
```

If inference works, your setup is correct!

---

## Step 5: Run Training

### 5.1 Smoke Test (ALWAYS RUN THIS FIRST!)

Test with 100 samples, 1 epoch (~10-30 minutes):

```bash
# From project root
bash examples/wanvideo/model_training/lora/run_renderme360_test.sh
```

**Watch for:**
- ✅ Models load successfully
- ✅ Dataset loads (100 samples)
- ✅ Training starts without OOM
- ✅ Loss values print
- ❌ If OOM: Report GPU model and VRAM to developer

### 5.2 Full Training (if smoke test passes)

```bash
bash examples/wanvideo/model_training/lora/run_renderme360_full.sh
```

**Specs:**
- 1,048 samples × 10 repeats
- 50 epochs
- Expected time: 24-48 hours
- Checkpoints saved every 10,000 steps

**Monitor training:**
```bash
# In another terminal
watch -n 1 nvidia-smi  # Monitor GPU usage
```

---

## 📁 What Got Created

After setup, your directory structure:

```
/ssd1/zhuoyuan/                          # Your storage base
├── renderme360_4cam/                    # Dataset (50GB)
│   ├── metadata_single2multi.csv
│   ├── 0001/, 0002/, ... (subjects)
├── diffsynth_models/models/Wan-AI/      # Models (43GB)
│   └── Wan2.2-S2V-14B/
│       ├── diffusion_pytorch_model-*.safetensors
│       ├── text_encoder/
│       ├── vae/
│       └── wav2vec2-large-xlsr-53-english/
├── hf_cache/                            # HuggingFace cache
└── diffsynth_training/                  # Training outputs
    ├── renderme360_test/                # Smoke test output
    └── renderme360_lora/                # Full training output
```

And in the project root:
```
models/Wan-AI → /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI  (symlink)
```

---

## 🚨 Troubleshooting

### "Permission denied" when creating /ssd1/zhuoyuan

If you don't have write access to /ssd1/:
1. Check what storage you DO have access to: `df -h`
2. Edit `setup/config_renderme360.sh` and change `LOCAL_STORAGE_BASE` to a path you can write to
   - Examples: `/ssd2/zhuoyuan`, `/tmp2/zhuoyuan`, `/data/zhuoyuan`, etc.
3. Run `bash setup/setup_renderme360.sh` again

### "ssh: connect to host ... port 22: Connection refused"

- Check SOURCE_SERVER in config is correct
- Ask developer to verify server is accessible
- Try pinging: `ping <SERVER_IP>`

### OOM during training

Try reducing LoRA rank:
```bash
# Edit both training scripts:
nano examples/wanvideo/model_training/lora/run_renderme360_test.sh
nano examples/wanvideo/model_training/lora/run_renderme360_full.sh

# Find and change:
--lora_rank 32  →  --lora_rank 8
```

### Dataset or models not found during training

Check symlink:
```bash
ls -lh models/Wan-AI
# Should show: models/Wan-AI -> /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI

# If broken, recreate:
cd <project-root>
rm models/Wan-AI  # Remove broken symlink
ln -s /ssd1/zhuoyuan/diffsynth_models/models/Wan-AI models/Wan-AI
```

---

## ✅ Checklist

Before running training, verify:

```bash
☐ Edited config_renderme360.sh with correct SOURCE_SERVER
☐ Ran setup_renderme360.sh successfully
☐ Ran transfer_renderme360_data.sh (dataset + models transferred)
☐ source ~/.bashrc applied
☐ conda env create completed
☐ pip install -e . completed
☐ Symlink exists: ls models/Wan-AI shows model files
☐ Dataset exists: ls /ssd1/zhuoyuan/renderme360_4cam/0001
☐ nvidia-smi shows all GPUs available
```

---

## 📞 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Contact developer with:
   - Full error message
   - Output of `nvidia-smi`
   - Which step failed
   - Your GPU model and VRAM

---

**Good luck! 🚀**

Training working? Let the developer know the smoke test passed and share your GPU specs!
