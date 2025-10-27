"""
Test RenderMe360 Custom Data Operators
Validates 2×2 grid creation with small batch
"""

import sys
import torch
from pathlib import Path
from diffsynth.trainers.renderme360_unified_dataset import RenderMe360UnifiedDataset

# Configuration
BASE_PATH = "/ssd4/zhuoyuan/renderme360_4cam"
METADATA_CSV = f"{BASE_PATH}/metadata_single2multi.csv"

print("=" * 80)
print("Testing RenderMe360 Custom Data Operators")
print("=" * 80)
print(f"Base path: {BASE_PATH}")
print(f"Metadata: {METADATA_CSV}")
print()

# Check files exist
if not Path(BASE_PATH).exists():
    print(f"❌ ERROR: Dataset path not found: {BASE_PATH}")
    sys.exit(1)

if not Path(METADATA_CSV).exists():
    print(f"❌ ERROR: Metadata CSV not found: {METADATA_CSV}")
    sys.exit(1)

print("✓ Paths verified")
print()

# Create dataset with custom operators
print("Creating RenderMe360UnifiedDataset...")
dataset = RenderMe360UnifiedDataset(
    base_path=BASE_PATH,
    metadata_path=METADATA_CSV,
    repeat=1,
)

print(f"✓ Dataset created!")
print()

# Test loading first 3 samples
print("=" * 80)
print("Testing Data Loading (first 3 samples)")
print("=" * 80)

for i in range(min(3, len(dataset))):
    print(f"\n--- Sample {i} ---")
    try:
        data = dataset[i]

        # Validate video
        video = data.get("video", [])
        if isinstance(video, list) and len(video) > 0:
            print(f"✓ Video: {len(video)} frames")
            # Check first frame
            first_frame = video[0]
            print(f"  - Frame size: {first_frame.size}")
            if first_frame.size == (832, 448):
                print(f"  - ✓ Correct 2×2 grid size (832×448)")
            else:
                print(f"  - ❌ Wrong size! Expected (832, 448)")
        else:
            print(f"❌ Video: Invalid or empty")

        # Validate input_image
        input_image = data.get("input_image")
        if input_image:
            print(f"✓ Input image: {input_image.size}")
            if input_image.size == (416, 224):
                print(f"  - ✓ Correct input size (416×224)")
            else:
                print(f"  - ❌ Wrong size! Expected (416, 224)")
        else:
            print(f"❌ Input image: Missing")

        # Validate audio
        input_audio = data.get("input_audio")
        if input_audio is not None:
            print(f"✓ Audio: {len(input_audio)} samples")
            expected_samples = int(81 / 16.0 * 16000)  # 81 frames at 16fps = 5.0625s at 16kHz
            print(f"  - Expected: ~{expected_samples} samples")
            if abs(len(input_audio) - expected_samples) < 1000:
                print(f"  - ✓ Correct audio length")
            else:
                print(f"  - ⚠️  Audio length mismatch")
        else:
            print(f"❌ Audio: Missing")

        # Validate prompt
        prompt = data.get("prompt", "")
        print(f"✓ Prompt: \"{prompt}\"")

    except Exception as e:
        print(f"❌ ERROR loading sample {i}:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print()
print("=" * 80)
print("Saving Sample Visualizations")
print("=" * 80)

# Save first frame of first sample for visual inspection
try:
    sample = dataset[0]
    video = sample.get("video", [])
    input_image = sample.get("input_image")

    if video:
        # Save first frame of 2×2 grid
        output_grid = "test_output_grid_frame0.png"
        video[0].save(output_grid)
        print(f"✓ Saved 2×2 grid (frame 0): {output_grid}")

    if input_image:
        # Save input image
        output_input = "test_output_input_image.png"
        input_image.save(output_input)
        print(f"✓ Saved input image: {output_input}")

except Exception as e:
    print(f"❌ ERROR saving visualizations:")
    print(f"   {type(e).__name__}: {e}")

print()
print("=" * 80)
print("✓ TESTING COMPLETE!")
print("=" * 80)
print()
print("Next steps:")
print("1. Check saved images:")
print("   - test_output_grid_frame0.png (should show 2×2 grid with 4 camera views)")
print("   - test_output_input_image.png (should show single cam_54 view)")
print("2. If images look correct, operators are working!")
print("3. Ready to integrate with training script")
