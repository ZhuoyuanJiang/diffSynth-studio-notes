"""
Generate metadata CSV for RenderMe360 S2V training.

Usage:
    python scripts/generate_metadata_single2multi.py

Output:
    /ssd2/zhuoyuan/renderme360_4cam/metadata_single2multi.csv
"""

import pandas as pd
from pathlib import Path


def count_frames(images_dir, camera="cam_28"):
    """Count frames in a camera directory."""
    cam_dir = images_dir / camera
    if not cam_dir.exists():
        return 0
    return len(list(cam_dir.glob("*.jpg")))


def generate_metadata(
    base_path="/ssd2/zhuoyuan/renderme360_4cam/",
    output_csv=None,
    num_frames=81,
    stride=40,
    reference_camera="cam_28"
):
    """
    Generate metadata CSV for RenderMe360 S2V training.

    Args:
        base_path: Base path to RenderMe360 dataset
        output_csv: Output CSV path (default: base_path/metadata_single2multi.csv)
        num_frames: Number of 16fps frames per clip (81)
        stride: Stride between clips in 30fps frames (40)
        reference_camera: Camera to use for frame counting
    """
    base = Path(base_path)

    if output_csv is None:
        output_csv = base / "metadata_single2multi.csv"

    rows = []

    print(f"Scanning {base} for subjects and performances...")

    # Scan all subjects
    subjects = sorted([d for d in base.iterdir() if d.is_dir() and d.name.isdigit()])

    for subject_dir in subjects:
        subject_id = subject_dir.name

        # Scan all performances
        performances = sorted([d for d in subject_dir.iterdir()
                              if d.is_dir() and d.name.startswith("s")])

        for perf_dir in performances:
            performance = perf_dir.name
            images_dir = perf_dir / "images"

            if not images_dir.exists():
                print(f"  Skipping {subject_id}/{performance}: no images/ directory")
                continue

            # Count frames in reference camera
            total_frames_30fps = count_frames(images_dir, reference_camera)

            if total_frames_30fps == 0:
                print(f"  Skipping {subject_id}/{performance}: no frames in {reference_camera}")
                continue

            # Calculate how many 81-frame clips we can extract
            # Last selected frame is at: start + ((80*30 + 8)//16) = start + 150
            # So we need: start_frame_30fps + 150 < total_frames_30fps
            max_start = total_frames_30fps - 151

            if max_start < 0:
                print(f"  Skipping {subject_id}/{performance}: only {total_frames_30fps} frames (need at least 151)")
                continue

            # Generate clips with stride
            start_frame = 0
            clip_count = 0

            while start_frame <= max_start:
                rows.append({
                    "subject": subject_id,
                    "performance": performance,
                    "start_frame_30fps": start_frame,
                    "num_frames": num_frames,
                    "audio_path": f"{subject_id}/{performance}/audio/audio.mp3",
                    "input_camera": "cam_28",
                    "prompt": "a person speaking",
                })
                start_frame += stride
                clip_count += 1

            print(f"  {subject_id}/{performance}: {total_frames_30fps} frames → {clip_count} clips")

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Generated {len(df)} training samples")
    print(f"✅ Saved to: {output_csv}")

    # Print summary statistics
    subjects_count = df['subject'].nunique()
    performances_count = len(df.groupby(['subject', 'performance']))
    avg_clips_per_perf = len(df) / performances_count if performances_count > 0 else 0

    print(f"\nSummary:")
    print(f"  Subjects: {subjects_count}")
    print(f"  Performances: {performances_count}")
    print(f"  Avg clips per performance: {avg_clips_per_perf:.1f}")
    print(f"  Total training samples: {len(df)}")

    return df


if __name__ == "__main__":
    df = generate_metadata(
        base_path="/ssd2/zhuoyuan/renderme360_4cam/",
        output_csv="/ssd2/zhuoyuan/renderme360_4cam/metadata_single2multi.csv",
        num_frames=81,
        stride=150  # 150 frames at 30fps = 5.0s, zero overlap between clips
    )
