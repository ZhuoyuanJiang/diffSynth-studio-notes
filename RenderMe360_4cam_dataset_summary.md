# RenderMe360 4-Camera Dataset Summary

**Generated:** 2025-10-22
**Location:** `/ssd2/zhuoyuan/renderme360_4cam/`

## Dataset Overview

- **Total Subjects:** 21
- **Performances per Subject:** 6 (s1_all to s6_all)
- **Cameras per Performance:** 4 (downsampled from 20-camera configuration)
- **Image Resolution:** 2448x2048 (uniform across all data)
- **Total Dataset Size:** 440.4 GB (~20% of full 20-camera dataset)

## Extraction Configuration

This is a storage-optimized version extracted from the full RenderMe360 dataset.

- **Camera Selection:** 4 cameras strategically selected for diverse viewpoints
  - Camera IDs: [28, 37, 49, 54]
  - cam_28: Wide front context view
  - cam_37: Profile view (right side)
  - cam_49: Rear hemisphere view
  - cam_54: Front hemisphere view

- **Modalities Extracted:**
  - RGB Images (2448x2048)
  - Segmentation Masks
  - Audio (MP3 format)
  - Camera Calibration
  - 3D Keypoints
  - Metadata

## Subject Details

| Subject ID | Total Size (GB) | s1_all | s2_all | s3_all | s4_all | s5_all | s6_all |
|------------|----------------|---------|---------|---------|---------|---------|---------|
| 0026 | 26.8 | 7.4 | 4.4 | 4.1 | 4.1 | 3.7 | 3.1 |
| 0041 | 15.5 | 4.1 | 2.4 | 2.3 | 2.5 | 2.4 | 1.8 |
| 0048 | 20.1 | 5.3 | 3.0 | 3.0 | 3.1 | 3.3 | 2.4 |
| 0094 | 18.6 | 3.4 | 3.3 | 3.5 | 4.0 | 2.4 | 2.0 |
| 0099 | 16.6 | 2.7 | 2.8 | 3.3 | 3.6 | 2.3 | 1.9 |
| 0100 | 16.1 | 3.6 | 2.5 | 2.5 | 2.7 | 3.1 | 1.7 |
| 0116 | 28.8 | 9.0 | 3.9 | 4.2 | 4.6 | 4.0 | 3.1 |
| 0156 | 27.7 | 6.4 | 4.6 | 5.0 | 4.5 | 4.4 | 2.8 |
| 0168 | 14.1 | 3.6 | 2.1 | 2.2 | 2.1 | 2.3 | 1.8 |
| 0175 | 24.2 | 4.8 | 4.2 | 3.9 | 3.6 | 4.3 | 3.4 |
| 0189 | 14.9 | 4.2 | 2.3 | 2.3 | 2.2 | 2.3 | 1.6 |
| 0195 | 21.1 | 6.6 | 3.1 | 3.0 | 2.8 | 3.2 | 2.4 |
| 0232 | 18.9 | 5.4 | 2.6 | 3.0 | 2.9 | 2.9 | 2.1 |
| 0250 | 13.2 | 4.0 | 2.0 | 1.9 | 2.0 | 2.0 | 1.3 |
| 0253 | 18.7 | 5.6 | 2.7 | 2.8 | 2.8 | 2.7 | 2.1 |
| 0259 | 17.9 | 4.7 | 2.9 | 2.6 | 2.6 | 2.9 | 2.2 |
| 0262 | 28.0 | 7.4 | 4.0 | 4.2 | 4.2 | 4.6 | 3.6 |
| 0278 | 20.2 | 2.9 | 4.3 | 3.9 | 4.3 | 2.5 | 2.3 |
| 0290 | 16.7 | 3.3 | 3.3 | 3.1 | 2.9 | 2.2 | 1.9 |
| 0295 | 26.9 | 5.1 | 4.5 | 5.8 | 5.6 | 3.1 | 2.8 |
| 0297 | 35.4 | 6.3 | 6.3 | 6.6 | 7.5 | 4.7 | 4.0 |


## Performance Details (Audio Duration in seconds)

Audio durations remain the same as the full dataset (independent of camera count):

| Subject ID | s1_all | s2_all | s3_all | s4_all | s5_all | s6_all |
|------------|---------|---------|---------|---------|---------|---------|
| 0026 | 84.3s | 51.3s | 47.5s | 47.3s | 46.0s | 34.3s |
| 0041 | 54.6s | 30.0s | 29.6s | 33.0s | 31.4s | 23.6s |
| 0048 | 68.6s | 38.4s | 38.9s | 40.4s | 42.8s | 31.1s |
| 0094 | 43.6s | 42.4s | 44.8s | 51.2s | 30.8s | 25.2s |
| 0099 | 40.4s | 42.6s | 50.1s | 54.5s | 34.5s | 28.8s |
| 0100 | 51.3s | 36.4s | 35.9s | 38.2s | 47.0s | 26.4s |
| 0116 | 81.6s | 37.4s | 39.2s | 43.2s | 37.6s | 29.5s |
| 0156 | 69.9s | 49.7s | 53.5s | 48.7s | 48.1s | 30.9s |
| 0168 | 47.1s | 28.3s | 29.5s | 27.9s | 30.0s | 22.9s |
| 0175 | 47.7s | 42.3s | 39.3s | 37.0s | 43.8s | 35.4s |
| 0189 | 66.2s | 35.9s | 36.8s | 34.3s | 35.8s | 25.4s |
| 0195 | 66.8s | 31.1s | 30.6s | 28.8s | 32.8s | 24.2s |
| 0232 | 64.2s | 30.9s | 35.0s | 34.1s | 33.6s | 24.5s |
| 0250 | 72.6s | 37.7s | 34.7s | 36.4s | 36.1s | 24.6s |
| 0253 | 65.6s | 31.3s | 32.5s | 32.8s | 31.7s | 24.6s |
| 0259 | 74.7s | 46.0s | 41.6s | 42.5s | 45.8s | 35.3s |
| 0262 | 56.2s | 30.4s | 32.2s | 31.8s | 34.6s | 26.0s |
| 0278 | 44.0s | 66.6s | 58.2s | 63.5s | 38.6s | 33.6s |
| 0290 | 53.8s | 53.4s | 49.9s | 48.2s | 36.6s | 31.2s |
| 0295 | 90.1s | 80.2s | 104.3s | 101.6s | 55.9s | 49.6s |
| 0297 | 67.7s | 68.4s | 70.9s | 81.0s | 51.1s | 43.0s |


## Frame Counts per Performance

Frame counts remain the same as the full dataset (independent of camera count):

| Subject ID | s1_all | s2_all | s3_all | s4_all | s5_all | s6_all |
|------------|---------|---------|---------|---------|---------|---------|
| 0026 | 2529 | 1536 | 1425 | 1417 | 1379 | 1028 |
| 0041 | 1636 | 898 | 886 | 988 | 942 | 707 |
| 0048 | 2058 | 1152 | 1165 | 1212 | 1284 | 932 |
| 0094 | 1308 | 1271 | 1342 | 1534 | 925 | 754 |
| 0099 | 1212 | 1277 | 1503 | 1635 | 1034 | 864 |
| 0100 | 1539 | 1091 | 1075 | 1144 | 1408 | 793 |
| 0116 | 2448 | 1120 | 1174 | 1295 | 1127 | 883 |
| 0156 | 2096 | 1490 | 1605 | 1462 | 1443 | 925 |
| 0168 | 1411 | 849 | 885 | 838 | 899 | 688 |
| 0175 | 1429 | 1267 | 1179 | 1109 | 1311 | 1060 |
| 0189 | 1986 | 1076 | 1102 | 1028 | 1073 | 762 |
| 0195 | 2003 | 931 | 918 | 862 | 983 | 726 |
| 0232 | 1925 | 927 | 1049 | 1021 | 1005 | 733 |
| 0250 | 2177 | 1130 | 1039 | 1091 | 1083 | 736 |
| 0253 | 1968 | 937 | 976 | 983 | 950 | 738 |
| 0259 | 2241 | 1380 | 1247 | 1275 | 1375 | 1057 |
| 0262 | 1684 | 911 | 964 | 953 | 1035 | 780 |
| 0278 | 1318 | 1997 | 1743 | 1903 | 1156 | 1007 |
| 0290 | 1614 | 1603 | 1495 | 1447 | 1098 | 936 |
| 0295 | 2704 | 2405 | 3126 | 3046 | 1675 | 1485 |
| 0297 | 2030 | 2049 | 2126 | 2430 | 1532 | 1290 |


## Summary Statistics

### Audio Duration Statistics
- **Average Duration:** 44.2 seconds
- **Min Duration:** 22.9 seconds
- **Max Duration:** 104.3 seconds
- **Total Audio:** 1.5 hours

### Frame Count Statistics
- **Average Frames per Performance:** 1324
- **Min Frames:** 688
- **Max Frames:** 3126
- **Total Frames in Dataset:** 166,911 frames
- **Total Images (4 cameras):** 667,644 images (vs. 3,338,220 in full dataset)

### Storage Distribution
- **Smallest Subject:** 0250 (13.2 GB)
- **Largest Subject:** 0297 (35.4 GB)
- **Average Subject Size:** 21.0 GB
- **Storage Reduction:** ~80% compared to full 20-camera dataset (440 GB vs 2153 GB)

## Dataset Structure

```
/ssd4/zhuoyuan/renderme360_4cam/
└── {subject_id}/          # 21 subjects (0026, 0041, ..., 0297)
    └── {s1-s6}_all/       # 6 performances per subject
        ├── audio/         # MP3 audio files
        ├── calibration/   # Camera calibration matrices
        ├── images/        # RGB images for 4 cameras
        │   ├── cam_28/
        │   ├── cam_37/
        │   ├── cam_49/
        │   └── cam_54/
        ├── keypoints3d/   # 3D facial keypoints
        ├── masks/         # Segmentation masks for 4 cameras
        │   ├── cam_28/
        │   ├── cam_37/
        │   ├── cam_49/
        │   └── cam_54/
        └── metadata/      # Additional metadata
```

## Camera Configuration Details

The 4 cameras were selected from the original 20-camera setup to provide:

1. **Frontal coverage** (cam_28, cam_54): Essential for talking head video generation
2. **Profile coverage** (cam_37): Side view for 3D consistency
3. **Rear coverage** (cam_49): Back-of-head view for complete head modeling

This configuration balances storage efficiency with sufficient multi-view coverage for training speech-to-video models.

## Comparison with Full Dataset

| Metric | Full (20-cam) | 4-cam | Ratio |
|--------|--------------|-------|-------|
| Total Size | 2153.1 GB | 440.4 GB | 20.5% |
| Cameras | 20 | 4 | 20.0% |
| Total Images | 3,338,220 | 667,644 | 20.0% |
| Subjects | 21 | 21 | 100% |
| Performances | 126 | 126 | 100% |
| Frame Count | 166,911 | 166,911 | 100% |

## Notes

1. All speech performances (s1_all to s6_all) contain synchronized multi-view video with audio
2. Each frame is captured simultaneously across all 4 cameras
3. Frame rates vary based on performance duration (typically 30 fps)
4. All data uses combined structure (no from_anno/from_raw separation)
5. Each performance includes:
   - 4 camera views with 2448x2048 resolution
   - Corresponding segmentation masks
   - Synchronized audio track
   - Camera calibration matrices for all 4 views
   - 3D facial keypoints

## Usage for Training

This dataset is suitable for:
- Speech-to-video model training (e.g., Wan2.2-S2V-14B)
- Audio-driven talking head generation
- Multi-view consistency learning
- 3D-aware facial animation

The 4-camera configuration provides sufficient viewpoint diversity while maintaining manageable storage requirements for iterative experimentation and training.
