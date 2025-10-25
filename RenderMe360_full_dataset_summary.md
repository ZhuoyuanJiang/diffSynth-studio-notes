# RenderMe360 Dataset Summary

**Generated:** 2025-09-26 15:40:26
**Location:** `/ssd4/zhuoyuan/renderme360_temp/test_download/subjects/`

## Dataset Overview

- **Total Subjects:** 21
- **Performances per Subject:** 6 (s1_all to s6_all)
- **Cameras per Performance:** 20
- **Image Resolution:** 2448x2048 (uniform across all data)
- **Total Dataset Size:** 2153.1 GB

## Extraction Configuration
Based on `config_21id_20cam.yaml`:
- **Camera Selection:** 20 cameras optimized for facial coverage
  - cameras IDs: [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 49, 51, 54, 55, 56]
  - Front hemisphere: cameras 21-32, 54-56 (14 cameras)
  - Profile views: cameras 36, 37 (2 cameras)  
  - Rear hemisphere: cameras 0, 49, 51 (3 cameras)
  - Wide context: camera 28
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
| 0026 | 119.7 | 32.9 | 19.8 | 18.4 | 18.3 | 16.9 | 13.4 |
| 0041 | 75.5 | 20.1 | 11.6 | 11.1 | 12.2 | 11.7 | 8.8 |
| 0048 | 99.1 | 26.1 | 14.6 | 14.8 | 15.4 | 16.3 | 11.9 |
| 0094 | 89.4 | 16.5 | 16.0 | 16.9 | 19.1 | 11.5 | 9.4 |
| 0099 | 77.7 | 12.6 | 13.2 | 15.7 | 16.8 | 10.6 | 8.8 |
| 0100 | 76.9 | 17.2 | 12.2 | 12.0 | 12.7 | 14.6 | 8.2 |
| 0116 | 142.2 | 44.4 | 19.7 | 20.8 | 22.5 | 19.5 | 15.3 |
| 0156 | 141.2 | 32.9 | 23.4 | 25.6 | 22.9 | 22.1 | 14.1 |
| 0168 | 73.3 | 18.6 | 11.2 | 11.6 | 11.0 | 11.8 | 9.1 |
| 0175 | 128.6 | 25.4 | 22.5 | 20.8 | 19.3 | 22.7 | 18.0 |
| 0189 | 70.2 | 19.9 | 10.8 | 11.0 | 10.2 | 10.7 | 7.6 |
| 0195 | 108.5 | 33.9 | 16.0 | 15.5 | 14.5 | 16.4 | 12.2 |
| 0232 | 98.6 | 28.8 | 13.6 | 15.4 | 15.1 | 14.9 | 10.8 |
| 0250 | 62.1 | 18.6 | 9.7 | 8.9 | 9.3 | 9.3 | 6.3 |
| 0253 | 95.5 | 28.7 | 13.9 | 14.3 | 14.3 | 13.7 | 10.6 |
| 0259 | 83.4 | 21.9 | 13.4 | 12.0 | 12.4 | 13.4 | 10.3 |
| 0262 | 142.2 | 37.7 | 20.4 | 21.7 | 21.4 | 23.2 | 17.7 |
| 0278 | 91.1 | 13.0 | 19.5 | 17.8 | 19.2 | 11.3 | 10.2 |
| 0290 | 79.3 | 15.3 | 15.7 | 14.6 | 14.0 | 10.6 | 9.1 |
| 0295 | 126.2 | 23.5 | 21.0 | 27.5 | 26.5 | 14.6 | 13.1 |
| 0297 | 172.5 | 30.9 | 30.9 | 31.9 | 36.7 | 22.9 | 19.1 |


## Performance Details (Audio Duration in seconds)

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
- **Total Images (20 cameras):** 3,338,220 images

### Storage Distribution
- **Smallest Subject:** 0250 (62.1 GB)
- **Largest Subject:** 0297 (172.5 GB)
- **Average Subject Size:** 102.5 GB

## Notes

1. All speech performances (s1_all to s6_all) contain synchronized multi-view video with audio
2. Each frame is captured simultaneously across all 20 cameras
3. Frame rates vary based on performance duration (typically 30 fps)
4. All data uses combined structure (no from_anno/from_raw separation)
5. Each performance includes:
   - 20 camera views with 2448x2048 resolution
   - Corresponding segmentation masks
   - Synchronized audio track
   - Camera calibration matrices
   - 3D facial keypoints