# 3D Change Detection for Unitree Go2
3DCDNet is a lightweight, high-sensitivity change detection pipeline designed specifically for the Unitree Go2 LiDAR and depth sensor suite. It allows the robot to identify new objects in a scene by comparing a "Baseline" scan to a "Comparison" scan, rendering the results in a clean, architectural Grey (Static) / Red (Change) visual style.

## Features

- Stable Frame Extraction: Scores every frame window by inter-frame motion and averages the 10 calmest frames, eliminating camera shake and per-frame flicker before any processing begins.
- ECC Dense Alignment: Uses Enhanced Correlation Coefficient alignment on all pixels (not sparse keypoints) to correct camera drift between the baseline and comparison recordings — works reliably on the low-texture indoor scenes the Go2 camera captures.
- Brightness-Delta Object Selection: Identifies the added object by measuring which changed region got darker in the comparison frame. Dark objects placed on a bright floor produce a strong negative brightness delta, reliably separating the bag/object from people moving in the background.
- Distance-Transform 3D Shape: Projects the detected 2D silhouette into 3D using a distance transform — center pixels of the object get the highest depth value (closest to camera), edges taper back — giving the object a natural convex 3D shape instead of a flat slab.

## Steps

1. create change-detection-in-gstream-data-using-pcd/change_detection_results
2. Record data. Each recording is 30 seconds.

       python3 capture_depth_pc.py
   
3. Detect Changes
   Pass that session folder path to the detector:
   
         python3 detect_changes.py change_detection_results/session_YYYYMMDD_HHMMSS

## Result
   Change detection result<img width="1106" height="798" alt="Screenshot from 2026-03-06 10-12-46" src="https://github.com/user-attachments/assets/4ee67bd2-098e-4728-8b27-8aace1e56570" />
