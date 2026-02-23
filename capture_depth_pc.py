import gs_change
import cv2
import open3d as o3d
import os
import numpy as np

def save_pcd_from_video(video_path, pcd_name):
    """Extracts a frame and saves it as a 3D Point Cloud."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20) # Use a stable frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return
    
    # Focus on the floor (ROI)
    roi = frame[150:, :, :] 
    gray = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(gray)
    
    # Using a scale of 20.0 to make depth differences more 'dramatic'
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=20.0, convert_rgb_to_intensity=False)
    
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    # Clean noise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
    
    o3d.io.write_point_cloud(pcd_name, pcd)
    print(f"   [PCD] Saved: {os.path.basename(pcd_name)}")

def main():
    # Unlock network port
    os.system("sudo fuser -k 1720/udp > /dev/null 2>&1")
    
    session = gs_change.create_session_folder()
    print(f"🚀 SESSION STARTED: {session}")

    # 1. Baseline
    print("\n[STEP 1] RECORDING BASELINE (Empty)")
    input("Action: Clear the floor and press ENTER...")
    vid1 = gs_change.record_30s_video(session, "baseline_video.avi")
    save_pcd_from_video(vid1, os.path.join(session, "baseline.pcd"))

    # 2. Comparison
    print("\n[STEP 2] RECORDING COMPARISON (Object)")
    input("Action: Place the bottle/backpack and press ENTER...")
    vid2 = gs_change.record_30s_video(session, "comparison_video.avi")
    save_pcd_from_video(vid2, os.path.join(session, "comparison.pcd"))

    print(f"\n✅ SUCCESS! Folder contains Videos and PCD files.")
    print(f"Run: python3 detect_changes.py {session}")

if __name__ == "__main__":
    main()
