import gs_change
import cv2
import open3d as o3d
import os
import numpy as np


def get_most_stable_frame(video_path):
    """
    Finds the single most stable frame in the video by picking the period
    with lowest inter-frame motion, then averaging frames in that window.
    This ensures we get a sharp, non-blurry representative frame.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 1: Compute motion score for each frame vs previous
    prev_gray = None
    motion = []
    step = 5
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).mean()
            motion.append((i, diff))
        prev_gray = gray

    if not motion:
        cap.release()
        return None

    # Step 2: Find 10-frame window with lowest cumulative motion
    window = 10  # frames in window
    best_start = 0
    best_score = float('inf')
    for i in range(len(motion) - window):
        score = sum(m[1] for m in motion[i:i+window])
        if score < best_score:
            best_score = score
            best_start = motion[i][0]

    # Step 3: Average frames in the stable window
    frames = []
    for i in range(best_start, min(best_start + window * step, total), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))
    cap.release()

    if not frames:
        return None

    avg = np.mean(frames, axis=0).astype(np.uint8)
    print(f"   [FRAME] Using stable window starting at frame {best_start} (motion={best_score:.2f})")
    return avg


def frame_to_pcd(frame, pcd_name):
    """
    Converts a frame to a point cloud.
    Uses proper perspective projection with luminance-based pseudo-depth.
    Dark objects on a bright floor/wall = correct depth ordering.
    """
    h, w = frame.shape[:2]

    # Build pixel grid
    xs = np.tile(np.arange(w), h).astype(np.float32)
    ys = np.repeat(np.arange(h), w).astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Invert: dark=close (objects are darker than walls/floor in this scene)
    depth = (255.0 - gray) / 255.0

    depth_flat = depth.flatten()
    rgb = frame[:, :, ::-1].reshape(-1, 3).astype(np.float32) / 255.0

    focal = 525.0
    cx, cy = w / 2.0, h / 2.0
    Z = depth_flat * 3.0 + 0.1
    X = (xs - cx) / focal * Z
    Y = (ys - cy) / focal * Z

    points = np.stack([X, Y, Z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Keep only meaningful depth range (skip bright background)
    mask = depth_flat > 0.10
    pcd = pcd.select_by_index(np.where(mask)[0])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud(pcd_name, pcd)
    print(f"   [PCD] Saved: {os.path.basename(pcd_name)}  ({len(pcd.points)} points)")


def save_pcd_from_video(video_path, pcd_name):
    frame = get_most_stable_frame(video_path)
    if frame is None:
        print(f"   [PCD] ERROR: Could not read frames from {video_path}")
        return
    # Save the best frame as jpg for debugging
    debug_jpg = pcd_name.replace(".pcd", "_best_frame.jpg")
    cv2.imwrite(debug_jpg, frame)
    print(f"   [FRAME] Saved best frame: {os.path.basename(debug_jpg)}")
    frame_to_pcd(frame, pcd_name)


def main():
    os.system("sudo fuser -k 1720/udp > /dev/null 2>&1")
    session = gs_change.create_session_folder()
    print(f"🚀 SESSION STARTED: {session}")

    print("\n[STEP 1] RECORDING BASELINE (Empty floor)")
    input("Action: Clear the floor completely and press ENTER...")
    vid1 = gs_change.record_30s_video(session, "baseline_video.avi")
    save_pcd_from_video(vid1, os.path.join(session, "baseline.pcd"))

    print("\n[STEP 2] RECORDING COMPARISON (With object)")
    input("Action: Place the object on the floor and press ENTER...")
    vid2 = gs_change.record_30s_video(session, "comparison_video.avi")
    save_pcd_from_video(vid2, os.path.join(session, "comparison.pcd"))

    print(f"\n✅ SUCCESS! Session folder: {session}")
    print(f"   Check *_best_frame.jpg files to verify both frames look similar in viewpoint!")
    print(f"   Run: python3 detect_changes.py {session}")


if __name__ == "__main__":
    main()
