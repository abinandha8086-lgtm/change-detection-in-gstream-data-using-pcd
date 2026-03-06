# detect_changes.py  — v4  (delta-brightness object selection)
import open3d as o3d
import numpy as np
import os, sys
import cv2

print("=== detect_changes.py v4 loaded ===")


def get_most_stable_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_gray, motion = None, []
    step = 5
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion.append((i, cv2.absdiff(gray, prev_gray).mean()))
        prev_gray = gray
    cap.release()
    if not motion: return None, -1
    window = 10
    best_start, best_score = 0, float('inf')
    for i in range(len(motion) - window):
        score = sum(m[1] for m in motion[i:i+window])
        if score < best_score:
            best_score, best_start = score, motion[i][0]
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(best_start, min(best_start + window * step, total), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret: frames.append(frame.astype(np.float32))
    cap.release()
    if not frames: return None, -1
    return np.mean(frames, axis=0).astype(np.uint8), best_start


def align_with_ecc(f1, f2):
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sc = 0.5
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            cv2.resize(g1, None, fx=sc, fy=sc),
            cv2.resize(g2, None, fx=sc, fy=sc),
            warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        warp[0,2] /= sc; warp[1,2] /= sc
        h, w = f1.shape[:2]
        out = cv2.warpAffine(f2, warp, (w,h),
                             flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        print(f"   [ECC] shift=({warp[0,2]:.1f},{warp[1,2]:.1f})px")
        return out
    except Exception as e:
        print(f"   [ECC] failed ({e}), skipping alignment")
        return f2


def find_added_object_mask(f1, f2, session):
    h, w = f1.shape[:2]
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    lab1 = cv2.GaussianBlur(cv2.cvtColor(f1, cv2.COLOR_BGR2LAB).astype(np.float32), (9,9), 0)
    lab2 = cv2.GaussianBlur(cv2.cvtColor(f2, cv2.COLOR_BGR2LAB).astype(np.float32), (9,9), 0)
    diff = np.max(np.abs(lab2 - lab1), axis=2).astype(np.uint8)

    _, raw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted([c for c in cnts if cv2.contourArea(c) > 400],
                  key=cv2.contourArea, reverse=True)

    vis = f2.copy()
    best_cnt, best_delta = None, 0.0

    print(f"\n   {'AREA':>8}  {'DELTA':>8}  LOCATION")
    for cnt in cnts:
        m = np.zeros((h,w), dtype=np.uint8)
        cv2.drawContours(m, [cnt], -1, 255, -1)
        delta = float(cv2.mean(g2, mask=m)[0]) - float(cv2.mean(g1, mask=m)[0])
        x,y,cw,ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        tag = "<<< OBJECT ADDED" if delta < -10 else "(noise/movement)"
        print(f"   {area:>8.0f}  {delta:>+8.1f}  ({x},{y} {cw}x{ch})  {tag}")
        col = (0,220,0) if delta < -10 else (80,80,80)
        cv2.drawContours(vis, [cnt], -1, col, 2)
        if delta < best_delta:
            best_delta, best_cnt = delta, cnt

    if best_cnt is None:
        print("   WARNING: nothing darkened, using largest")
        best_cnt = cnts[0] if cnts else None

    clean = np.zeros((h,w), dtype=np.uint8)
    if best_cnt is not None:
        cv2.drawContours(clean, [best_cnt], -1, 255, -1)
        cv2.drawContours(vis, [best_cnt], -1, (0,0,255), 3)
        x,y,cw,ch = cv2.boundingRect(best_cnt)
        print(f"\n   >>> SELECTED: {cw}x{ch}px  delta={best_delta:.1f}\n")

    cv2.imwrite(os.path.join(session, "debug_detection_vis.jpg"), vis)
    return clean


def mask_to_pcd(frame, mask):
    h, w = frame.shape[:2]
    fx = fy = 525.0
    cx, cy = w/2.0, h/2.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    depth = (255.0 - gray) / 255.0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    d = depth[ys, xs]
    Z = d * 3.0 + 0.1
    X =  (xs - cx) / fx * Z
    Y = -((ys - cy) / fy * Z)
    rgb = frame[ys, xs, ::-1].astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([X,Y,Z], axis=1))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd


def frame_to_bg_pcd(frame):
    h, w = frame.shape[:2]
    fx = fy = 525.0
    cx, cy = w/2.0, h/2.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    depth = (255.0 - gray) / 255.0
    xs = np.tile(np.arange(w), h).astype(np.float32)
    ys = np.repeat(np.arange(h), w).astype(np.float32)
    d  = depth.flatten()
    keep = d > 0.08
    xs, ys, d = xs[keep], ys[keep], d[keep]
    Z = d*3.0+0.1; X = (xs-cx)/fx*Z; Y = -((ys-cy)/fy*Z)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([X,Y,Z], axis=1))
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
    return pcd.voxel_down_sample(0.015)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_changes.py <session_folder>")
        return
    session = sys.argv[1]
    vid1 = os.path.join(session, "baseline_video.avi")
    vid2 = os.path.join(session, "comparison_video.avi")
    if not os.path.exists(vid1) or not os.path.exists(vid2):
        print("Error: Need baseline_video.avi and comparison_video.avi"); return

    print(f"\n🔍 Session: {session}")
    f1, s1 = get_most_stable_frame(vid1)
    f2, s2 = get_most_stable_frame(vid2)
    if f1 is None or f2 is None: print("❌ Frame extraction failed"); return
    print(f"   Stable frames: baseline={s1}, comparison={s2}")

    cv2.imwrite(os.path.join(session, "baseline_best_frame.jpg"),   f1)
    cv2.imwrite(os.path.join(session, "comparison_best_frame.jpg"), f2)

    f2a = align_with_ecc(f1, f2)
    cv2.imwrite(os.path.join(session, "debug_aligned.jpg"), f2a)

    mask = find_added_object_mask(f1, f2a, session)
    cv2.imwrite(os.path.join(session, "debug_mask.jpg"), mask)

    if np.count_nonzero(mask) < 300:
        print("❌ No change detected"); return

    obj = mask_to_pcd(f2a, mask)
    bg  = frame_to_bg_pcd(f1)
    if obj is None: print("❌ PCD failed"); return

    print(f"✅ {len(obj.points)} object points detected")
    bg.paint_uniform_color([0.55,0.55,0.55])
    obj.paint_uniform_color([1.0,0.0,0.0])
    o3d.visualization.draw_geometries([bg, obj],
        window_name="3D CDNet Success", width=1100, height=750)

if __name__ == "__main__":
    main()
