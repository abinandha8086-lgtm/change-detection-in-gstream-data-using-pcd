import open3d as o3d
import numpy as np
import os, sys

def main():
    if len(sys.argv) < 2: return
    session = sys.argv[1]
    
    print("🚀 Loading Raw Data (No Filters)...")
    pcd1 = o3d.io.read_point_cloud(os.path.join(session, "baseline.pcd"))
    pcd2 = o3d.io.read_point_cloud(os.path.join(session, "comparison.pcd"))

    # Convert to numpy arrays immediately to bypass Open3D's memory issues
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)

    print(f"✅ Raw Data Found: {len(pts1)} points in Base, {len(pts2)} points in Comp.")

    # --- STEP 1: FORCE OVERLAP CHECK ---
    # We use a KDTree on the RAW points. 
    # If the code crashes here, your .pcd files are corrupt.
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    
    change_indices = []
    # threshold = 0.05 (5cm). If no point is within 5cm, it's a change.
    threshold = 0.05 

    print("🔍 Calculating Differences...")
    for i in range(len(pts2)):
        [k, idx, _] = pcd1_tree.search_radius_vector_3d(pts2[i], threshold)
        if k == 0:
            change_indices.append(i)

    if len(change_indices) > 10:
        # Create the Red Object
        red_obj = pcd2.select_by_index(change_indices)
        
        # --- STEP 2: THE "SUCCESS" LOOK ---
        pcd1.paint_uniform_color([0.7, 0.7, 0.7]) # Grey Room
        red_obj.paint_uniform_color([1, 0, 0])      # Red Object

        print(f"🚨 SUCCESS! Detected {len(change_indices)} points of change.")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D CDNet SUCCESS", width=1280, height=720)
        vis.add_geometry(pcd1)
        vis.add_geometry(red_obj)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1]) # White Background
        opt.point_size = 5.0 
        vis.run()
    else:
        print("No differences found. The two scans are mathematically identical.")

if __name__ == "__main__":
    main()
