import cv2
import numpy as np
import subprocess
import time
import os
from datetime import datetime

# Exact command derived from your working manual test
PIPE_CMD = (
    "gst-launch-1.0 -q udpsrc address=230.1.1.1 port=1720 multicast-iface=eno1 ! "
    "application/x-rtp, media=video, payload=96, encoding-name=H264 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "videoscale ! video/x-raw, format=BGR, width=640, height=480 ! fdsink"
)

def create_session_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("change_detection_results", f"session_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path

def send_heartbeat():
    """Triggers the robot to start streaming."""
    cmd = ["ros2", "topic", "pub", "--once", "/api/videohub/request", 
           "unitree_api/msg/Request", "{parameter: '{\"api_id\":1001}'}"]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def record_30s_video(session_path, filename):
    send_heartbeat()
    full_path = os.path.join(session_path, filename)
    
    # Large buffer to prevent 'internal data stream error'
    process = subprocess.Popen(PIPE_CMD.split(), stdout=subprocess.PIPE, bufsize=10**8)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(full_path, fourcc, 20.0, (640, 480))
    
    print(f"   [REC] {filename} in progress (30s)...")
    start = time.time()
    try:
        while (time.time() - start) < 30:
            raw = process.stdout.read(640 * 480 * 3)
            if not raw: break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((480, 640, 3))
            out.write(frame)
            cv2.imshow("Go2 Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        process.terminate()
        out.release()
        cv2.destroyAllWindows()
    return full_path
