import pyrealsense2 as rs
import cv2
import torch
import numpy as np
import re
import socket
import time
import threading
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# ==========================================
# 1. Configuration & Model Initialization
# ==========================================

MODEL_ID = "google/paligemma-3b-pt-224" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading PaliGemma onto GPU...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
print("Model loaded successfully.", flush=True)

current_prompt = "detect screwdriver"  
prompt_updated = False

ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 30002
USE_ROBOT = False

# Extrinsics Transformation Matrix (Camera -> Robot Base Frame)
T_CAM_TO_ROBOT = np.array([
    [ 1.0,  0.0,  0.0,  0.150 ], 
    [ 0.0, -1.0,  0.0,  0.450 ], 
    [ 0.0,  0.0, -1.0,  0.600 ], 
    [ 0.0,  0.0,  0.0,  1.000 ]
])

def transform_to_robot_frame(camera_point, T_matrix):
    point_homo = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])
    robot_point = np.dot(T_matrix, point_homo)
    return robot_point[:3]

# ==========================================
# 2. Background Input Listener Thread
# ==========================================
def terminal_prompt_listener():
    global current_prompt, prompt_updated
    while True:
        try:
            user_input = input()
            if user_input.strip():
                new_prompt = user_input.strip()
                if not new_prompt.lower().startswith("detect"):
                    new_prompt = f"detect {new_prompt}"
                current_prompt = new_prompt
                prompt_updated = True
                print(f"\n[System] Target changed to: '{current_prompt}'...\n", flush=True)
        except (KeyboardInterrupt, EOFError):
            break

# ==========================================
# 3. Geometry & Orientation Helpers
# ==========================================
def calculate_orientation(image, box):
    x1, y1, x2, y2 = box
    h, w, _ = image.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)
    
    angle = rect[2]
    width, height = rect[1]
    
    if width < height:
        angle = angle + 90.0
    if angle > 45:
        angle -= 90
        
    global_box_points = box_points + [x1, y1]
    return angle, global_box_points

def parse_paligemma_coordinates(output_text, img_width, img_height):
    pattern = re.compile(r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>')
    matches = pattern.findall(output_text)
    
    detected_objects = []
    for match in matches:
        ymin, xmin, ymax, xmax = [int(val) / 1024.0 for val in match]
        
        px_xmin = int(xmin * img_width)
        px_ymin = int(ymin * img_height)
        px_xmax = int(xmax * img_width)
        px_ymax = int(ymax * img_height)
        
        px_center_x = int((px_xmin + px_xmax) / 2)
        px_center_y = int((px_ymin + px_ymax) / 2)
        
        detected_objects.append({
            "box": [px_xmin, px_ymin, px_xmax, px_ymax],
            "center": (px_center_x, px_center_y)
        })
    return detected_objects

# ==========================================
# 4. Main Execution Loop (RealSense Dashboard)
# ==========================================
def main():
    global current_prompt, prompt_updated

    print("\n[DEBUG] Initializing RealSense Pipeline...", flush=True)
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # RealSense Colorizer block (translates raw Z depth into a visible rainbow spectrum)
    colorizer = rs.colorizer()
    
    print("[DEBUG] RealSense camera streams started. Rendering side-by-side dashboard.", flush=True)

    input_thread = threading.Thread(target=terminal_prompt_listener, daemon=True)
    input_thread.start()

    last_inference_time = 0
    inference_interval = 0.4 
    detections = []
    is_first_inference = True

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Get color frame
        color_image = np.asanyarray(color_frame.get_data())
        
        # Colorize and generate depth frame
        colorized_depth_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(colorized_depth_frame.get_data())

        height, width, _ = color_image.shape
        current_time = time.time()

        # Copy frames for drawing overlays
        display_color = color_image.copy()
        display_depth = depth_color_image.copy()

        # Draw overlays on BOTH color and depth streams
        for obj in detections:
            box = obj["box"]
            center = obj["center"]
            
            # Color stream overlay
            cv2.rectangle(display_color, (box[0], box[1]), (box[2], box[3]), (0, 120, 255), 2)
            cv2.circle(display_color, center, 5, (0, 0, 255), -1)
            cv2.putText(display_color, f"Target: {current_prompt.replace('detect ', '')}", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Depth stream overlay
            cv2.rectangle(display_depth, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            cv2.circle(display_depth, center, 5, (0, 0, 0), -1)

            if "rot_box" in obj and obj["rot_box"] is not None:
                cv2.drawContours(display_color, [obj["rot_box"]], 0, (0, 255, 0), 2)
                cv2.putText(display_color, f"Rot: {obj['angle']:.1f}deg", (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # STACK SIDE-BY-SIDE: Stacks color image and depth color image horizontally
        dashboard = np.hstack((display_color, display_depth))

        # Show combined feeds in one unified window
        cv2.imshow("RealSense VLM Dashboard (Left: Color, Right: Depth)", dashboard)
        cv2.waitKey(1)

        # Run VLM
        if current_time - last_inference_time > inference_interval:
            active_prompt = current_prompt

            if is_first_inference:
                print("\n[VLM] Compiling first-time inference on Blackwell GPU...", flush=True)

            rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            inputs = processor(text=active_prompt, images=pil_img, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            detections = parse_paligemma_coordinates(output_text, width, height)
            last_inference_time = time.time()

            if len(detections) > 0:
                target = detections[0]
                box = target["box"]
                center = target["center"]
                
                # Get distance
                depth_value = depth_frame.get_distance(center[0], center[1])
                
                if depth_value > 0:
                    # 1. Get raw camera coordinates
                    camera_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [center[0], center[1]], depth_value)
                    
                    # 2. Transform to robot coordinates
                    rx, ry, rz = transform_to_robot_frame(camera_xyz, T_CAM_TO_ROBOT)
                    
                    # 3. Get rotation
                    angle, rot_box = calculate_orientation(color_image, box)
                    target["angle"] = angle
                    target["rot_box"] = rot_box
                    
                    # VERIFICATION DISPLAY: Prints both Cam-Frame and Robot-Frame values simultaneously
                    print(
                        f"\r[DEPTH] Cam Lens -> X: {camera_xyz[0]:.3f}m, Y: {camera_xyz[1]:.3f}m, Z_depth: {camera_xyz[2]:.3f}m | "
                        f"Robot Base -> X: {rx:.3f}m, Y: {ry:.3f}m, Z: {rz:.3f}m | "
                        f"Yaw: {angle:.1f}°", end="", flush=True
                    )
                    
                    if USE_ROBOT:
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.connect((ROBOT_IP, ROBOT_PORT))
                                payload = f"target_pose({rx:.4f},{ry:.4f},{rz:.4f},{angle:.2f})\n"
                                s.sendall(payload.encode('utf-8'))
                        except Exception:
                            pass
                else:
                    print(f"\r[Scanning] Target seen, but depth reading is invalid (occluded).", end="", flush=True)
            else:
                if is_first_inference:
                    print("[VLM] First-time compilation complete. System active.", flush=True)
                    is_first_inference = False
                print(f"\r[Scanning] Target: '{active_prompt}'...", end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()