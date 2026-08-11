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
# 1. Configuration & Initialization
# ==========================================

MODEL_ID = "google/paligemma-3b-pt-224" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model onto GPU in bfloat16...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
print("Model loaded successfully.", flush=True)

# Global thread-safe variables for real-time prompt updating
current_prompt = "detect bottle"  # Initial default prompt
prompt_updated = False

ROBOT_IP = "192.168.1.100"
ROBOT_PORT = 30002
USE_ROBOT = False

H_MATRIX = np.array([
    [0.0015,  0.0000, -0.450],
    [0.0000, -0.0015,  0.600],
    [0.0000,  0.0000,  1.000]
])

# ==========================================
# 2. Background Thread for Terminal Input
# ==========================================

def terminal_prompt_listener():
    """
    Runs in the background, listening for new user prompts in the terminal
    without interrupting the camera feed loop.
    """
    global current_prompt, prompt_updated
    while True:
        try:
            # Blocks here waiting for input, but only in this background thread
            user_input = input()
            if user_input.strip():
                new_prompt = user_input.strip()
                # Automatically prepend "detect " if missing
                if not new_prompt.lower().startswith("detect"):
                    new_prompt = f"detect {new_prompt}"
                
                # Update global states
                current_prompt = new_prompt
                prompt_updated = True
                print(f"\n[System] Switching target to: '{current_prompt}'...\n", flush=True)
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

def pixel_to_robot_coordinates(px_x, px_y, homography_matrix):
    pixel_vector = np.array([px_x, px_y, 1.0])
    robot_vector = np.dot(homography_matrix, pixel_vector)
    robot_x = robot_vector[0] / robot_vector[2]
    robot_y = robot_vector[1] / robot_vector[2]
    robot_z = 0.150
    return robot_x, robot_y, robot_z

# ==========================================
# 4. Main Execution Loop
# ==========================================

def main():
    global current_prompt, prompt_updated
    
    print("\n[DEBUG] Opening camera...", flush=True)
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("[DEBUG] ERROR: Camera is blocked or unavailable.", flush=True)
        return

    print("[DEBUG] Camera opened. Feed initializing. Window should pop up immediately.", flush=True)
    
    print("\n" + "="*60)
    print(" LIVE CONTROL ACTIVE")
    print(f" Current Target: '{current_prompt}'")
    print(" --> TYPE A NEW TARGET IN THIS TERMINAL AT ANY TIME AND PRESS ENTER to update!")
    print("="*60 + "\n")

    # Start the background thread for terminal inputs
    input_thread = threading.Thread(target=terminal_prompt_listener, daemon=True)
    input_thread.start()

    last_inference_time = 0
    inference_interval = 0.4  # Run VLM every 400ms
    detections = []
    is_first_inference = True

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        height, width, _ = frame.shape
        current_time = time.time()

        # Copy frame for overlays
        display_frame = frame.copy()

        # Draw overlays
        for obj in detections:
            box = obj["box"]
            center = obj["center"]
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 120, 255), 2)
            
            if "rot_box" in obj and obj["rot_box"] is not None:
                cv2.drawContours(display_frame, [obj["rot_box"]], 0, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Rot: {obj['angle']:.1f}deg", (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"Target: {current_prompt.replace('detect ', '')}", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Render the OpenCV Window immediately
        cv2.imshow("PaliGemma Hand-Eye Orientation System", display_frame)
        cv2.waitKey(1)

        # Run VLM inference
        if current_time - last_inference_time > inference_interval:
            # We local-copy to avoid mid-inference threading modifications
            active_prompt = current_prompt 

            if is_first_inference:
                print("\n[VLM] Running first-time model inference. GPU is compiling kernels (may freeze feed for up to 30 seconds)...", flush=True)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                
                rx, ry, rz = pixel_to_robot_coordinates(center[0], center[1], H_MATRIX)
                angle, rot_box = calculate_orientation(frame, box)
                target["angle"] = angle
                target["rot_box"] = rot_box
                
                if angle is not None:
                    # Clearer print formatting
                    print(f"\r[FOUND] Coords: X={rx:.3f}m, Y={ry:.3f}m | Yaw: {angle:.1f}° | Target: '{active_prompt}'", end="", flush=True)
                    
                    if USE_ROBOT:
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.connect((ROBOT_IP, ROBOT_PORT))
                                command = f"move_to({rx:.4f}, {ry:.4f}, {rz:.4f}, {angle:.2f})\n"
                                s.sendall(command.encode('utf-8'))
                        except Exception:
                            pass
            else:
                if is_first_inference:
                    print("[VLM] First-time compilation complete. System running.", flush=True)
                    is_first_inference = False
                # Silent searching to keep the console clean for typing
                print(f"\r[Searching] Target: '{active_prompt}'...", end="", flush=True)

        # Allow user to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()