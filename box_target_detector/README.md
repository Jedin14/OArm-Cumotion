# Box Target Detector

Takes a single camera frame, detects the boxes in it, and returns the pixel centroid of the **pick target**: the topmost box, using the leftmost box when several are tied for the top row.

- Model: YOLO26m, fine-tuned on one class (`box`)
- Input: one frame (OpenCV BGR array, or JPEG/PNG bytes), at any resolution
- Output: target centroid `(x, y)` in that frame's own pixel coordinates
- Speed: about 5 ms per frame on an RTX 5060 Ti, about 53 ms per frame on CPU

---

## Package contents

| File | Purpose |
|---|---|
| `box_detector.py` | `BoxTargetDetector` class plus a command-line test tool |
| `box_model.pt` | Trained model weights (44 MB) |
| `requirements.txt` | Python dependencies (pinned versions it was tested with) |
| `samples/frame_00600.jpg` | Test frame: a normal stack |
| `samples/frame_06500_box_in_hand.jpg` | Test frame: a box held in a hand (see [Known limitations](#known-limitations)) |

---

## Install

Tested with Python 3.10, `torch 2.13.0+cu130` and `ultralytics 8.4.103`.

```bash
# 1. Install PyTorch for your platform first (CUDA, CPU-only or Jetson):
#    https://pytorch.org/get-started/locally/
# 2. Then:
pip install -r requirements.txt
```

Quick check:

```bash
python3 box_detector.py samples/frame_00600.jpg
# {"x": 484.1, "y": 389.0, "bbox": [455.1, 350.4, 513.2, 427.7], "confidence": 0.918, "num_boxes": 7, "image_width": 1280, "image_height": 720}
```

---

## Usage

### In the robot code

Load the model **once** at startup, then call `detect()` for each frame the robot sends:

```python
from box_detector import BoxTargetDetector

detector = BoxTargetDetector(tol=0.3)      # loads the model and warms it up (~1–2 s)

result = detector.detect(frame)            # frame: BGR numpy array, shape (H, W, 3)
if result is None:
    ...                                    # no box in view
else:
    x, y = result["x"], result["y"]        # pixel coordinates of the target centroid
```

If the frame arrives as encoded bytes (for example a JPEG over a socket or HTTP):

```python
result = detector.detect_bytes(jpeg_bytes)
```

**Colour order:** `detect()` expects **BGR**, which is OpenCV's default. If the robot gives RGB frames, convert them first with `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)`. With ROS `cv_bridge`, use `imgmsg_to_cv2(msg, "bgr8")`.

### Command line (for testing)

```bash
python3 box_detector.py <image> [--tol 0.3] [--conf 0.5] [--device cpu|0] [--annotate out.jpg]
```

`--annotate` saves a debug image. The target is drawn in thick red with a cross at its centroid, other top-row boxes in yellow, and all remaining boxes in blue.

---

## Output

`detect()` returns `None` when no box is detected. Otherwise it returns:

```json
{
  "x": 484.1,
  "y": 389.0,
  "bbox": [455.1, 350.4, 513.2, 427.7],
  "confidence": 0.918,
  "num_boxes": 7,
  "image_width": 1280,
  "image_height": 720
}
```

| Field | Meaning |
|---|---|
| `x`, `y` | Target box centroid in pixels |
| `bbox` | Target box corners `[x1, y1, x2, y2]` in pixels |
| `confidence` | Detection confidence of the target box (0–1) |
| `num_boxes` | Total boxes detected in the frame |
| `boxes` | Every detected box as `[x1, y1, x2, y2]` (pixels), the target among them |
| `image_width`, `image_height` | Size of the input frame, so the coordinates can be interpreted |

### Coordinate system

- Origin `(0, 0)` is the **top-left** corner of the frame. **x increases to the right and y increases downward.**
- Coordinates are always in the **input frame's own resolution**; no size is hard-coded. The model internally runs at 640 px, and Ultralytics maps the boxes back to the original size. For example, the same scene gives `(484.1, 389.0)` at 1280×720 and `(726.2, 583.6)` at 1920×1080.
- These are **image pixels**, not robot or world coordinates. Converting them to a robot pose needs the camera intrinsics and the hand–eye (camera-to-robot) calibration, which this package does not include.

---

## Target selection rule

1. Detect all boxes with confidence ≥ `conf` and compute each box's centroid.
2. Find the **topmost** box, meaning the one with the smallest centroid `y`.
3. Every box whose centroid `y` is within **`tol` × box height** of that topmost centroid counts as **tied for the top row**. Box height is the median height of all detected boxes, so one unusual detection doesn't skew it.
4. Among the tied boxes, return the **leftmost** one, meaning the one with the smallest centroid `x`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `tol` | `0.3` | Top-row leeway as a fraction of box height (0.3 = ±30%). Can also be set per call: `detect(frame, tol=0.5)`. The next row of boxes sits about 1 box height lower, so keep it below about 0.8. |
| `conf` | `0.5` | Minimum detection confidence. Raise it to ignore uncertain detections. |
| `device` | `None` | `None` uses the GPU if one is available. `"cpu"` forces CPU; `"0"` selects the first GPU. |
| `model_path` | `box_model.pt` | Path to the model weights, which must sit next to the script. |

---

## Model performance

Trained on 145 frames, with 40 for validation and 21 for testing, at 640 px input.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation | 0.989 | 0.996 | 0.994 | 0.873 |
| Test | 0.986 | 0.996 | 0.995 | 0.877 |

All frames came from **one video**: the same camera, carton, lighting and box type. These scores show how well the model works in that setup. Validate it on frames from the robot's own camera before relying on it.

---

## Known limitations

- **A box held in a hand can become the target.** A box being placed or removed is usually the highest one in view. In `samples/frame_06500_box_in_hand.jpg`, the held box is detected at confidence 0.50 and returned as the target. Recommended mitigations:
  - Only act when the target stays at the same position (within a few pixels) for several consecutive frames, or
  - Capture the frame only when no person or arm is in the workspace, or
  - Raise `conf`. This is less reliable, because a held box can still score high.
- **Trained for a single scene.** A new camera position, lighting, carton or box type may lower accuracy. If that happens, collect frames from the new setup, annotate them with **rectangles only** (polygon annotations broke the first training run), and retrain.
- **Don't resize frames before sending them.** Stretching a frame to a different aspect ratio, such as squashing 16:9 into 640×640, distorts the boxes and caused missed detections in testing. Send the camera frame as captured.
- Single class only (`box`). There's no orientation or depth estimate, and no handling for boxes that overlap in depth.

---

## Retraining

The model was trained with Ultralytics:

```bash
yolo detect train model=yolo26m.pt data=dataset/data.yaml epochs=200 imgsz=640 batch=16 patience=50
```

To update the model, export the Roboflow dataset in YOLO26 format, train with the command above, and replace `box_model.pt` with the new `runs/detect/<name>/weights/best.pt`.
