"""Detect boxes in a single frame and return the pick target's centroid.

Selection rule:
  1. Find the topmost box (smallest centroid y; image y grows downward).
  2. Every box whose centroid y is within TOL x box height of that box counts
     as tied for the top row (box height = median height of the detections).
  3. Among the tied boxes, pick the leftmost (smallest centroid x).

Coordinates are pixels in the input frame's own resolution, origin at the
top-left corner.

Python:
    from box_detector import BoxTargetDetector
    detector = BoxTargetDetector()           # load once at startup
    result = detector.detect(frame)          # frame: BGR numpy array (H, W, 3)

Command line (for testing):
    python3 box_detector.py samples/frame_00600.jpg --tol 0.3 --annotate out.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_model.pt")


def select_target(xyxy, tol):
    """Return (index, top-row indices) of the target box, or (None, []) if no boxes."""
    if len(xyxy) == 0:
        return None, []
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    box_h = np.median(xyxy[:, 3] - xyxy[:, 1])
    top_row = np.flatnonzero(cy <= cy.min() + tol * box_h)
    return top_row[np.argmin(cx[top_row])], top_row


class BoxTargetDetector:
    def __init__(self, model_path=DEFAULT_MODEL, tol=0.3, conf=0.5, device=None):
        """
        model_path: path to the trained YOLO weights.
        tol:        top-row leeway as a fraction of box height (0.3 = +/-30%).
        conf:       minimum detection confidence (0-1).
        device:     "cpu", "0" (first GPU), ... ; None picks GPU if available.
        """
        self.model = YOLO(model_path)
        self.tol = tol
        self.conf = conf
        self.device = device
        # Warm-up so the first real frame is not slow.
        self.model.predict(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)

    def detect(self, frame, tol=None):
        """Return the target as a dict (see README), or None if no box is detected.

        frame: BGR uint8 numpy array of shape (H, W, 3), as produced by OpenCV.
        tol:   override the leeway for this call only.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR numpy array of shape (H, W, 3)")
        r = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        target, _ = select_target(xyxy, self.tol if tol is None else tol)
        if target is None:
            return None
        x1, y1, x2, y2 = xyxy[target]
        h, w = frame.shape[:2]
        return {
            "x": round(float((x1 + x2) / 2), 1),
            "y": round(float((y1 + y2) / 2), 1),
            "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "confidence": round(float(r.boxes.conf[target]), 3),
            "num_boxes": len(xyxy),
            "boxes": [[round(float(v), 1) for v in b] for b in xyxy],
            "image_width": w,
            "image_height": h,
        }

    def detect_bytes(self, data, tol=None):
        """Same as detect(), for an encoded image (JPEG/PNG bytes) received over a network."""
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode image bytes")
        return self.detect(frame, tol)

    def annotate(self, frame, tol=None):
        """Return a copy of the frame with all boxes drawn and the target marked (for debugging)."""
        xyxy = self.model.predict(frame, conf=self.conf, device=self.device,
                                  verbose=False)[0].boxes.xyxy.cpu().numpy()
        target, top_row = select_target(xyxy, self.tol if tol is None else tol)
        out = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            color = (0, 0, 255) if i == target else (0, 200, 255) if i in top_row else (255, 150, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if i == target else 1)
        if target is not None:
            x1, y1, x2, y2 = xyxy[target]
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.drawMarker(out, c, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(out, f"target ({c[0]}, {c[1]})", (c[0] + 12, c[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tol", type=float, default=0.3, help="top-row leeway as a fraction of box height")
    ap.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    ap.add_argument("--device", default=None, help='"cpu" or GPU index such as "0"')
    ap.add_argument("--annotate", metavar="OUT.jpg", help="also save an annotated image")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image}")
    detector = BoxTargetDetector(args.model, tol=args.tol, conf=args.conf, device=args.device)
    print(json.dumps(detector.detect(frame)))
    if args.annotate:
        cv2.imwrite(args.annotate, detector.annotate(frame))


if __name__ == "__main__":
    main()
