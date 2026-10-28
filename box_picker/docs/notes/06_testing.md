# 6. Testing

**In short:** three kinds of check, from fast to slow. Run the fast ones after every change.

Run everything from `src/`.

## 1. Unit tests on made-up data (seconds, no camera, no GPU)

```bash
.venv/bin/python test_faces.py       # depth step: boxes ray-cast into a fake depth image
.venv/bin/python test_box_masks.py   # outline clean-up rules on made-up outlines
```

`test_faces.py` builds scenes where the true answer is known (a box of 100 × 80 mm at a known place), renders a depth image, and checks faces, sizes and centres. Each scene runs twice: clean, and with 1.5 mm noise and 10% missing pixels.

## 2. Outlines against labelled frames (about 30 s, GPU)

*Not in this package: `eval_masks.py` and the labelled dataset stay in the development repository.*

```bash
.venv/bin/python eval_masks.py ../../dataset/BoxDetectionv2.yolo26/valid ../../dataset/BoxDetectionv2.yolo26/test --video
```

Compares SAM 3.1's outlines with 324 hand-labelled boxes in 54 frames: is each box found exactly once, are boxes joined together? "No label" outlines are often real boxes the labels leave out, so check them by eye.

## 3. Recordings: the whole pipeline on real colour + depth

**Record** (R to start/pause, Q to stop). It saves colour, depth and camera settings, 10 frames a second:

```bash
.venv/bin/python record_rgbd.py recordings/test3
```

An `.mp4` from `record_realsense.py` has colour only: the pipeline can't run on it, because a video file can't hold depth.

**Several frames of one still scene, one answer** (what the robot does after it stops):

```bash
.venv/bin/python pick_frame.py --live --frames 5
.venv/bin/python pick_frame.py "recordings/test3/frame_0001[0-4]"
```

**Run the pipeline on the recording** (about 1 s a frame):

```bash
.venv/bin/python pick_pipeline.py --capture "recordings/test3/frame_*" --out outputs/test3 --pitch 0
```

- `--pitch <degrees>`: how far the camera was tilted down. Needed while the accelerometer can't be read (note 1); 0 = level.
- For long recordings, every 10th frame (one a second): `--capture "recordings/test3/frame_*0"`.
- One frame: `.venv/bin/python pick_frame.py recordings/test3/frame_00030`

**Turn the results into a video:**

```bash
ffmpeg -framerate 5 -pattern_type glob -i 'outputs/test3/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p outputs/test3.mp4
```

Each frame also has a `.json` with the target and every box's faces.

**What to look for**, skipping frames with hands in them:

| Check | Good |
|---|---|
| Boxes found | every box has its own yellow outline |
| Faces | green on faces at the camera, blue on faces pointing up |
| Face size | the white rectangle covers the whole face, and the size in mm is right |
| Target | follows the rules in note 5 |
| Steady | while nothing moves, the target stays on the same box |

Note the frame numbers of anything wrong.

## Live

Camera plugged into another machine (ROS 2)? Start `ros_bridge.py` first and add `--ros`; see the [README](../../INTEGRATION.md).

```bash
.venv/bin/python live_pick.py --log outputs/live_pick/targets.jsonl
```

A window with the camera and the latest result; each result is also printed and logged as JSON. While the target stays on the same box, the printed result is the average of the last 5 (`--smooth 5`, the default; `--smooth 1` turns it off), because depth noise moves a still box's result by about a millimetre from frame to frame. S saves the current frame (colour + depth) so it can be re-run later; Q quits.
