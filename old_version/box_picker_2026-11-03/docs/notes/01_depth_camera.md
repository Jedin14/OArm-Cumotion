# 1. The depth camera

**In short:** the Intel RealSense D455 gives us two pictures at the same time: a normal colour picture, and a depth picture that says how far away every pixel is. It measures depth like our two eyes do, so some places it can't measure, and those pixels have no depth (0).

## Two pictures

| Picture | Each pixel holds | Saved as |
|---|---|---|
| Colour | red, green, blue | `*_color.png` |
| Depth | distance from the camera in millimetres (0 = no reading) | `*_depth_mm.png` (16-bit PNG) |

The depth picture is **aligned** to the colour picture: pixel (u, v) in one is the same spot as pixel (u, v) in the other. The code does this with `rs.align` in [`rgbd.py`](../../src/rgbd.py). Without it, the two pictures would be slightly shifted, because the colour sensor and the depth sensors sit in different places on the camera.

"Distance" here means the **z** distance: straight ahead from the camera, not along the line to the point. Note 2 explains why that matters.

## How it measures depth: stereo

The D455 has two infrared cameras, 95 mm apart (the **baseline**). Both see the same scene from slightly different places. A near object appears shifted more between the two images than a far one. That shift is the **disparity**, and depth comes from it:

```
depth = baseline × focal length / disparity
```

A projector also throws a pattern of infrared dots onto the scene, so the camera can match plain surfaces (like a white box) that have no texture of their own.

## Why some pixels have no depth

Stereo needs **both** infrared cameras to see the same spot. Where they don't, there is no depth. In this project that happens in three ways:

1. **The depth shadow.** Next to the edge of a near object, there is a strip of the background that one infrared camera can see but the other can't, because the object is in the way. That strip has no depth. Its width can be worked out: about `baseline × focal length × |1/z_near − 1/z_far|` pixels. With a box 4 cm in front of another, that's about 9 pixels; with a box 20 cm in front of a wall, about 35. [`faces.py`](../../src/faces.py) (`fill_holes`) uses this to fill in such strips on a face, and to refuse strips that are too wide to be a shadow.
2. **Surfaces seen almost edge-on.** A box top seen from nearly level reflects little of the dot pattern back, so most of it has no depth. That's why a top whose front edge is seen at under 10° is not trusted, and why, of several frames, the one that found the most of a top is used (note 5).
3. **Edges.** Right at an object's edge, a pixel can mix the object and what's behind it. The code shrinks each outline by 4 pixels (`ERODE_PX`) before fitting planes, to stay away from these.

## Noise

Depth is never exact. The error grows with distance: at 0.5 m it's a few millimetres. The code allows for this, for example a point counts as "on a plane" if it's within 1% of its distance (5 mm at 0.5 m).

## Which way is up: the accelerometer

The D455 also has an accelerometer. When the camera is still, it measures gravity, which tells us which way is **up** in the camera's view. We need that to tell a top face (pointing up) from a front face. On some PCs Linux blocks access to it (`Permission denied`); then the code assumes the camera is level, or you pass `--pitch <degrees the camera is tilted down>`. The fix is the librealsense udev rules (see the project README).
