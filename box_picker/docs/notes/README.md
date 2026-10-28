# Notes: the ideas behind the box picker

These notes explain, in plain words, the ideas the box-picking pipeline is built on. They are for someone new to the project who knows some Python but not much about cameras, 3D or computer vision.

Read them in order. Each one is short and builds on the one before.

| # | Note | What you learn |
|---|---|---|
| 1 | [The depth camera](01_depth_camera.md) | How the Intel RealSense D455 measures distance, and why some pixels have no depth |
| 2 | [From a pixel to a 3D point](02_pixels_to_3d.md) | Camera intrinsics, and how a pixel plus its depth becomes a point in metres |
| 3 | [Finding the boxes with SAM 3.1](03_finding_boxes.md) | What a segmentation model does, text prompts, scores, and how outlines are cleaned up |
| 4 | [Faces and planes](04_faces_and_planes.md) | Fitting flat surfaces to depth points (RANSAC), front / top / side, and measuring a face |
| 5 | [Choosing the target](05_choosing_the_target.md) | The picking rules, step by step, and what the output means |
| 6 | [Testing](06_testing.md) | How we check the pipeline: synthetic tests, labelled frames, recordings |
| — | [Glossary](glossary.md) | Short definitions of the words used in the code and in these notes |

The big picture is in [../architecture.md](../architecture.md); the rules themselves are in [../picking_rules.md](../picking_rules.md); how the robot uses the answer is in [../robot_pick_logic.md](../robot_pick_logic.md). The code is in [`src/`](../../src/); setup: [INTEGRATION.md](../../INTEGRATION.md).

## The pipeline in one paragraph

The camera takes a colour picture and a depth picture. SAM 3.1, an AI model, draws an outline around every box in the colour picture. For each outline, we take the depth of its pixels, turn them into 3D points, and find the flat faces of the box. A face pointing at the camera is a **front**, one pointing up is a **top**. We pick one box using simple rules (nothing on top of it, nothing in front of it, nearest, on top of a stack first, then leftmost) and give the robot the centre of its face in metres.
