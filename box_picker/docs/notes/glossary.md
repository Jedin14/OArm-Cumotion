# Glossary

| Word | Meaning |
|---|---|
| **aligned depth** | a depth picture shifted so each pixel matches the same spot in the colour picture |
| **baseline** | distance between the D455's two infrared cameras: 95 mm |
| **camera coordinates** | metres from the camera: x right, y down, z forward |
| **centroid** | the centre of the pick face, in metres; where the robot aims |
| **deproject** | turn a pixel + its depth into a 3D point |
| **depth shadow** | a strip beside a near object's edge where the camera has no depth, because only one of its two sensors sees it |
| **disparity** | how far a point shifts between the two infrared images; bigger = nearer |
| **front / top / side** | a face pointing at the camera / pointing up / anything else (never picked) |
| **inlier** | a point close enough to a plane to count as on it |
| **intrinsics** | `fx`, `fy`, `cx`, `cy`: the numbers that link pixels to directions from the camera |
| **mask / outline** | the exact pixels of one object, from SAM 3.1 |
| **in the way** | a box nearer by more than 1.5 cm, at the same height, and across from another box with less than 3 cm to spare |
| **normal** | the direction sticking straight out of a flat face |
| **PCA** | principal component analysis: finds the direction a set of points spreads most |
| **pitch** | how far the camera is tilted down, in degrees |
| **plane** | a flat surface: `n · p + d = 0` |
| **project** | turn a 3D point into a pixel |
| **RANSAC** | find the plane most points agree with by trying many random 3-point planes |
| **recording** | a folder of frames from `record_rgbd.py`: colour + depth + meta for each |
| **SAM 3.1** | Meta's Segment Anything Model 3.1: outlines objects named in a text prompt |
| **score** | how sure SAM is that an outline is a box, 0 to 1 |
| **segmentation** | finding the exact pixels of each object |
| **SVD** | a matrix method; here it finds the direction points spread least, the plane's normal |
| **target** | the one box chosen for the robot to pick next |
| **up** | the direction opposite gravity, in camera coordinates; from the accelerometer or `--pitch` |
| **view angle** | the angle between the camera's line of sight and a face (for a top: at its front edge); under 10° = almost edge-on |
