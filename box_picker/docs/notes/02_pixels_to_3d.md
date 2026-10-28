# 2. From a pixel to a 3D point

**In short:** with four numbers about the camera (its **intrinsics**), a pixel and its depth become a point in metres. The robot needs metres, not pixels.

## Camera coordinates

The code uses the RealSense convention, measured from the camera, in metres:

| Axis | Points |
|---|---|
| x | right |
| y | down |
| z | forward, away from the camera |

So `(-0.05, 0.03, 0.34)` is 5 cm left, 3 cm down and 34 cm in front of the camera. Every target the pipeline outputs (`x_m`, `y_m`, `z_m`) is in these coordinates. The robot side converts them to its own.

## The pinhole camera and its intrinsics

A camera is modelled as a **pinhole**: every point in the scene sends a ray through one point (the centre of the lens) onto the image. Four numbers describe how the image is laid out:

| Name | Meaning | This D455 at 1280×720 |
|---|---|---|
| `fx`, `fy` | focal length, in pixels | about 645 |
| `cx`, `cy` | the pixel the camera looks straight through (the image centre, roughly) | about 644, 369 |

They come from the camera itself and are saved in every `*_meta.json`.

## Pixel → 3D point (deproject)

For a pixel (u, v) with depth z:

```
x = (u − cx) × z / fx
y = (v − cy) × z / fy
z = z
```

Example: a pixel 100 pixels right of the centre, at 0.5 m, is `100 × 0.5 / 645 ≈ 0.078 m` (7.8 cm) to the right.

This is `deproject` in [`faces.py`](../../src/faces.py). It's why a box looks smaller in pixels when it's further away, but measures the same in millimetres.

## 3D point → pixel (project)

The other way round:

```
u = fx × x / z + cx
v = fy × y / z + cy
```

This is `Intrinsics.project`. The code uses it to draw the measured face rectangle back onto the picture.

## A ray

Every pixel is a **ray** from the camera: `((u − cx)/fx, (v − cy)/fy, 1)`, times any distance. When a pixel has no depth, but we know which flat face it lies on, we can still get its 3D point: where its ray meets the face's plane. That's how the depth-shadow strips are filled in (note 4).

## Rule of thumb

At 0.5 m, 1 pixel is about 0.8 mm. A 28 mm box edge is about 36 pixels wide.
