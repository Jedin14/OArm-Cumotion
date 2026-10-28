# 4. Faces and planes

**In short:** a box face is flat. We turn the outline's pixels into 3D points, find the flat surfaces among them (**planes**), decide which way each faces (front, top or side), and measure each one as a rectangle in millimetres.

Code: [`faces.py`](../../src/faces.py), function `box_faces`. Tests: [`test_faces.py`](../../src/test_faces.py).

## A plane and its normal

A **plane** is a flat surface. It's described by its **normal** `n`, a direction (length 1) sticking straight out of it, and an offset `d`. A point `p` lies on the plane when

```
n · p + d = 0
```

and `|n · p + d|` is the point's distance from the plane. The normal is what tells us which way a face points.

## Finding planes: RANSAC

Depth points are noisy and a box shows up to 3 faces, so we can't fit one plane to all points. **RANSAC** ("random sample consensus") finds the plane most points agree with:

1. Pick 3 random points. They define a plane.
2. Count the points within the threshold of it (the **inliers**). The threshold is 1% of the distance, at least 4 mm.
3. Repeat 200 times. Keep the plane with the most inliers.
4. Refine it: the best-fitting normal through all inliers is the direction in which they spread least (an **SVD** of the points).

Then take away the points close to that plane and look for the next one, up to 3 planes. Only points within **half** the threshold are taken away: a full-width band would also take the next face's points along the shared edge, and a narrow face (a box's 2 cm end) would be left too thin to fit.

Two fixes on top of plain RANSAC:

- **A plane found twice is skipped.** The points left in the band around a found plane can make a second, nearly identical one (within 10°). Kept, it would split the face's points in two.
- **Tops are set exactly level.** A plane within 15° of up is a top, and a box top lies flat, so its direction is set to exactly up and only its height is fitted. Depth on a top seen at an angle comes out slightly bent, and without this its fitted tilt swings by several degrees from frame to frame, moving its size and centre by centimetres. This needs the right up (the accelerometer, or `--pitch`): if up is wrong, tops come out wrong.

At the end, every point goes to the plane it's nearest to, and each face keeps its largest connected piece.

## Front, top or side

We need to know which way is **up** (from the accelerometer, or assumed; note 1). Then:

| Face | Its normal points | Picked? |
|---|---|---|
| **top** | within 30° of up | yes, from above |
| **front** | within 30° of towards the camera (level) | yes, from the front |
| **side** | anything else | never |

A box can't show two faces pointing the same way, so two such planes are merged or one is dropped.

## Measuring a face: a rectangle in mm

The face's 3D points are laid flat in the plane (coordinates in mm), and a rectangle is fitted:

- **Direction:** for a long face, the direction its points spread most (**PCA**, principal component analysis). For a nearly square face, the tightest rectangle around the points. A top seen with few points takes its direction from the line where it meets the box's front, because box faces meet at right angles.
- **Extent:** from where the points are **dense**. Across a real face, every 1 mm slice holds about the same number of points. A bump on the outline or a few stray points hold far fewer (less than 15% of a typical slice), so they're cut off.

The rectangle gives the face's **size** (long × short edge, mm) and **centre**: the point the robot aims at, drawn as the cross.

## Filling the depth shadow

A strip along a face's left or right edge with no depth (the depth shadow, note 1) would make the face too narrow and shift its centre. Each pixel in such a strip is put where its ray meets the face's plane (note 2), but only if:

- the strip runs from the face to the edge of the outline, along a row;
- it is no wider than a shadow can physically be there, `0.095 m × fx × |1/z_face − 1/z_next| + 4` pixels, with `z_next` the depth just past the outline. A wider strip is something else (a side seen edge-on, a gap), so it's left alone;
- the camera sees the face at 30° or more;
- it can't be the box's own side: a side seen at a steep angle has no depth either. The camera only sees a box's side on the edge facing the middle of the view (a box left of centre shows its right side). On that edge the strip is filled only if something nearer stands right beside it, whose shadow it is.

## Seen almost edge-on

The **view angle** is the angle between the camera's line of sight and the face. For a top it is measured at its **front edge**: the far part of a top seen at a slant often has no depth, so the centre of what was found moves back and forth between frames, but the front edge stays put. A top whose front edge is seen at under 10° has little depth and is measured badly. Note 5 explains what happens then.

**A top seen at a slant is re-measured.** Its depth comes in pieces (the front strip, a strip at the back, or the whole top, changing from frame to frame), so its middle would jump by several cm. Instead: the front edge is where the front face is, the height is the front's top edge (or the top's own points, whichever is higher), and the back edge comes from the box's outline. The top is level, so the outline's top row in the picture lies at one distance. The grip point is halfway between. Code: `top_from_outline` in [`faces.py`](../../src/faces.py).
