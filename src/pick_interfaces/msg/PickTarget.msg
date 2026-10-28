# What the box model found: the point to pick, and how to approach it.

# frame_id: the world frame the position is in ("world"); stamp: the image's
std_msgs/Header header

# the point the tool touches, m, in header.frame_id
#   top:   on the box's top, 2 cm in from its near edge
#   front: on the face the camera sees, at the box's centre
geometry_msgs/Point position

# "top"   - straight down onto the top (a box lying flat)
# "front" - level, from the camera's side
string approach
string APPROACH_TOP = "top"
string APPROACH_FRONT = "front"

# "left" / "right", or "" to let the pick stack choose (by the image half)
string arm

# the detection, in the colour image (pixels, origin top-left)
float32 confidence
float32 u                 # the box's centre
float32 v
float32[4] bbox           # x1 y1 x2 y2
uint32 image_width
uint32 image_height
uint32 num_boxes          # every box the model saw in the frame

# the gripping tool -- it also fixes the arm: "vacuum" -> "left", "finger" -> "right"
# (arm above is always filled in to match it)
string tool
string TOOL_VACUUM = "vacuum"
string TOOL_FINGER = "finger"

# the face to grip as the camera measured it, world frame; zeros if not measured
float32[2] face_mm                  # its long and short edge, mm
geometry_msgs/Vector3 face_normal   # unit, out of the face (towards the camera)
geometry_msgs/Vector3 face_long     # unit, along its long edge

# the known box (held_object/size) matched to that face, as it will hang on the
# tool: mm along face_long, across it, and into the box (along -face_normal).
# zeros: no size known, or the face matched none of the box's faces
float32[3] box_mm
