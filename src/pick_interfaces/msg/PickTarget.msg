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
