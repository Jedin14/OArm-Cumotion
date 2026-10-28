# The size of the object the arm picks next (or holds now), mm. Published on
# /beast_arm/held_object/size. On a felt contact the pick stack attaches a box
# of this size to that arm's tool for the planners, so the way back keeps the
# box clear of everything; released at the drop pose.
#
#   height: vertical, as the object stands
#   width:  sideways, across the camera's view
#   depth:  away from the camera (along a front approach)
#
# All zero = no object / unknown: nothing is attached.

# frame_id: who says so ("dummy" from the GUI's test toggle)
std_msgs/Header header
float32 height_mm
float32 width_mm
float32 depth_mm
