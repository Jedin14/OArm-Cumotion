# The pick stack cannot take a box from where the robot stands (out of reach,
# or --adjust-after failures in a row on it): how far the base should move
# SIDEWAYS -- left or right only, never forward, back or a turn -- so the box
# lines up with where the picking arm works best.
# Robot frame (header.frame_id, "world" = the body): x forward, y left, z up.
std_msgs/Header header

string direction           # "left", "right", or "none" (already lined up, within 1 cm:
                           # moving sideways cannot help -- the box is too near or too far)
float32 distance_m         # how far to move that way, metres (>= 0)
float32 left_m             # the same as one signed number: + to the left, - to the right

geometry_msgs/Point target # the box (the point gripped), robot frame, before the move
geometry_msgs/Point goal   # where the arm wants it, robot frame (its sweet spot; only its y counts)
string arm                 # the arm that picks it ("left": vacuum)
string tool                # "vacuum" / "finger"
string approach            # "front" / "top"
uint32 failures            # picks in a row that did not take this box
string reason              # why: "unreachable", or the check's outcomes, e.g. "missed, moved"
