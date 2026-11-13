# One detected object/obstacle. All geometry is in ObstacleArray.header.frame_id
# (the levelled, drift-free "world" frame of the stationary lidar).

uint32 id                          # tracker id, stable while the object stays tracked
bool confirmed                     # true once seen in >= min_hits consecutive-ish windows
float32 age                        # [s] since the track was created

geometry_msgs/Point position       # cluster centroid
geometry_msgs/Vector3 velocity     # [m/s] tracker estimate (0 until it has history)

geometry_msgs/Point center         # bounding-box centre
geometry_msgs/Vector3 size         # bounding-box extents along its own x/y (yawed) and world z
float32 yaw                        # [rad] box yaw about world z (PCA of the footprint)

geometry_msgs/Point closest_point  # cluster point closest to the lidar
float32 range                      # [m] distance from the lidar origin to closest_point
uint32 num_points                  # points in the cluster (accumulated window)
