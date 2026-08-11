#!/usr/bin/env python3
"""Start move_group headless and check the cuMotion planning pipeline loads.

Why this exists: openarm_bimanual_moveit_config's sensors_3d.yaml enables
occupancy_map_monitor/DepthImageOctomapUpdater, whose mesh self-filter uses
freeglut/OpenGL and therefore needs an X display. Launched from a terminal with
no DISPLAY it prints "freeglut failed to open display ''" and takes move_group
down with it, before any planner plugin is loaded -- which makes
demo.launch.py useless as a headless check.

This builds the same MoveIt configuration with the 3D sensors stripped out, so
move_group survives on a headless box and the planning pipelines (ompl, chomp,
pilz, cumotion) can be observed loading.

    python3 native/tests/test_move_group_planners.py

Exits 0 if move_group comes up and reports the cuMotion pipeline.
"""
import os
import re
import subprocess
import sys
import tempfile

import yaml
from moveit_configs_utils import MoveItConfigsBuilder

PIPELINES = ["ompl", "chomp", "pilz_industrial_motion_planner", "cumotion"]
SENSOR_KEYS_PREFIX = ("sensors", "octomap_", "max_range", "realsense_depth")


def build_params() -> dict:
    configs = (
        MoveItConfigsBuilder("openarm", package_name="openarm_bimanual_moveit_config")
        .planning_pipelines(pipelines=PIPELINES, default_planning_pipeline="cumotion")
        .to_moveit_configs()
    )
    params = configs.to_dict()

    # Drop everything that would start an octomap updater.
    removed = [k for k in params if k.startswith(SENSOR_KEYS_PREFIX)]
    for key in removed:
        del params[key]
    print(f"stripped 3D sensor params: {removed or '(none present)'}")

    params["use_sim_time"] = False
    return params


def main() -> int:
    params = build_params()

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"move_group": {"ros__parameters": params}}, fh)
        params_file = fh.name

    cmd = ["ros2", "run", "moveit_ros_move_group", "move_group",
           "--ros-args", "--params-file", params_file]
    print("running:", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    seen = []
    want = re.compile(r"cumotion|Using planning pipeline|planning pipeline", re.I)
    crashed_on_glut = False
    try:
        for line in proc.stdout:
            seen.append(line.rstrip())
            if "freeglut" in line:
                crashed_on_glut = True
            if want.search(line):
                print("  >", line.rstrip())
            if "MoveGroup context initialization complete" in line:
                print("\nmove_group came up cleanly")
                proc.terminate()
                break
            if len(seen) > 400:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.unlink(params_file)

    joined = "\n".join(seen)
    ok = "MoveGroup context initialization complete" in joined
    has_cumotion = "cumotion" in joined.lower()

    print()
    if crashed_on_glut:
        print("FAIL: freeglut/display error still present")
    print(f"move_group initialized : {ok}")
    print(f"cuMotion pipeline seen : {has_cumotion}")
    if not ok:
        print("\n--- last 25 lines ---")
        print("\n".join(seen[-25:]))
    return 0 if (ok and has_cumotion) else 1


if __name__ == "__main__":
    sys.exit(main())
