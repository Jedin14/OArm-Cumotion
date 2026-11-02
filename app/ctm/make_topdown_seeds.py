#!/usr/bin/env python3
"""Precompute starting postures for top-down IK: ctm/topdown_seeds.npz.

MoveIt's KDL IK only searches locally, and from the arms' usual postures
(tool level) it never finds the narrow set of postures that point the tool
straight down -- 0 of 20 random seeds either (2026-10-08). cuRobo's IK,
with hundreds of seeds at once, does. This runs it over a grid in front of
each arm, for the eight tool turns click_to_move tries, and stores what it
found; solve_ik() then starts from the nearest stored posture. Rerun it after
moving the arm mounts in the URDF:

    source native/setup.bash && python3 ctm/make_topdown_seeds.py
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctm.common import WS, quat_from_matrix, tool_frame_along  # noqa: E402

OUT = os.path.join(WS, 'ctm', 'topdown_seeds.npz')
YAWS = (0, 90, -90, 180, 45, -45, 135, -135)


def main():
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    tensor = TensorDeviceType()
    data = {}
    for arm in ('left', 'right'):
        rc = RobotConfig.from_basic(os.path.join(WS, 'openarm.urdf'), 'world',
                                    f'openarm_{arm}_hand_tcp', tensor)
        ik = IKSolver(IKSolverConfig.load_from_robot_config(
            rc, None, num_seeds=200, self_collision_check=False, use_cuda_graph=False,
            tensor_args=tensor, position_threshold=0.002, rotation_threshold=0.02))
        side = 1.0 if arm == 'left' else -1.0
        grid = np.array([(x, side * y, z) for x in np.arange(0.10, 0.601, 0.05)
                         for y in np.arange(-0.15, 0.451, 0.05)
                         for z in np.arange(0.15, 0.851, 0.05)])
        found_p, found_yaw, found_q = [], [], []
        for yaw in YAWS:
            x, y, z, w = quat_from_matrix(tool_frame_along(np.array([0.0, 0.0, -1.0]),
                                                           math.radians(yaw)))
            oks, qs = [], []
            for a in range(0, len(grid), 150):          # chunks: the GPU is shared
                chunk = grid[a:a + 150]
                pos = torch.tensor(chunk, dtype=torch.float32, device='cuda')
                quat = torch.tensor([[w, x, y, z]] * len(chunk), dtype=torch.float32,
                                    device='cuda')
                r = ik.solve_batch(Pose(pos, quat))
                oks.append(r.success.view(-1).cpu().numpy())
                qs.append(r.solution.view(len(chunk), -1).cpu().numpy())
            ok, q = np.concatenate(oks), np.concatenate(qs)
            found_p.append(grid[ok]); found_yaw += [yaw] * int(ok.sum()); found_q.append(q[ok])
            print(f'{arm} yaw {yaw:+4d}: {int(ok.sum())} of {len(grid)} grid points', flush=True)
        data[f'{arm}_tcp'] = np.concatenate(found_p)
        data[f'{arm}_yaw'] = np.array(found_yaw)
        data[f'{arm}_q'] = np.concatenate(found_q)
    np.savez_compressed(OUT, **data)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
