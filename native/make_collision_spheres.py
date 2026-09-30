#!/usr/bin/env python3
"""Fit cuRobo collision spheres to this robot's own collision meshes.

Why this exists: openarm.yml had no collision_spheres block at all, and
cuRobo represents the robot as spheres for *both* self-collision and world
collision. With none defined there is nothing on the robot to collide
with, so cuMotion returned "Trajectory success!" for paths that put the
gripper through the torso -- measured twice, left arm 1790058419 and right
arm 1790164017, each caught only by MoveIt's own FCL validator afterwards
and surfacing as a late error code rather than a planning failure.

The spheres are fitted to the meshes the URDF already ships, not authored
by hand: made-up collision geometry is worse than none, because it is
believed.

Writes the block to stdout. Counts scale with each mesh's volume so that
the torso is not described by the same number of spheres as a finger.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH = [os.path.join(WS, 'src'), WS]


def resolve(filename):
    rel = filename.replace('package://', '')
    for root in SEARCH:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return path
    return None


def rpy_xyz(origin):
    if origin is None:
        return [0.0] * 3, [0.0] * 3
    xyz = [float(v) for v in (origin.get('xyz') or '0 0 0').split()]
    rpy = [float(v) for v in (origin.get('rpy') or '0 0 0').split()]
    return xyz, rpy


def collisions(urdf):
    """(link name, mesh path, xyz, rpy, scale) for every collision mesh."""
    out = []
    for link in ET.parse(urdf).getroot().findall('link'):
        for col in link.findall('collision'):
            mesh = col.find('geometry/mesh')
            if mesh is None:
                continue
            path = resolve(mesh.get('filename'))
            if path is None:
                print(f'# missing mesh: {mesh.get("filename")}',
                      file=sys.stderr)
                continue
            xyz, rpy = rpy_xyz(col.find('origin'))
            scale = mesh.get('scale')
            scale = ([float(v) for v in scale.split()] if scale
                     else [1.0, 1.0, 1.0])
            out.append((link.get('name'), path, xyz, rpy, scale))
    return out


def adjacency(urdf, srdf=None):
    """Link pairs that must not be checked against each other.

    The SRDF's own disable_collisions list when there is one, because it is
    the authoritative answer for this robot: MoveIt's setup assistant
    computes it by sampling the whole configuration space, and it covers
    both "Adjacent" pairs and the much larger set of "Never" pairs -- links
    that cannot reach each other however the arm is posed.

    Deriving it from URDF joints alone was tried and is not enough. That
    gives only the 22 adjacent pairs against the SRDF's 139, and cuRobo
    then treats every permanently-overlapping pair as a collision: measured
    on run 1790241099, every plan came back MotionGenStatus.GRAPH_FAIL and
    neither arm could reach its own staging pose.

    Falls back to joint adjacency if no SRDF is given, which is better than
    nothing but is why the fallback warns.
    """
    pairs = {}

    def note(a, b):
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)

    for joint in ET.parse(urdf).getroot().findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is not None and child is not None:
            note(parent.get('link'), child.get('link'))
    if srdf and os.path.exists(srdf):
        found = 0
        for entry in ET.parse(srdf).getroot().findall('disable_collisions'):
            note(entry.get('link1'), entry.get('link2'))
            found += 1
        print(f'# {found} disable_collisions pairs from the SRDF',
              file=sys.stderr)
    else:
        print('# WARNING: no SRDF, so only joint adjacency is ignored. '
              'cuRobo will very likely fail every plan with GRAPH_FAIL.',
              file=sys.stderr)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(WS, 'openarm.urdf'))
    ap.add_argument('--srdf', default=os.path.join(
        WS, 'src', 'openarm_ros2', 'openarm_bimanual_moveit_config', 'config',
        'openarm_bimanual.srdf'))
    # Sized by each mesh's own length scale, not by a shared budget.
    # Volume-proportional sharing was tried first and starved the arm: the
    # torso took nearly all of it and a forearm came back as four 5 mm
    # spheres, which represents a 60 mm limb about as well as nothing does.
    ap.add_argument('--spacing', type=float, default=0.018,
                    help='roughly one sphere per this much of a link, metres')
    ap.add_argument('--min-per-link', type=int, default=10)
    ap.add_argument('--max-per-link', type=int, default=60)
    ap.add_argument('--surface-radius', type=float, default=0.005)
    args = ap.parse_args()

    from curobo.geom.types import Mesh
    from curobo.geom.sphere_fit import SphereFitType
    from curobo.types.math import Pose
    from curobo.types.base import TensorDeviceType
    import numpy as np

    tensor_args = TensorDeviceType()
    items = collisions(args.urdf)
    if not items:
        sys.exit('no collision meshes found')

    # One sphere per `spacing` of the mesh's own extent, summed over the
    # three axes so a long thin link gets spheres along its length and a
    # bulky one gets them in every direction.
    counts = []
    for _, path, _, _, scale in items:
        mesh = Mesh(name='m', file_path=path, pose=[0, 0, 0, 1, 0, 0, 0],
                    scale=list(scale))
        extents = mesh.get_trimesh_mesh().bounding_box.extents
        want = int(round(float(sum(extents)) / args.spacing))
        counts.append(max(args.min_per_link, min(args.max_per_link, want)))

    print('    # Generated by native/make_collision_spheres.py from the')
    print('    # collision meshes in openarm.urdf. Regenerate rather than')
    print('    # editing by hand -- see that script for why.')
    print('    collision_spheres:')
    per_link = {}
    for (name, path, xyz, rpy, scale), count in zip(items, counts):
        pose = Pose.from_list([xyz[0], xyz[1], xyz[2], 1.0, 0.0, 0.0, 0.0])
        if any(rpy):
            from scipy.spatial.transform import Rotation as R
            quat = R.from_euler('xyz', rpy).as_quat()      # x y z w
            pose = Pose.from_list([xyz[0], xyz[1], xyz[2],
                                   float(quat[3]), float(quat[0]),
                                   float(quat[1]), float(quat[2])])
        mesh = Mesh(name=name, file_path=path, pose=[0, 0, 0, 1, 0, 0, 0],
                    scale=list(scale))
        spheres = mesh.get_bounding_spheres(
            n_spheres=count,
            surface_sphere_radius=args.surface_radius,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            pre_transform_pose=pose,
            tensor_args=tensor_args)
        kept = [s for s in spheres if s.radius > 0.0]
        per_link.setdefault(name, []).extend(kept)

    for name, spheres in per_link.items():
        print(f'      {name}:')
        for s in spheres:
            c = [float(v) for v in np.ravel(s.position)[:3]]
            print(f'        - center: [{c[0]:.5f}, {c[1]:.5f}, {c[2]:.5f}]')
            print(f'          radius: {float(s.radius):.5f}')

    print('    collision_link_names: ['
          + ', '.join(f'"{n}"' for n in per_link) + ']')

    near = adjacency(args.urdf, args.srdf)
    print('    # Pairs not checked against each other, taken from the')
    print("    # SRDF's own disable_collisions list plus URDF joint")
    print('    # adjacency. Joint adjacency alone is not enough -- it')
    print('    # misses the much larger set of pairs that can never reach')
    print('    # each other, and without those every plan fails GRAPH_FAIL.')
    print('    self_collision_ignore:')
    for name in per_link:
        others = sorted(n for n in near.get(name, ()) if n in per_link)
        print(f'      {name}: [' + ', '.join(f'"{n}"' for n in others) + ']')
    print('    self_collision_buffer:')
    for name in per_link:
        print(f'      {name}: 0.002')
    total_spheres = sum(len(v) for v in per_link.values())
    print(f'# {total_spheres} spheres across {len(per_link)} links',
          file=sys.stderr)


if __name__ == '__main__':
    main()
