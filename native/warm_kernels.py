#!/usr/bin/env python3
"""Pre-compile every cuRobo CUDA extension into the workspace-local cache.

The cuRobo debs ship cubins for sm_75/86/89 and are linked against a different
libtorch ABI, so on this machine (RTX 5070 Ti, sm_120) cuRobo falls back to
JIT-compiling each extension on first use. Doing that lazily costs ~30 s per
module the first time a planner starts. Running this once up front moves that
cost here, into native/cache/torch_extensions.

Safe to re-run: already-built extensions are reused.
"""
import time

MODULES = [
    "curobo.curobolib.geom",
    "curobo.curobolib.kinematics",
    "curobo.curobolib.tensor_step",
    "curobo.curobolib.opt",
    "curobo.curobolib.ls",
]


def main() -> int:
    import importlib

    failures = []
    for name in MODULES:
        start = time.time()
        try:
            importlib.import_module(name)
            print(f"  ok    {name}  ({time.time() - start:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} module(s) failed: {', '.join(failures)}")
        return 1
    print("\nall cuRobo CUDA extensions built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
