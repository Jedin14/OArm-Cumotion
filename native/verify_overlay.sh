#!/usr/bin/env bash
#
# Check that every shared library in the overlay (and in the native workspace
# build) can actually resolve its dependencies under the native environment.
#
# This catches the failure mode apt cannot: ROS debs declare unversioned
# dependencies, so apt happily accepts an older host copy of a package even when
# the new binaries are linked against a newer soname. The result is a library
# that installs fine and fails to dlopen at launch time -- typically surfacing as
# a MoveIt plugin that silently does not load.
#
# For each unresolved soname the likely deb name is suggested; add it to TARGETS
# in fetch_debs.sh, then re-run fetch/extract.
#
set -uo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${NATIVE_DIR}/root/opt/ros/humble"

set +u
# shellcheck disable=SC1091
source "${NATIVE_DIR}/setup.bash" > /dev/null
set -u

echo "== checking shared library resolution =="

# torch's own libraries (libc10, libtorch_cpu, ...) are not on LD_LIBRARY_PATH:
# python resolves them when `import torch` runs. Add them here so ldd reports on
# the same footing the runtime does, instead of flagging every torch extension.
TORCH_LIB="$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

declare -A missing_by_lib=()
checked=0
broken=0

while IFS= read -r so; do
    checked=$((checked + 1))
    while IFS= read -r soname; do
        [[ -z "${soname}" ]] && continue
        broken=$((broken + 1))
        missing_by_lib["${soname}"]+="$(basename "${so}") "
    done < <(ldd "${so}" 2>/dev/null | awk '/not found/{print $1}')
done < <(find "${PREFIX}/lib" "${NATIVE_DIR}/install" \
             -name '*.so' -o -name '*.so.*' 2>/dev/null)

echo "   scanned ${checked} shared objects"

if [[ ${#missing_by_lib[@]} -eq 0 ]]; then
    echo "   all dependencies resolve"
else
    echo
    echo "   ${#missing_by_lib[@]} unresolved soname(s), ${broken} broken link(s):"
    for soname in "${!missing_by_lib[@]}"; do
        # libgeometric_shapes.so.2.3.4 -> ros-humble-geometric-shapes
        stem="${soname#lib}"
        stem="${stem%%.so*}"
        suggestion="ros-humble-$(echo "${stem}" | tr '_' '-')"
        echo
        echo "   ${soname}"
        echo "     wanted by : ${missing_by_lib[${soname}]}"
        echo "     try adding: ${suggestion}"
        host_ver="$(dpkg-query -W -f='${Version}' "${suggestion}" 2>/dev/null || true)"
        if [[ -n "${host_ver}" ]]; then
            echo "     host has  : ${host_ver}  (too old -- the overlay must carry a newer one)"
        fi
    done
fi

# --- python import check ----------------------------------------------------
echo
echo "== checking key python imports =="
python3 - <<'PY'
mods = [
    ("numpy", "1.26.4"),
    ("torch", "2.7.0+cu128"),
    ("warp", None),
    ("curobo", None),
    ("trimesh", None),
    ("yourdfpy", None),
    ("rclpy", None),
    ("isaac_ros_cumotion", None),
]
bad = 0
for name, want in mods:
    try:
        m = __import__(name)
        got = getattr(m, "__version__", "?")
        flag = ""
        if want and got != want:
            flag = f"  <-- expected {want}"
            bad += 1
        print(f"   ok   {name:22s} {got}{flag}")
    except Exception as exc:
        print(f"   FAIL {name:22s} {type(exc).__name__}: {exc}")
        bad += 1

import torch
print(f"   cuda available: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
else:
    print()
    bad += 1
raise SystemExit(1 if bad else 0)
PY
py_status=$?

echo
if [[ ${#missing_by_lib[@]} -eq 0 && ${py_status} -eq 0 ]]; then
    echo "== overlay verified =="
    exit 0
fi
echo "== overlay has problems (see above) =="
exit 1
