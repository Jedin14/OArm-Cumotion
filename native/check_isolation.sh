#!/usr/bin/env bash
#
# Prove that the project's Python stacks do not overlap with the system's.
#
#   ./check_isolation.sh
#
# This workspace runs three mutually incompatible Python stacks on one machine:
#
#   native/venv   numpy 1.26.4, torch 2.7.0+cu128   -- the ABI cuRobo's prebuilt
#                                                      CUDA kernels are linked
#                                                      against. Not negotiable.
#   VLM/.venv     numpy 2.2.6,  torch 2.14 +cu130   -- what PaliGemma needs.
#   the host      numpy 1.21.5 (apt) and 2.2.6 (~/.local), torch 2.12+cu130
#
# Mixing any two of them produces either a numpy ABI error on import or a
# silently wrong torch. This script checks that each stack resolves to its own
# copies and that neither venv can see the host's ~/.local -- so a failure here
# is an early warning for a crash that would otherwise show up mid-launch.
#
# Exit 0 = isolated. Exit 1 = something leaks.
#
set -uo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_VENV="${WS_DIR}/native/venv/bin/python3"
VLM_VENV="${WS_DIR}/VLM/.venv/bin/python"

bad=0
note() { printf '   %-7s %s\n' "$1" "$2"; }

# Report on one interpreter: which numpy/torch it gets, from where, and whether
# the host's user-site directory is reachable.
#
#   $1 label   $2 interpreter   $3 expected numpy   $4 expected torch prefix
probe() {
    local label="$1" py="$2" want_np="$3" want_torch="$4"

    echo
    echo "-- ${label}"

    if [[ ! -x "${py}" ]]; then
        note MISSING "${py} does not exist -- run ./install.sh"
        bad=1
        return
    fi

    # PYTHONNOUSERSITE is what both runtime wrappers export; check under the
    # same conditions the real nodes run in. PYTHONPATH is cleared so an
    # already-active native env in this shell cannot skew the result.
    PYTHONNOUSERSITE=1 PYTHONPATH= "${py}" - \
        "${want_np}" "${want_torch}" "${WS_DIR}" <<'PY'
import sys, os

want_np, want_torch, ws = sys.argv[1], sys.argv[2], sys.argv[3]
bad = 0

def check(name, want):
    global bad
    try:
        mod = __import__(name)
    except Exception as exc:
        print(f"   FAIL    {name}: {type(exc).__name__}: {exc}")
        bad = 1
        return
    got = getattr(mod, "__version__", "?")
    path = getattr(mod, "__file__", "") or ""
    ok_ver = got.startswith(want)
    # The decisive test: the module must come from inside this workspace, not
    # from /usr/lib/python3/dist-packages or ~/.local.
    ok_src = path.startswith(ws)
    if ok_ver and ok_src:
        print(f"   ok      {name} {got}")
    else:
        if not ok_ver:
            print(f"   FAIL    {name} is {got}, expected {want}*")
        if not ok_src:
            print(f"   FAIL    {name} loaded from outside the workspace: {path}")
        bad = 1

check("numpy", want_np)
check("torch", want_torch)

leaks = [p for p in sys.path if "/.local/" in p]
if leaks:
    print(f"   FAIL    ~/.local is on sys.path: {leaks[0]}")
    bad = 1
else:
    print("   ok      ~/.local not on sys.path")

sysdist = [p for p in sys.path
           if p.endswith(("dist-packages", "site-packages")) and p.startswith("/usr/")]
if sysdist:
    # native/venv is built with --system-site-packages on purpose: it needs
    # rclpy and the ROS message modules from the host. That is fine as long as
    # numpy/torch above still resolved to the venv's own copies, which the
    # checks above already enforce.
    print(f"   note    system dist-packages visible (needed for rclpy): {sysdist[0]}")

raise SystemExit(bad)
PY
    [[ $? -ne 0 ]] && bad=1
}

echo
echo "==================================================================="
echo "  python stack isolation"
echo "==================================================================="

probe "native/venv   (cuMotion / cuRobo / MoveIt)" "${NATIVE_VENV}" "1.26.4" "2.7.0"
probe "VLM/.venv     (PaliGemma detector)"         "${VLM_VENV}"    "2.2"    "2.14"

# --- the two stacks must not be the same stack -----------------------------
echo
echo "-- cross-check"
if [[ -x "${NATIVE_VENV}" && -x "${VLM_VENV}" ]]; then
    n_np="$(PYTHONNOUSERSITE=1 PYTHONPATH= "${NATIVE_VENV}" -c 'import numpy;print(numpy.__version__)' 2>/dev/null)"
    v_np="$(PYTHONNOUSERSITE=1 PYTHONPATH= "${VLM_VENV}" -c 'import numpy;print(numpy.__version__)' 2>/dev/null)"
    if [[ -n "${n_np}" && -n "${v_np}" && "${n_np}" != "${v_np}" ]]; then
        note ok "the two venvs carry different numpy (${n_np} vs ${v_np}), as intended"
    else
        note FAIL "both venvs report numpy '${n_np}' / '${v_np}' -- they are not separate"
        bad=1
    fi
fi

# --- nothing of ours should have been installed system-wide -----------------
echo
echo "-- host cleanliness"
if [[ -d /usr/lib/python3/dist-packages ]]; then
    strays=""
    for pkg in curobo warp transformers paligemma; do
        [[ -e "/usr/lib/python3/dist-packages/${pkg}" ]] && strays+="${pkg} "
    done
    if [[ -n "${strays}" ]]; then
        note FAIL "project packages found in system dist-packages: ${strays}"
        bad=1
    else
        note ok "no project package installed into system dist-packages"
    fi
fi

# The overlay must be a private prefix, never the host's /opt/ros/humble.
if [[ -d "${WS_DIR}/native/root/opt/ros/humble" ]]; then
    note ok "ROS overlay is private (native/root), host /opt/ros/humble untouched"
elif [[ -d "${WS_DIR}/native" ]]; then
    note note "native/root not built yet -- run ./install.sh --native-only"
fi

echo
if [[ ${bad} -eq 0 ]]; then
    echo "== isolated: no overlap between project and system libraries =="
    exit 0
fi
echo "== NOT isolated: see the FAIL lines above =="
exit 1
