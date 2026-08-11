#!/usr/bin/env bash
#
# Unpack every deb from native/debs into native/root, producing a workspace-local
# ament overlay prefix at native/root/opt/ros/humble that layers on top of the
# host's /opt/ros/humble. Nothing is registered with dpkg and nothing lands
# outside this directory.
#
# A per-package file listing is kept in native/logs/contents/ so it stays obvious
# which file came from which package.
#
set -euo pipefail

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEB_DIR="${NATIVE_DIR}/debs"
ROOT_DIR="${NATIVE_DIR}/root"
PREFIX="${ROOT_DIR}/opt/ros/humble"

if [[ -d "${ROOT_DIR}" ]]; then
    echo "== removing previous extraction at ${ROOT_DIR} =="
    rm -rf "${ROOT_DIR}"
fi
mkdir -p "${ROOT_DIR}" "${NATIVE_DIR}/logs/contents"

shopt -s nullglob
debs=("${DEB_DIR}"/*.deb)
if [[ ${#debs[@]} -eq 0 ]]; then
    echo "no debs in ${DEB_DIR}; run ./fetch_debs.sh first" >&2
    exit 1
fi

echo "== unpacking ${#debs[@]} debs into ${ROOT_DIR} =="
for deb in "${debs[@]}"; do
    name="$(basename "${deb}" .deb)"
    dpkg-deb -c "${deb}" | awk '{print $6}' > "${NATIVE_DIR}/logs/contents/${name}.txt"
    dpkg -x "${deb}" "${ROOT_DIR}"
done

# --- relocation -------------------------------------------------------------
# These debs were built to live at /opt/ros/humble. Under the overlay they live
# somewhere else, so any absolute reference to their own prefix inside CMake
# config, pkg-config or ament marker files has to be rewritten. Other packages
# are still resolved through AMENT_PREFIX_PATH / CMAKE_PREFIX_PATH, so only
# self-references matter here -- but a blanket rewrite is safe because a path
# that does not exist in the overlay simply is not used: CMake config files for
# host packages are read from the host prefix, not from ours.
echo "== rewriting absolute prefix references =="
mapfile -t reloc_files < <(
    find "${PREFIX}" \
        \( -name '*.cmake' -o -name '*.pc' -o -name '*.dsv' \
           -o -name '*.sh' -o -name '*.bash' -o -name '*.zsh' \) -type f
)
rewritten=0
for f in "${reloc_files[@]}"; do
    if grep -q '/opt/ros/humble' "${f}" 2>/dev/null; then
        # Only rewrite a path if the target actually exists in the overlay.
        python3 - "${f}" "${PREFIX}" <<'PY'
import re, sys, os
path, prefix = sys.argv[1], sys.argv[2]
with open(path, 'r', errors='surrogateescape') as fh:
    text = fh.read()

def repl(m):
    tail = m.group(1)
    candidate = os.path.join(prefix, tail.lstrip('/'))
    # Rewrite only when this overlay really provides the referenced path.
    if os.path.exists(candidate):
        return prefix + tail
    return m.group(0)

new = re.sub(r'/opt/ros/humble(/[^\s"\'();:]*)?', lambda m: repl(m) if m.group(1) else m.group(0), text)
if new != text:
    with open(path, 'w', errors='surrogateescape') as fh:
        fh.write(new)
    print(path)
PY
        rewritten=$((rewritten + 1))
    fi
done
echo "   inspected ${#reloc_files[@]} files, ${rewritten} contained prefix references"

# Python entry-point scripts installed by the debs carry a `#!/usr/bin/python3`
# shebang. They must run under the workspace venv (pinned numpy/torch), so point
# them at it via a relative-free absolute path into native/venv.
echo "== retargeting node script shebangs at the workspace venv =="
retargeted=0
while IFS= read -r script; do
    if head -c 2 "${script}" 2>/dev/null | grep -q '#!' && \
       head -1 "${script}" | grep -qE '^#!.*python3?$'; then
        sed -i "1s|.*|#!${NATIVE_DIR}/venv/bin/python3|" "${script}"
        retargeted=$((retargeted + 1))
    fi
done < <(find "${PREFIX}/lib" -maxdepth 2 -type f -perm -u+x 2>/dev/null)
echo "   retargeted ${retargeted} node scripts"

# --- dangling development symlinks -----------------------------------------
# A -dev deb ships libfoo.so -> libfoo.so.N.M, where the versioned file belongs
# to the runtime deb. When the host already has the runtime package (freeglut3,
# for instance) only the -dev half is unpacked here, leaving a symlink pointing
# at a file that is not in the overlay. Linking against it then fails. Retarget
# those at the host's real library.
echo "== repairing dangling -dev symlinks against host libraries =="
repaired=0
still_broken=0
while IFS= read -r link; do
    [[ -e "${link}" ]] && continue          # resolves fine already
    target="$(readlink "${link}")"
    base="$(basename "${target}")"
    # Look for the versioned library in the usual host locations.
    for cand in "/usr/lib/x86_64-linux-gnu/${base}" "/usr/lib/${base}" \
                "/opt/ros/humble/lib/${base}" "/opt/ros/humble/lib/x86_64-linux-gnu/${base}"; do
        if [[ -e "${cand}" ]]; then
            ln -sf "${cand}" "${link}"
            repaired=$((repaired + 1))
            break
        fi
    done
    [[ -e "${link}" ]] || { echo "   still dangling: ${link} -> ${target}"; still_broken=$((still_broken + 1)); }
done < <(find "${ROOT_DIR}" -xtype l 2>/dev/null)
echo "   repaired ${repaired}, still dangling ${still_broken}"

# A merged deb install has per-package share/<pkg>/local_setup.bash but no
# prefix-level one, which makes colcon warn that the prefix "doesn't contain any
# 'local_setup.*' files". setup.bash already exports everything this prefix
# needs, so a no-op marker is enough to keep colcon quiet and honest.
for ext in sh bash zsh; do
    cat > "${PREFIX}/local_setup.${ext}" <<'EOF'
# Intentionally empty.
#
# This prefix is a merged unpack of Debian packages (see native/extract_debs.sh),
# not a colcon install tree. Everything it needs -- AMENT_PREFIX_PATH,
# LD_LIBRARY_PATH, PYTHONPATH, CMAKE_PREFIX_PATH -- is exported by
# native/setup.bash. This file exists only so that colcon and ament recognise the
# prefix instead of warning about a missing local_setup.
EOF
done

echo
echo "== overlay ready =="
echo "   prefix:  ${PREFIX}"
echo "   packages: $(ls -1 "${PREFIX}/share/ament_index/resource_index/packages" 2>/dev/null | wc -l)"
du -sh "${ROOT_DIR}"
