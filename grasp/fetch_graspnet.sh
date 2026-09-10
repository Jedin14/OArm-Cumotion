#!/usr/bin/env bash
#
# Fetch and build graspnet-baseline, so grasp/run_grasp_server.sh has
# something to run. Idempotent: re-running skips whatever is already there.
#
#   grasp/fetch_graspnet.sh
#
# LICENCE. graspnet-baseline is Shanghai Jiao Tong University's, under an
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY licence.
# Read third_party/graspnet-baseline/LICENSE. Running this script downloads it
# and thereby agrees to those terms -- so do not run it unless your use really
# is noncommercial research. Nothing it fetches is committed here; third_party
# is .gitignored, precisely so this repository does not redistribute it.
#
# What it does, and why each step is needed:
#
#   1. Clones graspnet-baseline.
#   2. Patches two torch APIs that were removed years after the code was
#      written -- .data<T>() and .type().is_cuda(). 58 sites, mechanical.
#      There are no THC headers, which is the usual thing that makes code of
#      this age unbuildable.
#   3. Makes a venv with its own torch. Not the cuRobo one and not the VLM
#      one: the extensions have to be compiled against exactly the torch that
#      will import them.
#   4. Builds pointnet2 and knn for the GPU actually present. Without
#      TORCH_CUDA_ARCH_LIST the extension builds for whatever the toolkit
#      defaults to, and a missing kernel shows up as "no kernel image is
#      available" deep inside inference rather than as a build error.
#   5. Downloads checkpoint-rs.tar, the RealSense-trained weights -- the right
#      ones for a D455.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WS="$(cd "${HERE}/.." && pwd)"
THIRD="${WS}/third_party"
REPO="${THIRD}/graspnet-baseline"
VENV="${THIRD}/grasp_venv"
WEIGHTS="${THIRD}/weights/checkpoint-rs.tar"
# checkpoint-rs.tar on the authors' Google Drive, linked from their README.
WEIGHTS_ID="1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk"

command -v uv > /dev/null || {
    echo "error: uv is not on PATH. install.sh puts it there." >&2
    exit 1
}

mkdir -p "${THIRD}/weights"

echo "== 1/5  graspnet-baseline"
if [[ -d "${REPO}/.git" ]]; then
    echo "   already cloned"
else
    git clone --depth 1 https://github.com/graspnet/graspnet-baseline "${REPO}"
fi

echo "== 2/5  patching the removed torch APIs"
python3 "${THIRD}/patch_graspnet.py"

echo "== 3/5  the environment"
if [[ -x "${VENV}/bin/python3" ]]; then
    echo "   already there"
else
    export UV_LINK_MODE=copy
    uv venv --python 3.10 "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    # cu128 because this is a Blackwell card; the default wheel has no sm_120.
    uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
    # open3d is what the collision detector wants; graspnetAPI is deliberately
    # not installed -- the one class needed from it is a five-line adapter in
    # grasp_node.py over the raw prediction array.
    uv pip install numpy scipy Pillow tqdm open3d
    deactivate
fi

echo "== 4/5  the CUDA extensions"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
if python3 -c "import pointnet2._ext, knn_pytorch" 2> /dev/null; then
    echo "   already built"
else
    ARCH="$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f'{major}.{minor}')
else:
    print('')
PY
)"
    if [[ -z "${ARCH}" ]]; then
        echo "error: no CUDA device visible, so there is no architecture to" >&2
        echo "build for. The extensions are GPU-only." >&2
        exit 1
    fi
    echo "   building for sm_${ARCH/./}"
    export TORCH_CUDA_ARCH_LIST="${ARCH}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    (cd "${REPO}/pointnet2" && python3 setup.py install)
    (cd "${REPO}/knn" && python3 setup.py install)
    python3 -c "import pointnet2._ext, knn_pytorch; print('   extensions import')"
fi

echo "== 5/5  weights"
if [[ -s "${WEIGHTS}" ]]; then
    echo "   already downloaded"
else
    export UV_LINK_MODE=copy
    uv tool run --from gdown gdown \
        "https://drive.google.com/uc?id=${WEIGHTS_ID}" -O "${WEIGHTS}"
fi
python3 - "${WEIGHTS}" <<'PY'
import sys
import zipfile
path = sys.argv[1]
# A torch checkpoint is a zip. A Google Drive quota page is HTML, and it
# arrives with a 200 and a plausible size -- so check the shape, not just that
# a file exists.
if not zipfile.is_zipfile(path):
    sys.exit(f'error: {path} is not a torch checkpoint. Google Drive '
             f'sometimes serves an HTML quota page instead; delete it and '
             f'try later, or use the Baidu link in the upstream README.')
print('   checkpoint looks like a torch archive')
PY

deactivate
echo
echo "done. Start it with:  grasp/run_grasp_server.sh"
