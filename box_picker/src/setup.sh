#!/usr/bin/env bash
# Create .venv (Python 3.11) with the box picker's packages, and get the SAM 3.1 weights (~3.5 GB).
#   ./setup.sh              install + download the weights (no Hugging Face account needed)
#   ./setup.sh --no-model   install only (then copy the weights in: see INTEGRATION.md, "Offline")
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_HTTP_TIMEOUT=600   # large CUDA wheels can time out on slow links
uv venv --python 3.11 .venv
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements.txt
uv pip install --python .venv --no-deps \
    "sam3 @ git+https://github.com/facebookresearch/sam3@7f473cfd983d5eb43040ff608f66bff0d7bdcb53"

.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

if [[ "${1:-}" != "--no-model" ]]; then
    .venv/bin/python -c "from box_masks import sam31_checkpoint; print('SAM 3.1 weights:', sam31_checkpoint())"
fi
echo "Done. Check:  .venv/bin/python test_faces.py   and   .venv/bin/python pick_frame.py --live --frames 5"
