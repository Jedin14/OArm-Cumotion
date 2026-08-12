#!/usr/bin/env bash
# Start the VLM detector node in VLM/.venv. This is what pick_place.launch.py
# runs; run_in_vlm_env.sh does the environment work and explains why it is
# needed.
set -euo pipefail

VLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${VLM_DIR}/run_in_vlm_env.sh" "${VLM_DIR}/vlm_detector_node.py" "$@"
