#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [TRAIN_ARGS...]" >&2
    exit 2
fi

CONFIG=$1
GPUS=$2
shift 2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PORT=${PORT:-29245}

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
python -m torch.distributed.launch \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/train.py" \
    "${CONFIG}" \
    --launcher pytorch \
    "$@"

