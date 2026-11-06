#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
alpha_threads=${OMP_NUM_THREADS:-2}
. ./path.sh
set -u
export OMP_NUM_THREADS=$alpha_threads
export PYTHONPATH="$PWD/local:${PYTHONPATH:-}"
exec "${DUAL_PYTHON:-python3}" -m dual_whisper.alpha_sweep "$@"
