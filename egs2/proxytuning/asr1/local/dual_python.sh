#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${recipe_dir}/local:${PYTHONPATH:-}"
if [[ "${1:-}" == -m ]]; then
    case "${2:-}" in
        espnet2.bin.asr_train) shift 2; exec "${DUAL_PYTHON:-python3}" -m dual_whisper.train "$@" ;;
        espnet2.bin.asr_inference) shift 2; exec "${DUAL_PYTHON:-python3}" -m dual_whisper.inference "$@" ;;
    esac
fi
exec "${DUAL_PYTHON:-python3}" "$@"
