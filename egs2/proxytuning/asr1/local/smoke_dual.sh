#!/usr/bin/env bash
# Two real training utterances, one validation utterance, then checkpoint resume.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
py=${DUAL_PYTHON:-python3}
export DUAL_SMOKE_DIR=${DUAL_SMOKE_DIR:-exp/dual_whisper_smoke}
"$py" - <<'PY'
from pathlib import Path
import os
p = Path(os.environ["DUAL_SMOKE_DIR"])
p.mkdir(exist_ok=True)
for split, n in [('train', 2), ('valid', 1)]:
    keys = [row.split()[0] for row in Path(f'dump/raw/{split}/wav.scp').read_text().splitlines()[:n]]
    for name in ('wav.scp', 'text'):
        rows = [r for r in Path(f'dump/raw/{split}/{name}').read_text().splitlines() if r.split()[0] in keys]
        (p / f'{split}.{name}').write_text('\n'.join(rows) + '\n')
    rows = [r for r in Path(f'exp/asr_stats_raw_zh_whisper_multilingual/{split}/speech_shape').read_text().splitlines() if r.split()[0] in keys]
    (p / f'{split}.shape').write_text('\n'.join(rows) + '\n')
PY
opts=(
    --config conf/tuning/train_asr_whisper_dual.yaml
    --token_list data/zh_token_list/whisper_multilingual/tokens.txt
    --token_type whisper_multilingual --bpemodel whisper_multilingual
    --output_dir ${DUAL_SMOKE_DIR}/model
    --num_workers 0 --batch_type unsorted --valid_batch_type unsorted
    --batch_size 1 --accum_grad 2 --log_interval 1
    --model_conf validation_max_tokens=8
    --train_data_path_and_name_and_type ${DUAL_SMOKE_DIR}/train.wav.scp,speech,kaldi_ark
    --train_data_path_and_name_and_type ${DUAL_SMOKE_DIR}/train.text,text,text
    --train_shape_file ${DUAL_SMOKE_DIR}/train.shape
    --valid_data_path_and_name_and_type ${DUAL_SMOKE_DIR}/valid.wav.scp,speech,kaldi_ark
    --valid_data_path_and_name_and_type ${DUAL_SMOKE_DIR}/valid.text,text,text
    --valid_shape_file ${DUAL_SMOKE_DIR}/valid.shape
)
./local/dual_python.sh -m espnet2.bin.asr_train "${opts[@]}" --max_epoch 1 "$@" > ${DUAL_SMOKE_DIR}/train.log 2>&1
./local/dual_python.sh -m espnet2.bin.asr_train "${opts[@]}" --max_epoch 2 --resume true "$@" > ${DUAL_SMOKE_DIR}/resume.log 2>&1
./local/dual_python.sh -m espnet2.bin.asr_inference \
    --asr_train_config ${DUAL_SMOKE_DIR}/model/config.yaml \
    --asr_model_file ${DUAL_SMOKE_DIR}/model/valid.loss.ave.pth \
    --config conf/whisper_decode_dual.yaml --maxlenratio -8 \
    --data_path_and_name_and_type ${DUAL_SMOKE_DIR}/valid.wav.scp,speech,kaldi_ark \
    --num_workers 0 --ngpu "${DUAL_SMOKE_NGPU:-0}" \
    --output_dir ${DUAL_SMOKE_DIR}/decode > ${DUAL_SMOKE_DIR}/inference.log 2>&1
PYTHONPATH="local:${PYTHONPATH:-}" "$py" -m dual_whisper.inspect_smoke
