#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#--inference_asr_model "3epoch.pth" \
#    --gpu_inference true \
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test"
#test_sets="test valid"

asr_config=conf/tuning/train_asr_whisper_small_lora.yaml
inference_config=conf/whisper_decode_asr.yaml

./asr.sh \
    --stage 10 \
    --stop_stage 13 \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang zh \
    --ngpu 1 \
    --nj 2 \
    --token_type whisper_multilingual \
    --inference_nj 4 \
    --gpu_inference true \
    --nbpe 5000 \
    --feats_normalize "" \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" 
