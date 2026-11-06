#!/usr/bin/env bash
set -euo pipefail
./asr.sh \
    --python "$PWD/local/dual_python.sh" \
    --stage 12 --stop_stage 13 \
    --inference_args "--inference_fusion_alpha 0.5 --num_workers 0" \
    --inference_tag decode_smalllora_aligned_alpha05 \
    --use_proxy_tuning false \
    --lang zh --ngpu 1 --nj 2 \
    --token_type whisper_multilingual \
    --inference_nj 4 --gpu_inference true \
    --feats_normalize "" --max_wav_duration 15 \
    --audio_format flac.ark --feats_type raw --use_lm false \
    --asr_config conf/tuning/train_asr_whisper_dual_smalllora_aligned.yaml \
    --asr_stats_dir exp/asr_stats_raw_zh_whisper_multilingual \
    --asr_speech_fold_length 800 --asr_text_fold_length 150 \
    --inference_config conf/whisper_decode_dual_smallfft_aligned.yaml \
    --inference_asr_model valid.acc.ave.pth \
    --asr_exp exp/asr_whisper_dual_smalllora_aligned \
    --train_set train --valid_set valid \
    --eval_valid_set false --test_sets "test org/valid" \
    --lm_train_text data/train/text --bpe_train_text data/train/text "$@"
