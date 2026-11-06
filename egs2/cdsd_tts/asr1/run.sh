#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#--inference_asr_model "3epoch.pth" \
#    --gpu_inference true \
#--max_wav_duration 30 \
set -u
set -o pipefail
export HF_ENDPOINT="https://hf-mirror.com"
# Patient to process: change to "02", "03", etc. when needed.
# Original patient data must already be prepared before augmentation.
export PATIENT_ID="01"
# Generated ratio means fraction of the available generated audio, not original audio.
use_generated=true
generated_ratio=1
generated_seed=42
original_train_set="train${PATIENT_ID}"
generated_data_dir="/home/q/tts/outputs/G1_200/espnet"

# Consume augmentation options; pass normal ESPnet options through unchanged.
asr_args=()
requested_stage=3
while [ "$#" -gt 0 ]; do
    case "$1" in
        --use-generated|--generated-ratio|--generated-seed|--generated-data-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            case "$1" in
                --use-generated) use_generated=$2 ;;
                --generated-ratio) generated_ratio=$2 ;;
                --generated-seed) generated_seed=$2 ;;
                --generated-data-dir) generated_data_dir=$2 ;;
            esac
            shift 2 ;;
        *)
            if [ "$1" = --stage ] && [ "$#" -ge 2 ]; then requested_stage=$2; fi
            asr_args+=("$1")
            shift ;;
    esac
done
case "$use_generated" in
    true|false) ;;
    *) echo "--use-generated must be true or false" >&2; exit 1 ;;
esac
train_set="$original_train_set"
if [ "$use_generated" = true ]; then
    if [[ ! "$requested_stage" =~ ^[0-9]+$ ]] || [ "$requested_stage" -lt 3 ]; then
        echo "Augmentation requires existing original data; start at stage 3 or later." >&2
        exit 1
    fi
    ratio_tag=$(python3 -c 'from decimal import Decimal; import sys; r=Decimal(sys.argv[1]); assert r.is_finite() and 0 <= r <= 1, "ratio must be between 0 and 1"; print(format(r.normalize(), "f").replace(".", "p"))' "$generated_ratio") || exit 1
    train_set="${original_train_set}_aug_r${ratio_tag}_s${generated_seed}"
    python3 local/prepare_augmented_train.py \
        --original "data/${original_train_set}" \
        --generated "$generated_data_dir" --output "data/${train_set}" \
        --ratio "$generated_ratio" --seed "$generated_seed" \
        --held-out data/valid data/test || exit 1
fi
valid_set="valid"
#test_sets="test"
test_sets="test valid"

asr_config=conf/tuning/train_asr_whisper_tiny_fft.yaml
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
    --max_wav_duration 15 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --asr_tag "$(basename "$asr_config" .yaml)_${train_set}" \
    --asr_stats_dir "exp/asr_stats_${train_set}_raw_zh_whisper_multilingual" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "${asr_args[@]}"

