#!/usr/bin/env bash
set -u
set -o pipefail
export HF_ENDPOINT="https://hf-mirror.com"

train_set="train"
valid_set="valid"
test_sets="test valid"

# =============================================================
# 0. 自动热修复: 解决 asr.py 中残留的 proxy_encoder 属性缺失报错
# =============================================================
ython3 -c "
asr_file = '/home/q/espnet/espnet2/tasks/asr.py'
with open(asr_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
fixed = False
for line in lines:
    if 'proxy_encoder_class = proxy_encoder_choices.get_class' in line:
        indent = line[:len(line) - len(line.lstrip())]
        # 用 getattr 单行安全读取，如果不存在则赋值为 None,保持严格相同的缩进
        new_lines.append(f'{indent}proxy_encoder_class = proxy_encoder_choices.get_class(args.proxy_encoder) if getattr(args, \"proxy_encoder\", None) is not None else None\n')
        fixed = True
    elif 'if hasattr(args, \"proxy_encoder\")' in line:
        # 清理上一次生成的残留行
        continue
    else:
        new_lines.append(line)

if fixed:
    with open(asr_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('>>> [自动热修复] asr.py 缩进与安全检查修复成功！')
"

# -------------------------------------------------------------
# 1. 专家模型（微调过的 Tiny）路径配置
# -------------------------------------------------------------
expert_exp="exp/asr_train_asr_whisper_small_finetune_raw_zh_whisper_multilingual_8:1:_1e-4_tiny"
expert_train_config="${expert_exp}/config.yaml"
expert_model_file="${expert_exp}/valid.acc.ave.pth"

# -------------------------------------------------------------
# 2. 自动化构建 Anti-Expert 模型（官方纯净未微调 Tiny）
# -------------------------------------------------------------
anti_model_dir="exp/asr_tiny_pretrained"
anti_train_config="${expert_train_config}"
anti_model_file="${anti_model_dir}/model.pth"

if [ ! -f "${anti_model_file}" ]; then
    echo ">>> [1/2] 正在自动提取官方纯净 Tiny 权重..."
    mkdir -p "${anti_model_dir}"
    python3 -c "
import torch
whisper_raw = torch.load('/home/q/.cache/whisper/tiny.pt', map_location='cpu')
torch.save(whisper_raw['model_state_dict'], '${anti_model_file}')
print('>>> Anti-Expert Tiny 纯净权重准备完成！')
"
fi

# -------------------------------------------------------------
# 3. 自动化构建 Base 模型（官方纯净未微调 Small）
# -------------------------------------------------------------
base_exp="exp/asr_whisper_small_zeroshot"
base_config="${base_exp}/config.yaml"
base_model_file="${base_exp}/model.pth"

if [ ! -f "${base_config}" ] || [ ! -f "${base_model_file}" ]; then
    echo ">>> [2/2] 正在基于 Tiny 模板自动生成纯净 Small 配置与权重..."
    mkdir -p "${base_exp}"
    sed 's/whisper_model: tiny/whisper_model: small/g' "${expert_train_config}" > "${base_config}"
    
    python3 -c "
import torch, whisper
try:
    whisper_raw = torch.load('/home/q/.cache/whisper/small.pt', map_location='cpu')
except Exception:
    print('>>> 未在缓存中检测到 small.pt,正在通过 whisper 库加载...')
    _m = whisper.load_model('small', download_root='/home/q/.cache/whisper')
    whisper_raw = torch.load('/home/q/.cache/whisper/small.pt', map_location='cpu')

torch.save(whisper_raw['model_state_dict'], '${base_model_file}')
print('>>> Base Small 纯净模型准备完成！')
"
fi

# -------------------------------------------------------------
# 4. 推理参数打包透传
# -------------------------------------------------------------
inference_config=conf/whisper_decode_asr.yaml
extra_inference_args="--expert_train_config ${expert_train_config} --expert_model_file ${expert_model_file} --anti_train_config ${anti_train_config} --anti_model_file ${anti_model_file}"

# -------------------------------------------------------------
# 5. 启动 Stage 12 解码与 Stage 13 评估
# -------------------------------------------------------------
./asr.sh \
    --stage 12 \
    --stop_stage 13 \
    --asr_exp "${base_exp}" \
    --inference_asr_model "model.pth" \
    --use_proxy_tuning true \
    --skip_data_prep true \
    --skip_train true \
    --skip_eval false \
    --lang zh \
    --ngpu 1 \
    --nj 2 \
    --token_type whisper_multilingual \
    --inference_nj 2 \
    --gpu_inference true \
    --nbpe 5000 \
    --feats_normalize "" \
    --max_wav_duration 15 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --inference_args "${extra_inference_args}" "$@"

