# import torch
# import matplotlib.pyplot as plt

# # 1. 加载模型权重
# model_path = "/home/q/espnet/egs2/cdsds/asr1/exp/shiyan/asr_train_asr_whisper_small_deegora_raw_zh_whisper_multilingual32rank+48rank2A/valid.acc.ave.pth"
# state_dict = torch.load(model_path, map_location="cpu")

# if "model" in state_dict:
#     state_dict = state_dict["model"]

# # 2. 定向提取精选的 4 个层，并使用极简的学术缩写作为键值
# selected_layers = {
#     "encoder.encoders.blocks.0.attn.value": "Enc-B0 (Attn)",
#     "encoder.encoders.blocks.7.attn.value": "Enc-B7 (Attn)",
#     "decoder.decoders.blocks.0.attn.value": "Dec-B0 (Attn)",
#     "decoder.decoders.blocks.3.mlp.2":     "Dec-B3 (MLP)"
# }

# plot_labels = []
# w1_weights = []
# w2_weights = []

# # 按照 selected_layers 的顺序提取，确保横坐标逻辑从前到后
# for key_name in selected_layers.keys():
#     weight_key = key_name + ".lora_expert"
#     if weight_key in state_dict:
#         value = state_dict[weight_key]
#         scales = torch.softmax(value, dim=0)
#         w1_weights.append(scales[0].item())
#         w2_weights.append(scales[1].item())
#         plot_labels.append(selected_layers[key_name])

# # 3. 绘制精炼的学术柱状图
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.unicode_minus'] = False

# # 调整画布比例 (6x4.5)，让整体结构更紧凑
# fig, ax = plt.subplots(figsize=(6, 4.5))
# colors = ['#4A90E2', '#E06666'] # 学术蓝 vs 暖砖红

# # 堆叠柱状图 (略微调窄柱子宽度至 0.4，视觉上更精致)
# ax.bar(plot_labels, w1_weights, label='$w_1$ (Standard)', color=colors[0], width=0.4, edgecolor='white', linewidth=0.5)
# ax.bar(plot_labels, w2_weights, bottom=w1_weights, label='$w_2$ (Pathological)', color=colors[1], width=0.4, edgecolor='white', linewidth=0.5)

# # ✨ 精简后的标题与纵坐标（标准 IEEE/ACM 期刊风格）
# ax.set_title('Expert Weight Distribution Across Representative Layers', fontsize=11, fontweight='bold', pad=12)
# ax.set_ylabel('Expert Weight', fontsize=10)
# ax.set_ylim(0, 1.05)

# # 横坐标标签完全水平居中排列，不再需要旋转，极其整齐
# plt.xticks(rotation=0, ha='center')
# ax.grid(axis='y', linestyle='--', alpha=0.4)

# # 在柱子内部打上具体的数值标签
# for i in range(len(plot_labels)):
#     ax.text(i, w1_weights[i]/2, f"{w1_weights[i]:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
#     ax.text(i, w1_weights[i] + w2_weights[i]/2, f"{w2_weights[i]:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

# # 图例放在图表正下方
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10)
# plt.tight_layout()

# # 保存
# output_img = 'selected_expert_specialization_clean.png'
# plt.savefig(output_img, dpi=300, bbox_inches='tight')
# print(f"🎉 简洁版定量证据图已生成: {output_img}")
import os
import glob
import numpy as np
import torch
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

# 引入 ESPnet 的 ASR 任务构建器
from espnet2.tasks.asr import ASRTask

# ==========================================
# 1. 路径与设备配置（已配置为你指定的路径）
# ==========================================
AUDIO_ROOT = "/home/q/Downloads/CDSD/CDSD-Interspeech/after_catting/1h/Audio"
EXP_DIR = "/home/q/espnet/egs2/cdsds/asr1/exp/shiyan/asr_train_asr_whisper_small_lora_raw_zh_whisper_multilingual96rank8bs"

CONFIG_FILE = os.path.join(EXP_DIR, "config.yaml")
CHECKPOINT_FILE = os.path.join(EXP_DIR, "valid.acc.ave.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 检查路径是否存在
# ==========================================
for path in [AUDIO_ROOT, CONFIG_FILE, CHECKPOINT_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到路径，请检查: {path}")

# ==========================================
# 3. 离线加载 ESPnet 模型（自动合并 LoRA 权重）
# ==========================================
print(f"正在从本地离线加载 ESPnet 模型 (设备: {DEVICE})...")
model, asr_train_args = ASRTask.build_model_from_file(
    CONFIG_FILE,
    CHECKPOINT_FILE,
    DEVICE
)
model.eval()
encoder = model.encoder

# ==========================================
# 4. 扫描说话人文件夹，每个说话人仅保留前 2 条语音
# ==========================================
speaker_dirs = sorted([
    d for d in os.listdir(AUDIO_ROOT) 
    if os.path.isdir(os.path.join(AUDIO_ROOT, d))
])

speaker_data = {}
for spk in speaker_dirs:
    spk_path = os.path.join(AUDIO_ROOT, spk)
    # 按字母顺序排序，确保每次运行拿到的前2条音频是一致的
    wav_files = sorted(glob.glob(os.path.join(spk_path, "*.wav")))
    if len(wav_files) > 0:
        speaker_data[spk] = wav_files[:2]

print(f"成功扫描到 {len(speaker_data)} 个说话人。")
print("每个说话人将提取前 2 条语音进行特征融合。")

# ==========================================
# 5. 提取特征并融合生成说话人声纹画像
# ==========================================
speaker_embeddings = {}
spk_ids = []

print("\n>>> 开始提取特征...")
with torch.no_grad():
    for spk_id, wav_paths in tqdm(speaker_data.items(), desc="进度"):
        single_spk_embeddings = []
        
        for wav_path in wav_paths:
            try:
                # 1. 载入音频并强制重采样为 16kHz
                speech, sr = librosa.load(wav_path, sr=16000)
                
                # 2. 消除静音段（top_db=30db 过滤病态发音中的长静音段干扰）
                speech, _ = librosa.effects.trim(speech, top_db=30)
                if len(speech) < 1600:  # 过滤掉短于 0.1 秒的无效切片
                    continue
                
                # 3. 转换成 PyTorch Tensor 并送入 GPU/CPU
                speech_tensor = torch.tensor(speech, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                lengths = torch.tensor([speech_tensor.shape[1]], dtype=torch.long, device=DEVICE)
                
                # 4. 前向传播：使用 ESPnet 封装的 whisper_encoder 抽取特征
                # 输出形状为: (Batch, Time, Dimension)
                encoder_outputs, encoder_lens, _ = encoder(speech_tensor, lengths)
                
                # 5. 时间维度平均池化 (Mean Pooling) 得到当前句子的 768 维表征
                sentence_embedding = torch.mean(encoder_outputs, dim=1).squeeze(0)
                single_spk_embeddings.append(sentence_embedding.cpu().numpy())
                
            except Exception as e:
                print(f"\n[Warning] 无法处理文件: {wav_path}，错误信息: {e}")
                continue
                
        if len(single_spk_embeddings) > 0:
            # 【声纹纯化】：对当前说话人的所有特征取算术平均，压制不同词汇语义带来的偏差
            mean_embedding = np.mean(single_spk_embeddings, axis=0)
            speaker_embeddings[spk_id] = mean_embedding
            spk_ids.append(spk_id)

# ==========================================
# 6. 特征归一化与 K-Means 聚类
# ==========================================
X = np.array([speaker_embeddings[sid] for sid in spk_ids])

# L2 归一化：将所有特征向量缩放到单位球面上，使 K-Means 计算余弦相似度（夹角大小），防止声音大小干扰分类
X_normalized = normalize(X, norm='l2')

print(f"\n>>> 正在将 {len(spk_ids)} 个说话人划分为 2 个聚类群...")
# 使用 K-Means，K=2，设置 n_init=20 以保证聚类稳定性
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_normalized)

# ==========================================
# 7. 输出划分结果
# ==========================================
group_0 = []
group_1 = []

for spk_id, label in zip(spk_ids, cluster_labels):
    if label == 0:
        group_0.append(spk_id)
    else:
        group_1.append(spk_id)

# 按照文件夹数字（01-44）大小进行升序排序，方便观察
group_0.sort(key=int)
group_1.sort(key=int)

print("\n" + "="*50)
print("🎯 说话人二分类聚类结果")
print("="*50)
print(f"【第一组 (Group 0)】 (共 {len(group_0)} 人):")
print(", ".join(group_0))
print("-" * 50)
print(f"【第二组 (Group 1)】 (共 {len(group_1)} 人):")
print(", ".join(group_1))
print("="*50)