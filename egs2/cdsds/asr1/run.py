import torch
import matplotlib.pyplot as plt

# 1. 加载模型权重
model_path = "/home/q/espnet/egs2/cdsds/asr1/exp/shiyan/asr_train_asr_whisper_small_deegora_raw_zh_whisper_multilingual32rank+48rank2A/valid.acc.ave.pth"
state_dict = torch.load(model_path, map_location="cpu")

if "model" in state_dict:
    state_dict = state_dict["model"]

# 2. 定向提取精选的 4 个层，并使用极简的学术缩写作为键值
selected_layers = {
    "encoder.encoders.blocks.0.attn.value": "Enc-B0 (Attn)",
    "encoder.encoders.blocks.7.attn.value": "Enc-B7 (Attn)",
    "decoder.decoders.blocks.0.attn.value": "Dec-B0 (Attn)",
    "decoder.decoders.blocks.3.mlp.2":     "Dec-B3 (MLP)"
}

plot_labels = []
w1_weights = []
w2_weights = []

# 按照 selected_layers 的顺序提取，确保横坐标逻辑从前到后
for key_name in selected_layers.keys():
    weight_key = key_name + ".lora_expert"
    if weight_key in state_dict:
        value = state_dict[weight_key]
        scales = torch.softmax(value, dim=0)
        w1_weights.append(scales[0].item())
        w2_weights.append(scales[1].item())
        plot_labels.append(selected_layers[key_name])

# 3. 绘制精炼的学术柱状图
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 调整画布比例 (6x4.5)，让整体结构更紧凑
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ['#4A90E2', '#E06666'] # 学术蓝 vs 暖砖红

# 堆叠柱状图 (略微调窄柱子宽度至 0.4，视觉上更精致)
ax.bar(plot_labels, w1_weights, label='$w_1$ (Standard)', color=colors[0], width=0.4, edgecolor='white', linewidth=0.5)
ax.bar(plot_labels, w2_weights, bottom=w1_weights, label='$w_2$ (Pathological)', color=colors[1], width=0.4, edgecolor='white', linewidth=0.5)

# ✨ 精简后的标题与纵坐标（标准 IEEE/ACM 期刊风格）
ax.set_title('Expert Weight Distribution Across Representative Layers', fontsize=11, fontweight='bold', pad=12)
ax.set_ylabel('Expert Weight', fontsize=10)
ax.set_ylim(0, 1.05)

# 横坐标标签完全水平居中排列，不再需要旋转，极其整齐
plt.xticks(rotation=0, ha='center')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 在柱子内部打上具体的数值标签
for i in range(len(plot_labels)):
    ax.text(i, w1_weights[i]/2, f"{w1_weights[i]:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    ax.text(i, w1_weights[i] + w2_weights[i]/2, f"{w2_weights[i]:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

# 图例放在图表正下方
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10)
plt.tight_layout()

# 保存
output_img = 'selected_expert_specialization_clean.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"🎉 简洁版定量证据图已生成: {output_img}")