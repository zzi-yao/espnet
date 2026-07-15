import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 加载你的真实 JSON 文件
json_path = "/home/q/espnet/egs2/cdsds/asr1/exp/asr_train_asr_whisper_small_gora_raw_zh_whisper_multilingual——32rankclip/gora_rank_comparison.json"
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 定义映射关系，精确对齐原图的横坐标组件名
enc_components = {
    "attn.query": "SA-Q", "attn.key": "SA-K", "attn.value": "SA-V", 
    "attn.out": "SA-O", "mlp.0": "MLP0", "mlp.2": "MLP2"
}

dec_components = {
    "attn.query": "SA-Q", "attn.key": "SA-K", "attn.value": "SA-V", "attn.out": "SA-O",
    "cross_attn.query": "CA-Q", "cross_attn.key": "CA-K", "cross_attn.value": "CA-V", "cross_attn.out": "CA-O",
    "mlp.0": "MLP0", "mlp.2": "MLP2"
}

# 初始化空矩阵（12层，从L0到L11）
enc_matrix = np.zeros((12, len(enc_components)))
dec_matrix = np.zeros((12, len(dec_components)))

# 3. 填充矩阵 (提取 pre_clipping_rank)
# 注意：原图是从下往上排列 (L0在最下面，L11在最上面)
for i in range(12):
    # Encoder 提取
    for j, (comp_key, _) in enumerate(enc_components.items()):
        full_key = f"encoder.encoders.blocks.{i}.{comp_key}"
        if full_key in raw_data:
            enc_matrix[i, j] = raw_data[full_key]["pre_clipping_rank"]
            
    # Decoder 提取
    for j, (comp_key, _) in enumerate(dec_components.items()):
        full_key = f"decoder.decoders.blocks.{i}.{comp_key}"
        if full_key in raw_data:
            dec_matrix[i, j] = raw_data[full_key]["pre_clipping_rank"]

# 4. 开始作图 (严格还原原图风格)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 创建画布，包含左右两个子图以及最右侧的 Colorbar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [6, 10]})

# 设置一个与原图色系一致的渐变色 (原图是浅黄到深砖红，这里用 YlOrRd 极其接近)
cmap_style = "YlOrRd"

# y轴标签 (L11到L0，反转矩阵以实现L11在顶部)
yticklabels = [f"L{i}" for i in range(11, -1, -1)]

# 绘制 Encoder
sns.heatmap(np.flipud(enc_matrix), annot=True, fmt=".1f", cmap=cmap_style, 
            xticklabels=list(enc_components.values()), yticklabels=yticklabels,
            cbar=False, ax=ax1, linewidths=0.2, linecolor='white', annot_kws={"size": 9})
ax1.set_title("Expert 1: Encoder (Pre-clipping)", fontsize=12, fontweight='bold', pad=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# 绘制 Decoder
# 通过 robust=True 或不设 vmax，让 Pre-clip 那些极大值（如超出48或64的部分）自然显现出更深的颜色
sns.heatmap(np.flipud(dec_matrix), annot=True, fmt=".1f", cmap=cmap_style, 
            xticklabels=list(dec_components.values()), yticklabels=yticklabels,
            cbar=True, ax=ax2, linewidths=0.2, linecolor='white', annot_kws={"size": 9},
            cbar_kws={'label': 'Pre-clipping Rank Value', 'pad': 0.08})
ax2.set_title("Expert 1: Decoder (Pre-clipping)", fontsize=12, fontweight='bold', pad=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# 全局微调
ax1.set_ylabel("Layer", fontsize=12, fontweight='bold')
plt.tight_layout()

# 保存为高质量 PNG
output_img = "gora_pre_clipping_heatmap.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"🎉 完美的 Pre-clipping 热力图已成功直出：{output_img}")