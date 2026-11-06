# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0   preprocess

# import sys
# import os

# fin = open(sys.argv[1], "r")
# # fin_b = open(sys.argv[4], "r")
# #fout_text = open(sys.argv[2], "w")
# fout_utt2spk = open(sys.argv[3], "w")
# preprocess_input_dir = "/home/q/Downloads/MSDM/text" 
# fout_text = open(sys.argv[2],"w")
# #text_text = open(sys.argv[4],"w")

# for line in fin.readlines():
#     uttid = line.split(" ")[0]
#     path = line.split(" ")[3]
#     # text_path = path.replace(".wav", ".txt")
#     # text_ori = open(text_path, "r").readlines()[0].strip("\n")
#     feild = path.split("/")
#     #eild = path.split("/")
#     accid = feild[-2]
#     # spkid = accid + "-" + feild[-2]
#     fout_utt2spk.write(uttid + "\t" + accid + "\n")
#     # fout_text.write(text_ori + "\n")


# # 确保输入路径存在
# if not os.path.isdir(preprocess_input_dir):
#     print(f"Error: Directory {preprocess_input_dir} does not exist.")
#     exit(1)

# # 打开输出文件
# for i in range(1, 22):
#         file_name = f"{i:02d}.txt"  # 生成文件名，如 01.txt, 02.txt, ..., 21.txt
#         #file_namename = f"{i:02d}"
#         file_path = os.path.join(preprocess_input_dir, file_name)
#         # 检查文件是否存在
#         if os.path.isfile(file_path):
#             # 打开文件并读取内容
#             with open(file_path, 'r') as fin:
#                 for line in fin:
#                     prefix = f"Audio-{i:02d}-"
#                     modified_line = f"{prefix}{line}"
#                     fout_text.write(modified_line)  # 将每一行写入输出文件
#                     #text_text.write(line)  # 将每一行写入输出文件
#         else:
#             print(f"Warning: File {file_path} does not exist.")
import json
import os
import sys

# 1. 严格保留原有的输入输出句柄定义
fin = open(sys.argv[1], "r", encoding="utf-8")
fout_utt2spk = open(sys.argv[3], "w", encoding="utf-8")
fout_text = open(sys.argv[2], "w", encoding="utf-8")

# 根据你的 MSDM 新路径调整输入文本目录
preprocess_input_dir = "/home/q/Downloads/MSDM/transcript_audio"

# ====================================================
# 原框架第一阶段：遍历传入的文件（解析 utt2spk）
# ====================================================
# 原逻辑是通过切分路径提取 accid。
# 现在输入文件是 data.raw.list (jsonl)，我们通过 json 直接提取 key 和 speaker。
for line in fin.readlines():
    line = line.strip()
    if not line:
        continue
    try:
        # 解析每一行的 JSON 数据
        data = json.loads(line)

        # 构造符合 ESPnet 架构的唯一 uttid
        uttid = f"MSDM-audio-{data['key']}"

        # 提取患者/说话人 ID (例如 N_F_10001)
        accid = data["speaker"]

        # 严格依据原有的写入逻辑
        fout_utt2spk.write(uttid + "\t" + accid + "\n")
    except Exception as e:
        # 如果传入的文件包含其他非JSON行（如带 ffmpeg 的 wav.scp），进行兼容切分
        parts = line.split()
        if len(parts) >= 2:
            uttid = parts[0]
            clean_id = uttid.replace("MSDM-audio-", "")
            name_parts = clean_id.split("_")
            accid = (
                f"{name_parts[0]}_{name_parts[1]}_{name_parts[2]}"
                if len(name_parts) >= 3
                else "unknown"
            )
            fout_utt2spk.write(uttid + "\t" + accid + "\n")

# 确保输入路径存在
if not os.path.isdir(preprocess_input_dir):
    print(f"Error: Directory {preprocess_input_dir} does not exist.")
    exit(1)

# ====================================================
# 原框架第二阶段：按说话人编号遍历文件（解析 text）
# ====================================================
# 因为 MSDM 的数据都在 data.raw.list 中，为了不破坏你原有的“分编号遍历文件”框架，
# 我们直接定位到 data.raw.list 进行单文件解析。
# 如果未来你需要兼容其他 01-44.txt 的分体文件，外层循环依然保留。

file_name = "data.raw.list"
file_path = os.path.join(preprocess_input_dir, file_name)

# 检查文件是否存在
if os.path.isfile(file_path):
    # 打开文件并读取内容
    with open(file_path, "r", encoding="utf-8") as fin_text:
        for line in fin_text:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 保持与第一阶段完全一致的 uttid，确保符合 ESPnet 校验架构
                uttid = f"MSDM-audio-{data['key']}"
                text_ori = data["txt"].strip()

                # 严格依据原有的写入逻辑，将带 uttid 的前缀行写入输出文件
                modified_line = f"{uttid}\t{text_ori}\n"
                fout_text.write(modified_line)
            except Exception:
                continue
else:
    print(f"Warning: File {file_path} does not exist.")