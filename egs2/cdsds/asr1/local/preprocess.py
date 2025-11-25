# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0   preprocess

import sys
import os

fin = open(sys.argv[1], "r")
# fin_b = open(sys.argv[4], "r")
#fout_text = open(sys.argv[2], "w")
fout_utt2spk = open(sys.argv[3], "w")
preprocess_input_dir = "/home/q/Downloads/CDSD/CDSD-Interspeech/after_catting/1h/text"
# preprocess_input_dir = "/root/shared-data/zhangxiaoqing-data/CDSD/CDSD-Interspeech/after_catting/1h/text"
fout_text = open(sys.argv[2],"w")
#text_text = open(sys.argv[4],"w")

for line in fin.readlines():
    uttid = line.split(" ")[0]
    path = line.split(" ")[3]
    # text_path = path.replace(".wav", ".txt")
    # text_ori = open(text_path, "r").readlines()[0].strip("\n")
    feild = path.split("/")
    #eild = path.split("/")
    accid = feild[-2]
    # spkid = accid + "-" + feild[-2]
    fout_utt2spk.write(uttid + "\t" + accid + "\n")
    # fout_text.write(text_ori + "\n")



# 确保输入路径存在
if not os.path.isdir(preprocess_input_dir):
    print(f"Error: Directory {preprocess_input_dir} does not exist.")
    exit(1)

# 打开输出文件
for i in range(1, 45):
        file_name = f"{i:02d}.txt"  # 生成文件名，如 01.txt, 02.txt, ..., 44.txt
        #file_namename = f"{i:02d}"
        file_path = os.path.join(preprocess_input_dir, file_name)
        #print(f"Warning: {file_path} ")
        # 检查文件是否存在
        if os.path.isfile(file_path):
            # 打开文件并读取内容
            with open(file_path, 'r') as fin:
                for line in fin:
                    prefix = f"Audio-{i:02d}-"
                    modified_line = f"{prefix}{line}"
                    fout_text.write(modified_line)  # 将每一行写入输出文件
                    #text_text.write(line)  # 将每一行写入输出文件
        else:
            print(f"Warning: File {file_path} does not exist.")

print(f"All lines from 01.txt to 44.txt have been written to {fout_text}.")