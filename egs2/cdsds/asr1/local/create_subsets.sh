#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
echo "First argument (\$1): $1"
if [ -d ${data} ];then
    # Generate a list of all wav files
    cat $data/data_all/wav.scp > $data/all.scp

    # 1. 抽取 test 集：每隔 10 行取 1 行（第 10, 20, 30, ... 行）
    # 使用 awk 实现间隔采样，NR 表示行号，NR % 10 == 0 表示行号能被 10 整除
    awk 'NR % 10 == 0' $data/all.scp > $data/test.scp

    # 2. 从 all.scp 中排除 test 集，得到剩余数据（用于后续划分 train 和 valid）
    ./utils/filter_scp.pl --exclude $data/test.scp $data/all.scp > $data/remaining.scp

    # 3. 从 remaining.scp 中抽取 valid 集：每隔 10 行取 1 行
    # 注意：此时 remaining.scp 的行号已经重新排列，直接对其进行间隔采样即可
    awk 'NR % 10 == 0' $data/remaining.scp > $data/valid.scp

    # 4. 从 remaining.scp 中排除 valid 集，得到最终的 train 集
    ./utils/filter_scp.pl --exclude $data/valid.scp $data/remaining.scp > $data/train.scp

    # Create the corresponding data directories
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
    ./utils/subset_data_dir.sh --utt-list $data/valid.scp $data/data_all $data/valid
    ./utils/subset_data_dir.sh --utt-list $data/test.scp $data/data_all $data/test

    echo "Data split into train, valid, and test sets successfully."
else
    echo "Error: Directory $data does not exist."
    exit 1
fi
