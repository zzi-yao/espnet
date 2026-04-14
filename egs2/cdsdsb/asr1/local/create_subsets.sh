#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
echo "First argument (\$1): $1"
if [ -d ${data} ];then
    cat $data/data_all/wav.scp > $data/all.scp
    awk 'NR % 10 == 0' $data/all.scp > $data/test.scp
    ./utils/filter_scp.pl --exclude $data/test.scp $data/all.scp > $data/remaining.scp
    awk 'NR % 10 == 0' $data/remaining.scp > $data/valid.scp
    ./utils/filter_scp.pl --exclude $data/valid.scp $data/remaining.scp > $data/train.scp
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
    ./utils/subset_data_dir.sh --utt-list $data/valid.scp $data/data_all $data/valid
    ./utils/subset_data_dir.sh --utt-list $data/test.scp $data/data_all $data/test
    # #parta部分划分train1和train2
    # awk 'BEGIN{split("13 04 14 42 05 12 15 20 22 07 23 29 34 43 09 38 25 26 10 02 01 08 40 33 37 39",E);for(i in E)easy[E[i]]=1}
    #     {spk=$2;if(spk in easy)print $1}' $data/train/utt2spk > $data/train1.utt
    # awk 'BEGIN{split("13 03 14 11 05 17 15 19 22 24 23 28 34 31 09 35 25 41 10 01 40 37",E);for(i in E)easy[E[i]]=1}
    #     {spk=$2;if(spk in easy)print $1}' $data/train/utt2spk > $data/train1.utt     #随机划分
    # #partb部分划分train1和train2
    awk 'BEGIN{split("01 12 20",E);for(i in E)easy[E[i]]=1}
        {spk=$2;if(spk in easy)print $1}' $data/train/utt2spk > $data/train1.utt
    ./utils/subset_data_dir.sh --utt-list $data/train1.utt $data/train $data/train1
    ./utils/subset_data_dir.sh --utt-list $data/train1.utt $data/train $data/train1
    ./utils/filter_scp.pl --exclude $data/train1.utt $data/train/utt2spk | awk '{print $1}' > $data/train2.utt
    ./utils/subset_data_dir.sh --utt-list $data/train2.utt $data/train $data/train2
    echo "Data split into train, valid, and test sets successfully."
else
    echo "Error: Directory $data does not exist."
    exit 1
fi
