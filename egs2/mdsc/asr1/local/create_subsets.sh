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
    # awk 'BEGIN{split("01 12 20",E);for(i in E)easy[E[i]]=1}
    #     {spk=$2;if(spk in easy)print $1}' $data/train/utt2spk > $data/train1.utt
    # ./utils/subset_data_dir.sh --utt-list $data/train1.utt $data/train $data/train1
    # ./utils/subset_data_dir.sh --utt-list $data/train1.utt $data/train $data/train1
    # ./utils/filter_scp.pl --exclude $data/train1.utt $data/train/utt2spk | awk '{print $1}' > $data/train2.utt
    # ./utils/subset_data_dir.sh --utt-list $data/train2.utt $data/train $data/train2
    # echo "Data split into train, valid, and test sets successfully."
else
    echo "Error: Directory $data does not exist."
    exit 1
fi
