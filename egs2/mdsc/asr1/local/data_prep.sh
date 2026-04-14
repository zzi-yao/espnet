#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

raw_data=$1     # raw data with metadata, txt and wav
data=$2         # data transformed into kaldi format

# generate kaldi format data for all
if [ -d ${raw_data} ];then
    echo "Generating kaldi format data."
    mkdir -p $data/data_all
    find $raw_data -type f -name "*.wav" > $data/data_all/wavpath
    awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}' $data/data_all/wavpath | sed 's:\.wav::g' > $data/data_all/uttlist
    #paste $data/data_all/uttlist $data/data_all/wavpath > $data/data_all/wav.scp
    paste $data/data_all/uttlist $data/data_all/wavpath | awk '{printf("%s ffmpeg -i %s -f wav -ar 16000 -ac 1 -loglevel quiet - |\n", $1, $2)}' > $data/data_all/wav.scp 
    python local/preprocess.py $data/data_all/wav.scp $data/data_all/trans $data/data_all/utt2spk #$data/data_all/text
    # python local/preprocess.py $data/data_all/uttlist $data/data_all/trans $data/data_all/utt2spk $data/data_all/wavpath #$data/data_all/text
    ./utils/utt2spk_to_spk2utt.pl $data/data_all/utt2spk > $data/data_all/spk2utt
    #paste $data/data_all/uttlist $data/data_all/wavpath | awk '{printf("%s ffmpeg -i %s -f wav -ar 16000 -ac 1 -loglevel quiet - |\n", $1, $2)}' > $data/data_all/wav.scp
fi

cp $data/data_all/trans $data/data_all/text
echo "local/data_prep.sh succeeded"
exit 0;