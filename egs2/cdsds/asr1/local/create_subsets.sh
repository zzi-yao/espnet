#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
echo "First argument (\$1): $1"
# Check if the data directory exists
if [ -d ${data} ];then
    # Generate a list of all wav files
    cat $data/data_all/wav.scp > $data/all.scp

    # Split the data into training, validation, and test sets
    # Assume 80% for training, 10% for validation, and 10% for test
    total_lines=$(wc -l < $data/all.scp)
    train_lines=$((total_lines * 80 / 100))
    valid_lines=$((total_lines * 10 / 100))
    test_lines=$((total_lines - train_lines - valid_lines))

    # Extract the first part for training
    head -n $train_lines $data/all.scp > $data/train.scp
    # Extract the next part for validation
    sed -n "$((train_lines+1)),$((train_lines+valid_lines))p" $data/all.scp > $data/valid.scp
    # Extract the remaining part for test
    tail -n $test_lines $data/all.scp > $data/test.scp

    # Create the corresponding data directories
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/train
    ./utils/subset_data_dir.sh --utt-list $data/valid.scp $data/data_all $data/valid
    ./utils/subset_data_dir.sh --utt-list $data/test.scp $data/data_all $data/test

    echo "Data split into train, valid, and test sets successfully."
else
    echo "Error: Directory $data does not exist."
    exit 1
fi
