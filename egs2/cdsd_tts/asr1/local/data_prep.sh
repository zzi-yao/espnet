#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

. ./path.sh || exit 1
. ./cmd.sh || exit 1
set -euo pipefail

# Usage: local/data_prep.sh RAW_DATA DATA [PATIENT_ID]
# PATIENT_ID environment variable also works for the whole pipeline; default: 01.
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 RAW_DATA DATA [PATIENT_ID]" >&2
    exit 1
fi
raw_data=$1
data=$2
patient_id=${3:-${PATIENT_ID:-01}}
if [[ ! "$patient_id" =~ ^[0-9]{2}$ ]]; then
    echo "Error: Patient ID must contain exactly two digits (e.g. 01 or 02)." >&2
    exit 1
fi
if [ ! -d "$raw_data" ]; then
    echo "Error: Directory $raw_data does not exist." >&2
    exit 1
fi
if [ ! -f "$raw_data/text/$patient_id.txt" ]; then
    echo "Error: Transcript $raw_data/text/$patient_id.txt does not exist." >&2
    exit 1
fi

mkdir -p "$data/data_all"
# A patient's wav files are directly inside the directory named with their ID.
find "$raw_data" -type f -name '*.wav' -path "*/$patient_id/*.wav" |
    awk -F/ -v patient="$patient_id" '("id:" $(NF-1)) == ("id:" patient)' |
    LC_ALL=C sort > "$data/data_all/wavpath"
if [ ! -s "$data/data_all/wavpath" ]; then
    echo "Error: No speech found for patient $patient_id in $raw_data." >&2
    exit 1
fi
awk -F/ '{print $(NF-2)"-"$(NF-1)"-"$NF}' "$data/data_all/wavpath" |
    sed 's:\.wav$::' > "$data/data_all/uttlist"
paste "$data/data_all/uttlist" "$data/data_all/wavpath" |
    awk -F '\t' '{printf("%s ffmpeg -i \"%s\" -f wav -ar 16000 -ac 1 -loglevel quiet - |\n", $1, $2)}' > "$data/data_all/wav.scp"
python3 local/preprocess.py "$data/data_all/wav.scp" "$data/data_all/trans" \
    "$data/data_all/utt2spk" --patient-id "$patient_id" --text-dir "$raw_data/text"
./utils/utt2spk_to_spk2utt.pl "$data/data_all/utt2spk" > "$data/data_all/spk2utt"
cp "$data/data_all/trans" "$data/data_all/text"
echo "local/data_prep.sh succeeded (patient $patient_id)"
