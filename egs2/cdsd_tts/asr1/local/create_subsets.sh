#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

set -euo pipefail

# Usage: local/create_subsets.sh DATA [PATIENT_ID]
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 DATA [PATIENT_ID]" >&2
    exit 1
fi
patient_id=${2:-${PATIENT_ID:-01}}
if [[ ! "$patient_id" =~ ^[0-9]{2}$ ]]; then
    echo "Error: Patient ID must contain exactly two digits (e.g. 01 or 02)." >&2
    exit 1
fi

data=$1     # data transformed into kaldi format
echo "First argument (\$1): $1"
if [ -d "$data" ];then
    # Select the patient using the speaker mapping.
    awk -v patient="$patient_id" 'FILENAME == ARGV[1] {if (("id:" $2) == ("id:" patient)) patient_utts[$1] = 1; next}
         $1 in patient_utts' "$data/data_all/utt2spk" "$data/data_all/wav.scp" > "$data/all.scp"
    if [ ! -s "$data/all.scp" ]; then
        echo "Error: No speech found for patient $patient_id in $data/data_all." >&2
        exit 1
    fi
    awk 'NR % 10 == 0' "$data/all.scp" > "$data/test.scp"
    ./utils/filter_scp.pl --exclude "$data/test.scp" "$data/all.scp" > "$data/remaining.scp"
    awk 'NR % 9 == 0' "$data/remaining.scp" > "$data/valid.scp"
    ./utils/filter_scp.pl --exclude "$data/valid.scp" "$data/remaining.scp" > "$data/train.scp"
    ./utils/subset_data_dir.sh --utt-list "$data/train.scp" "$data/data_all" "$data/train${patient_id}"
    ./utils/subset_data_dir.sh --utt-list "$data/valid.scp" "$data/data_all" "$data/valid"
    ./utils/subset_data_dir.sh --utt-list "$data/test.scp" "$data/data_all" "$data/test"
else
    echo "Error: Directory $data does not exist."
    exit 1
fi

