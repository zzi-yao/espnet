# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

import argparse
import os
from pathlib import Path
import re
import shlex


def main():
    parser = argparse.ArgumentParser(description="Prepare metadata for one CDSD patient.")
    parser.add_argument("wav_scp")
    parser.add_argument("trans")
    parser.add_argument("utt2spk")
    parser.add_argument("--patient-id", default=os.environ.get("PATIENT_ID", "01"))
    parser.add_argument(
        "--text-dir",
        default="/home/q/Downloads/CDSD/CDSD-Interspeech/after_catting/1h/text",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{2}", args.patient_id):
        parser.error("patient ID must contain exactly two digits (e.g. 01 or 02)")
    text_path = Path(args.text_dir) / f"{args.patient_id}.txt"
    if not text_path.is_file():
        parser.error(f"transcript does not exist: {text_path}")

    utterances = set()
    mappings = []
    with open(args.wav_scp, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            uttid, command = line.split(maxsplit=1)
            fields = shlex.split(command)
            # Support both the recipe's ffmpeg commands and plain wav paths.
            wav_path = fields[fields.index("-i") + 1] if fields[0] == "ffmpeg" else fields[0]
            if Path(wav_path).parent.name == args.patient_id:
                utterances.add(uttid)
                mappings.append(f"{uttid}\t{args.patient_id}\n")
    if not utterances:
        parser.error(f"no speech found for patient {args.patient_id}")

    transcripts = []
    with text_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            entry = f"Audio-{args.patient_id}-{line.rstrip()}"
            if entry.split(maxsplit=1)[0] in utterances:
                transcripts.append(entry + "\n")
    if not transcripts:
        parser.error(f"no matching transcripts found in {text_path}")

    with open(args.utt2spk, "w", encoding="utf-8") as output:
        output.writelines(mappings)
    with open(args.trans, "w", encoding="utf-8") as output:
        output.writelines(transcripts)


if __name__ == "__main__":
    main()
