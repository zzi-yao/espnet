#!/usr/bin/env python3
"""Build a reproducible augmented Kaldi directory without changing either source."""

import argparse
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import unicodedata


def read_mapping(path):
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"{path}:{number}: invalid or duplicate entry")
        result[fields[0]] = fields[1]
    if not result:
        raise ValueError(f"Empty file: {path}")
    return result


def load(directory):
    if (directory / "segments").exists():
        raise ValueError(f"Segmented audio is not supported: {directory}")
    files = {name: read_mapping(directory / name)
             for name in ("wav.scp", "text", "utt2spk")}
    if not files["wav.scp"].keys() == files["text"].keys() == files["utt2spk"].keys():
        raise ValueError(f"Utterance IDs do not match in {directory}")
    return files


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if not unicodedata.category(c).startswith(("P", "Z", "C")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", default="1", help="Fraction of generated utterances, 0–1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    try:
        ratio = Decimal(args.ratio)
        if not ratio.is_finite() or not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")
        original, generated = load(args.original), load(args.generated)
        if original["text"].keys() & generated["text"].keys():
            raise ValueError("Original and generated utterance IDs overlap")
        # Hash ordering gives reproducible, nested subsets as the ratio increases.
        ordered = sorted(generated["text"], key=lambda key: (
            hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest(), key))
        count = int((len(ordered) * ratio).to_integral_value(rounding=ROUND_FLOOR))
        if ratio > 0 and count == 0:
            raise ValueError("Ratio selects zero utterances; increase it or use ratio=0")
        selected = ordered[:count]
        held_ids, held_text = set(), set()
        for directory in args.held_out:
            texts = read_mapping(directory / "text")
            held_ids.update(texts)
            held_text.update(normalize(text) for text in texts.values())
        for key in selected:
            if key in held_ids or normalize(generated["text"][key]) in held_text:
                raise ValueError(f"Generated utterance overlaps held-out data: {key}")
            wav = generated["wav.scp"][key]
            if not Path(wav).is_absolute() or not Path(wav).is_file():
                raise ValueError(f"Generated audio must be an existing absolute file path: {wav}")
        contents = {}
        for name in original:
            merged = dict(original[name])
            merged.update((key, generated[name][key]) for key in selected)
            contents[name] = "".join(f"{key} {merged[key]}\n" for key in sorted(merged))
        speakers = {}
        for line in contents["utt2spk"].splitlines():
            key, speaker = line.split()
            speakers.setdefault(speaker, []).append(key)
        contents["spk2utt"] = "".join(
            f"{speaker} {' '.join(speakers[speaker])}\n" for speaker in sorted(speakers))
        contents["augmentation.json"] = json.dumps({
            "original": str(args.original.resolve()),
            "generated": str(args.generated.resolve()),
            "ratio": str(ratio.normalize()), "seed": args.seed,
            "original_count": len(original["text"]),
            "generated_available": len(ordered), "generated_selected": count,
            "selected_ids": selected,
        }, ensure_ascii=False, indent=2) + "\n"
        if args.output.exists():
            # Resume only if all inputs and selection still produce identical output.
            if not args.output.is_dir() or any(
                not (args.output / name).is_file()
                or (args.output / name).read_text(encoding="utf-8") != text
                for name, text in contents.items()
            ):
                raise ValueError(f"Output exists with different data; choose another output: {args.output}")
        else:
            args.output.mkdir(parents=True)
            for name, text in contents.items():
                (args.output / name).write_text(text, encoding="utf-8")
        print(f"{args.output}: {len(original['text'])} original + {count}/{len(ordered)} generated")
    except (OSError, ValueError, InvalidOperation) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
