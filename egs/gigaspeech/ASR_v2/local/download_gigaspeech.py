#!/usr/bin/env python3

import soundfile as sf
import re

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset, Audio


def has_no_oov(text):
    return re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>").search(text) is None


def normalize_text(text):
    return re.compile(r"\s\s+").sub(" ", re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>").sub("", text))


L_COUNT = 2266371

CONFIG2COUNT = {
    "xs": int(L_COUNT * 0.004),
    "s": int(L_COUNT * 0.1),
    "m": int(L_COUNT * 0.4),
    "l": L_COUNT,
    "xl": int(L_COUNT * 4),
    "dev": 5715,
    "test": 19930,
}

CONFIG2SPLIT = {
    "xs": "train",
    "s": "train",
    "m": "train",
    "l": "train",
    "xl": "train",
    "dev": "validation",
    "test": "test",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="l")
    parser.add_argument("--sampling_rate", type=int, default=16_000)
    parser.add_argument("--output_dir", type=str, default="./download")
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def main(args):
    path = Path(args.output_dir) / "GigaSpeech"
    splits_dir = path / CONFIG2SPLIT[args.config]
    splits_dir.mkdir(exist_ok=True)

    # if splits_dir.exists():
    #     print(f"GigaSpeech {args.config} already exists - skipping")
    #     return

    ds = load_dataset(
        "speechcolab/gigaspeech",
        args.config,
        split=CONFIG2SPLIT[args.config],
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    def process_datum(datum):
        audio = datum["audio"]
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        id = datum["segment_id"]
        text = datum["text"]

        if not has_no_oov(text):
            return

        # split training audio into chunks to avoid overly populated directories
        if CONFIG2SPLIT[args.config] == "train":
            chunk_name = audio["path"].split("/")[0]
            chunk_dir = splits_dir / chunk_name
            chunk_dir.mkdir(exist_ok=True)

            audio_path = (chunk_dir / id).with_suffix(".wav")
            text_path = (chunk_dir / id).with_suffix(".txt")
        else:
            audio_path = (splits_dir / id).with_suffix(".wav")
            text_path = (splits_dir / id).with_suffix(".txt")

        if audio_path.exists() or text_path.exists():
            return

        sf.write(audio_path, audio_array, sr)
        with open(text_path, "w") as f:
            f.write(normalize_text(text))

    for datum in tqdm(iter(ds), total=CONFIG2COUNT[args.config]):
        process_datum(datum)


if __name__ == "__main__":
    main(parse_args())
