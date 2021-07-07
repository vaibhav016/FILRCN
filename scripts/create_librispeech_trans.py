# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import argparse
import librosa
from tqdm.auto import tqdm
import unicodedata

from tensorflow_asr.utils.file_util import preprocess_paths


def fetch_transcripts(text_files, sanity):
    transcripts = []
    text_files_read = 0
    for text_file in tqdm(text_files, desc="[Loading]"):
        current_dir = os.path.dirname(text_file)
        text_files_read += 1
        with open(text_file, "r", encoding="utf-8") as txt:
            lines = txt.read().splitlines()
        for line in lines:
            line = line.split(" ", maxsplit=1)
            audio_file = os.path.join(current_dir, line[0] + ".flac")
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y, sr)
            text = unicodedata.normalize("NFC", line[1].lower())
            transcripts.append(f"{audio_file}\t{duration}\t{text}\n")
            if sanity:
                break
        if sanity and text_files_read > 10:
            break
    return transcripts


parser = argparse.ArgumentParser(prog="Setup LibriSpeech Transcripts")

parser.add_argument("--dir", "-d", type=str, default=None, help="Directory of dataset")

parser.add_argument("--sanity", "-s", default=False, action="store_true", help="wheather just to check for sanity")

parser.add_argument("output", type=str, default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output

args.dir = preprocess_paths(args.dir, isdir=True)
args.output = preprocess_paths(args.output)

text_files = glob.glob(os.path.join(args.dir, "**", "*.txt"), recursive=True)

transcripts = fetch_transcripts(text_files, args.sanity)

with open(args.output, "w", encoding="utf-8") as out:
    out.write("PATH\tDURATION\tTRANSCRIPT\n")
    for line in tqdm(transcripts, desc="[Writing]"):
        out.write(line)
