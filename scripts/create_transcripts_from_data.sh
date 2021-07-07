#!/bin/sh
echo "This will create transcripts for you, Kindly change the datapaths accordingly"

INPUT_DIR_1="Datasets/train-clean-100/LibriSpeech/train-clean-100"
OUTPUT_FILE="Datasets/train-clean-100/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/dev-clean/LibriSpeech/dev-clean"
OUTPUT_FILE="Datasets/dev-clean/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/dev-other/LibriSpeech/dev-other"
OUTPUT_FILE="Datasets/dev-other/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/test-clean/LibriSpeech/test-clean"
OUTPUT_FILE="Datasets/test-clean/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/train-clean-360/LibriSpeech/train-clean-360"
OUTPUT_FILE="Datasets/train-clean-360/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/train-other-500/LibriSpeech/train-other-500"
OUTPUT_FILE="Datasets/train-other-500/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE

INPUT_DIR_1="Datasets/test-other/LibriSpeech/test-other"
OUTPUT_FILE="Datasets/test-other/LibriSpeech/transcripts.tsv"
python3 create_librispeech_trans.py -d $INPUT_DIR_1 $OUTPUT_FILE
