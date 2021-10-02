#!/bin/bash

# default values for optional arguments
MIN_SCORE=-100
CUT_PREFIX=18
SCRIPTS_DIR="scripts"
OFFSET=-180
LANGUAGE='eng' # 'eng', 'ru', 'other'
MIN_SEGMENT_LEN=0
MAX_SEGMENT_LEN=1
ADDITIONAL_SPLIT_SYMBOLS=" "
AUDIO_FORMAT='.mp3'
USE_NEMO_NORMALIZATION='False'

DATA_DIR="/home/ebakhturina/data/segmentation/test/data"
MODEL_NAME_OR_PATH=stt_en_citrinet_256 # "QuartzNet15x5Base-En" #
OUTPUT_DIR="/home/ebakhturina/data/segmentation/test/out_${MODEL_NAME_OR_PATH}"

#rm -rf ${OUTPUT_DIR}

for ARG in "$@"
do
    key=$(echo $ARG | cut -f1 -d=)
    value=$(echo $ARG | cut -f2 -d=)

    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare $v="${value}"
    fi
done

echo "MODEL_NAME_OR_PATH = $MODEL_NAME_OR_PATH"
echo "DATA_DIR = $DATA_DIR"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "MIN_SCORE = $MIN_SCORE"
echo "CUT_PREFIX = $CUT_PREFIX"
echo "SCRIPTS_DIR = $SCRIPTS_DIR"
echo "OFFSET = $OFFSET"
echo "LANGUAGE = $LANGUAGE"
echo "MIN_SEGMENT_LEN = $MIN_SEGMENT_LEN"
echo "MAX_SEGMENT_LEN = $MAX_SEGMENT_LEN"
echo "ADDITIONAL_SPLIT_SYMBOLS = $ADDITIONAL_SPLIT_SYMBOLS"
echo "AUDIO_FORMAT = $AUDIO_FORMAT"
echo "USE_NEMO_NORMALIZATION = $USE_NEMO_NORMALIZATION"

#if [[ -z $MODEL_NAME_OR_PATH ]] || [[ -z $DATA_DIR ]] || [[ -z $OUTPUT_DIR ]]; then
#  echo "Usage: $(basename "$0")
#  --MODEL_NAME_OR_PATH=[model_name_or_path]
#  --DATA_DIR=[data_dir]
#  --OUTPUT_DIR=[output_dir]
#  --LANGUAGE=[language (Optional)]
#  --OFFSET=[offset value (Optional)]
#  --CUT_PREFIX=[cut prefix in sec (Optional)]
#  --SCRIPTS_DIR=[scripts_dir_path (Optional)]
#  --MIN_SEGMENT_LEN=[min number of characters of the text segment for alignment (Optional)]
#  --MAX_SEGMENT_LEN=[max number of characters of the text segment for alignment (Optional)]
#  --ADDITIONAL_SPLIT_SYMBOLS=[Additional symbols to use for
#    sentence split if eos sentence split resulted in sequence longer than --max_length.
#    Use '|' as a separator between symbols, for example: ';|:' (Optional)]
#  --AUDIO_FORMAT=[choose from ['.mp3', '.wav'], input audio files format
#  --USE_NEMO_NORMALIZATION Set to 'True' to use NeMo Normalization tool to convert
#    numbers from written to spoken format. By default num2words package will be used. (Optional)"
#  exit 1
#fi

NEMO_NORMALIZATION=""
    if [[ ${USE_NEMO_NORMALIZATION,,} == "true" ]]; then
      NEMO_NORMALIZATION="--use_nemo_normalization "
    fi

## STEP #1
## Prepare text and audio data for segmentation
#python $SCRIPTS_DIR/prepare_data.py \
#--in_text=$DATA_DIR/text \
#--audio_dir=$DATA_DIR/audio \
#--audio_format=$AUDIO_FORMAT \
#--output_dir=$OUTPUT_DIR/processed/ \
#--language=$LANGUAGE \
#--cut_prefix=$CUT_PREFIX \
#--model=$MODEL_NAME_OR_PATH \
#--min_length=$MIN_SEGMENT_LEN \
#--max_length=$MAX_SEGMENT_LEN \
#--additional_split_symbols=$ADDITIONAL_SPLIT_SYMBOLS $NEMO_NORMALIZATION || exit

# STEP #2
# Run CTC-segmenatation
# one might want to perform alignment with various window sizes
# note if the alignment with the initial window size isn't found, the window size will be double to re-attempt
# alignment
for WINDOW in 8000 #12000
do
  python $SCRIPTS_DIR/run_ctc_segmentation.py \
  --output_dir=$OUTPUT_DIR \
  --data=$OUTPUT_DIR/processed/ \
  --model=$MODEL_NAME_OR_PATH  \
  --window_len $WINDOW \
  --debug \
  --no_parallel || exit
done

# STEP #3 (Optional)
# Verify aligned segments only if multiple WINDOWs used in the Step #2)
python $SCRIPTS_DIR/verify_segments.py \
--base_dir=$OUTPUT_DIR  || exit

# STEP #4
# Cut the original audio files based on the alignments
# (use --alignment=$OUTPUT_DIR/segments if only 1 WINDOW size was used in the Step #2)
# Three manifests and corresponding clips folders will be created:
#   - high scored clips
#   - low scored clips
#   - deleted segments
python $SCRIPTS_DIR/cut_audio.py \
--output_dir=$OUTPUT_DIR \
--model=$MODEL_NAME_OR_PATH \
--alignment=$OUTPUT_DIR/verified_segments \
--threshold=$MIN_SCORE \
--offset=$OFFSET || exit

# STEP #5 (Optional)
# If multiple audio files were segmented in the step #2, this step will aggregate manifests with high scored segments
# for all audio files into all_manifest.json
# Also a separate manifest with samples from across all high scored segments will be credated if --num_samples > 0
# --num_samples samples will be taken from the beginning, end and the middle of the each audio file high score manifest
# and will be stored at sample_manifest.json
python $SCRIPTS_DIR/process_manifests.py \
--output_dir=$OUTPUT_DIR \
--manifests_dir=$OUTPUT_DIR/manifests/ \
--num_samples 0

python /home/ebakhturina/NeMo/examples/asr/transcribe_speech.py \
pretrained_name=$MODEL_NAME_OR_PATH \
dataset_manifest=/home/ebakhturina/data/segmentation/test/out_${MODEL_NAME_OR_PATH}/all_manifest.json \
output_filename=/home/ebakhturina/data/segmentation/test/out_${MODEL_NAME_OR_PATH}/manifests/transcribed_${MODEL_NAME_OR_PATH}.json


python /home/ebakhturina/NeMo/tools/speech_data_explorer/data_explorer.py \
--port 8055 \
/home/ebakhturina/data/segmentation/test/out_${MODEL_NAME_OR_PATH}/manifests/transcribed_${MODEL_NAME_OR_PATH}.json

