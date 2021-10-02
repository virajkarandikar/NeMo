#!/bin/bash

TMP_DIR=$3 # for temporary log file
mkdir -p ${TMP_DIR}
python duplex_text_normalization_test.py    data.test_ds.data_path=$2  decoder_pretrained_model=$1 data.test_ds.use_cache=True  data.test_ds.batch_size=128 lang=en mode=tn 2>&1 | tee ${TMP_DIR}/log_errors.txt
#declare -a cls=("CARDINAL" "DECIMAL" "ORDINAL" "DIGIT" "TIME" "DATE" "ADDRESS" "TELEPHONE" "MEASURE" "MONEY" "FRACTION")
declare -a cls=("MONEY")
for c in "${cls[@]}"
do
    echo "finding unrecoverable errors for ${c}"
    grep "^${c}\s" ${TMP_DIR}/log_errors.txt > ${TMP_DIR}/log_errors_${c}.txt
    python find_unrecoverable_errors.py ${TMP_DIR}/log_errors_${c}.txt 
done 

