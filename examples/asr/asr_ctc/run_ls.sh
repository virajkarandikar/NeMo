#!/bin/bash
python speech_to_text_ctc_bpe_ts.py \
model.tokenizer.type=bpe \
model.tokenizer.dir=/home/yangzhang/code/ts_asr/ami_tokenizer/tokenizer_spe_unigram_v128/ \
model.train_ds.manifest_filepath=/mnt/data/LibriSpeech/train_clean_100.json \
model.train_ds.augmentor.rir_noise_aug.rir_manifest_path=/mnt/data/rir_noises/noises_20s_tarred/tarred_audio_manifest.json \
model.train_ds.augmentor.rir_noise_aug.rir_tar_filepaths=/mnt/data/rir_noises/noises_20s_tarred/audio_{0..255}.tar \
model.train_ds.augmentor.rir_noise_aug.noise_manifest_paths=[/mnt/data/LibriSpeech/train_clean_100.json] \
model.train_ds.augmentor.rir_noise_aug.bg_noise_manifest_paths=[/mnt/data/rir_noises/rir_tarred/tarred/tarred_audio_manifest.json] \
model.train_ds.augmentor.rir_noise_aug.bg_noise_tar_filepaths=[/mnt/data/rir_noises/rir_tarred/tarred/audio_{0..127}.tar] \
trainer.max_epochs=1 \
model.validation_ds.manifest_filepath=/mnt/data/LibriSpeech/dev_clean.json \
model.train_ds.max_duration=20 \
trainer.gpus=1 model.tokenizer.type=bpe \
trainer.num_sanity_val_steps=2 \
+nemo_checkpoint_path=/home/yangzhang/code/ts_asr/stt_en_conformer_ctc_large_v1.0.0/stt_en_conformer_ctc_large.nemo
