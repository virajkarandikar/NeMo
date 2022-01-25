#!/bin/bash
python speech_to_text_ctc_bpe_ts.py \
model.tokenizer.type=bpe \
model.tokenizer.dir=/home/yangzhang/code/ts_asr/tokenizer_all_v128/tokenizer_spe_unigram_v128/ \
model.train_ds.manifest_filepath=/mnt/data/ami/processed/mixed_utterance/manifest_train.json \
trainer.max_epochs=1 \
model.validation_ds.manifest_filepath=/mnt/data/ami/processed/mixed_utterance/manifest_dev.json \
model.train_ds.embedding_filepath=//mnt/data/ami/processed/individual/embeddings/manifest_train_embeddings.pkl \
model.train_ds.max_duration=28 \
+model.validation_ds.embedding_filepath=//mnt/data/ami/processed/individual/embeddings/manifest_dev_embeddings.pkl \
trainer.gpus=1 model.tokenizer.type=bpe \
trainer.num_sanity_val_steps=2 \
+model.test_ds.embedding_filepath=/mnt/data/ami/processed/individual/embeddings/manifest_dev_embeddings.pkl \
model.test_ds.manifest_filepath=/mnt/data/ami/processed/mixed_utterance/manifest_dev.json \
+nemo_checkpoint_path=/home/yangzhang/code/ts_asr/stt_en_conformer_ctc_large_v1.0.0/stt_en_conformer_ctc_large.nemo
