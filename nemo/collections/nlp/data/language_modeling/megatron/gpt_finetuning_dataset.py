# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""
Prompt tuning dataset
Expects data to be in the format:
{"text": "example question1", "answer": "answer1"}
{"text": "example question2", "answer": "answer2"}
{"text": "example question3", "answer": "answer3"}

"""
import json

import torch
from tqdm import tqdm

from nemo.core import Dataset
from nemo.utils import logging

__all__ = ["GPTFineTuningDataset"]


class GPTFineTuningDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos_eos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.add_bos_eos = add_bos_eos
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.data = []

        assert min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert max_seq_length > 0, "Max sequence length should be greater than 0"

        dataset_file = open(dataset_path, 'r', encoding='utf-8')

        logging.info("Loading and tokenizing dataset ... ")

        skipped = 0
        for json_line in tqdm(dataset_file):
            doc = json.loads(json_line)
            sent = str(doc["text"])

            sent_ids = tokenizer.text_to_ids(sent)

            if self.add_bos_eos:
                sent_ids = [tokenizer.bos_id] + sent_ids + [tokenizer.eos_id]

            # Need to leave space for prompt tokens in sequence
            if self.min_seq_length <= len(sent_ids) <= self.max_seq_length:
                self.data.append(sent_ids)

            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too long or too short')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        """Build masks and position id for left to right model with prompt tuning."""

        input_ids = batch

        # Get max sequence length of batch
        batch_size = len(input_ids)
        batch_max = max(len(ids) for ids in input_ids)

        # Pad tokens in batch to max batch length while building loss mask
        loss_masks = []
        for idx, ids in enumerate(input_ids):
            text_length = len(ids)

            # Loss mask padding only
            text_loss_mask = [1.0] * text_length
            padding_length = batch_max - text_length

            # Pad loss mask and text tokens
            ids.extend([self.tokenizer.eos_id] * padding_length)
            text_loss_mask.extend([0.0] * padding_length)
            loss_masks.append(torch.tensor(text_loss_mask, dtype=torch.float))

        tokens_ = torch.tensor(input_ids, dtype=torch.long)
        tokens = tokens_[:,:-1].contiguous()
        labels = tokens_[:, 1:].contiguous()

        # Loss mask should match the labels
        loss_mask = torch.stack(loss_masks)
        loss_mask = loss_mask[:, 1:]

        batch_max = batch_max - 1

        # Position ids for text
        text_position_ids = torch.arange(batch_max, dtype=torch.long,)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(tokens).clone()

        # Attention mask (lower triangular) starting with prompt tokens
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        return tokens, labels, attention_mask, loss_mask, text_position_ids
