# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from pyexpat import features
from typing import Dict, List, Optional
from nemo.collections.common.parts.preprocessing import manifest, parsers
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np

import torch

from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset
from nemo.core.neural_types import AcousticEncodedRepresentation, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class FeatureDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to feature files, sequences of labels. 
    Each new line is a different sample. Example below:
    and their target labels. JSON files should be of the following format:
        {"feature_filepath": "/path/to/feature_0.p", "seq_label": speakerA speakerB SpeakerA ....} \
        ...
        {"feature_filepath": "/path/to/feature_n.p", "seq_label": target_seq_label_n} 
    target_seq_label_n is the string of sequence of speaker label, separated by space.

    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (Optional[list]): Dataset parameter. List of unique labels collected from all samples.
        feature_loader : Dataset parameter. Feature loader to load (external) feature.       
    """

    def __init__(
        self, *, manifest_filepath: str, embedding_file_path: Optional[str]=None
    ):
        super().__init__()

        def parse_item(line: str, manifest_file: str) -> Dict[str, Any]:
            item = json.loads(line)

            
            audio_path = item.pop('audio_filepath')
            speaker_id = item.pop('label')
            session_id = item.pop('session_id')
            item.pop('session_start_time')
            item.pop('session_end_time')
            item.pop('text')
            item.pop('min_speech_duration')
            item.pop('max_speech_interruption')
            item['feature_id'] = f"processed@individual@{speaker_id}" # TODO
            return item

        self.features = []
        self.embeddings = np.load(embedding_file_path, allow_pickle=True)
        for  item in manifest.item_iter(manifest_filepath, parse_func=parse_item):
            self.features.append(item['feature_id'])


        for idx in range(len(self.features[:5])):
            logging.debug("{}-th feature id is {}".format(idx, self.features[idx]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature_id = self.features[index]

        embedding = self.embeddings[feature_id][0]
        t = torch.tensor(embedding).float()
        return t


class ZipDataset(Dataset):

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """
        output_types = {}
        for k, v in self.asr_dataset.output_types.items():
            output_types[k]=v
        output_types.pop("sample_id")
        output_types['embeddings']= NeuralType(('B', 'T'), AcousticEncodedRepresentation())
        return output_types

    def __init__(self, asr_dataset, feature_dataset):
        super().__init__()
        self.asr_dataset  = asr_dataset
        self.feature_dataset = feature_dataset
    
    def __len__(self):
        return len(self.asr_dataset)

    def __getitem__(self, index):
        return self.asr_dataset[index], self.feature_dataset[index]

    def _collate_fn(self, batch):
        asr_batch, feature_batch = list(zip(*batch))
        asr_collate = self.asr_dataset._collate_fn(asr_batch)
        feature_collate = self.feature_dataset._collate_fn(feature_batch)
        return *asr_collate, feature_collate