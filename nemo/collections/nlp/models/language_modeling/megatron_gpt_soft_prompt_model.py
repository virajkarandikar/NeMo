# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from os import path
from typing import Dict, List

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

# from nemo.collections.nlp.data.glue_benchmark.gpt_ptune_dataset import GPTPTuneDataset, GPTPTuneInferenceDataset
# from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import (
#     make_attention_mask_3d,
#     make_history_mask_3d,
# )
from nemo.collections.nlp.data.language_modeling.megatron import GPTSoftPromptDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common import (
    PromptEncoder,
    PromptTable
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from apex.transformer import tensor_parallel
    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTPSoftPromptModel']

class MegatronGPTPSoftPromptModel(MegatronBaseModel):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron GPT model. 

    Prompt Tuning initalizes soft prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained GPT model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in 
    initalization are specified by the user in the config file. The model can 
    be prompt-tuned for multiple tasks at once. Soft prompts are stored in a 
    prompt table and can be added or deleted without disrupting soft prompts 
    for other tasks. 

    P-tuning initializes an LSTM encoder model that generates soft prompt
    embeddings for every task. Each task shares the same encoder. After ptuning
    is compelete, the learned soft prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a 
    new soft prompt via p-tuning, they do not need to retrain on all previous 
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)

        self.cfg = cfg

        # Load pretrained GPT model and tokenizer
        self.model = MegatronGPTModel.restore_from(
            self.register_artifact('language_model_path', cfg.get('language_model_path', None)),
            trainer=trainer,
        )

        # Freeze all GPT model weights for prompt-tuning/p-tuning
        if not cfg.lm_finetune:
            self.model.freeze()

        self.tokenizer = self.model.tokenizer
        self.float_type = self.model.model.language_model.encoder.layers[0].dtype
        self.hidden_size = self.model.cfg.hidden_size
        self.word_embeddings = self.model.model.language_model.embedding.word_embeddings
        self.total_soft_tokens = self.cfg.total_soft_tokens
        
        # Prompt table stores all task embeddings, p-tuning soft prompts get added to the table after training
        self.prompt_table = PromptTable(
            existing_tasks=self.cfg.get('existing_tasks', None),
            total_soft_tokens=self.total_soft_tokens,
            hidden_size=self.hidden_size,
        )

        # Load templates for assiging soft prompt token positions
        self.load_task_templates(self.cfg.task_templates) 
        self.soft_prompt_style = cfg.soft_prompt_style.lower()

        # Prompt tuning stores soft prompts in the prompt table and tunes their weight directly
        if self.soft_prompt_style == 'prompt-tuning':
            self.soft_token_source = 'prompt-table'
            
        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        elif self.soft_prompt_style == 'p-tuning':
            self.soft_token_source = 'prompt-encoder'
            self.prompt_encoder = PromptEncoder(
                total_soft_tokens=self.total_soft_tokens,
                hidden_size=self.hidden_size,
                lstm_dropout=cfg.p_tuning.dropout,
                num_layers=cfg.p_tuning.num_layers,
            )
        else:
            raise ValueError(
                f"\nSoft prompt style '{cfg.soft_prompt_type}' not recognized, please use one of 'prompt-tuning' or 'p-tuning'" )

        self._reduced_loss_buffer = []

        # Setup special tokens 
        self.pseudo_token = cfg.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.token_to_id(self.pseudo_token)
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

    def init_new_prompts(self):
        """
        Initialize new soft prompts to be tuned using prompt tuning 
        """
        for idx, taskname in enumerate(self.cfg.new_tasks):
            init_method = self.cfg.prompt_tuning.new_prompt_init_methods[idx].lower()

            if init_method == "text":
                init_text = self.cfg.prompt_tuning.new_prompt_init_text[idx]
                init_text_ids = self.tokenizer.text_to_ids(init_text)
                self.prompt_table.init_prompt_from_text(taskname, init_text_ids, self.word_embeddings)

            elif init_method == 'random':
                self.prompt_table.init_prompt_from_random(taskname)

            else:
                raise AttributeError(
                    f'\nSoft prompt init method {init_method} is not recognized\
                                        please use one of text or random'
                )

        # Add new tags to existing tag list for loading during inference later
        with open_dict(self.cfg):
            self.cfg.existing_tasks = self.cfg.existing_tasks + self.cfg.new_tasks
               

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        self.new_tasknames = []
        task_id_num_to_name = {}
        task_id_num = 0

        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_token_splits": task.prompt_token_splits,
                "task_id_num": task_id_num
            }
            
            task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Make sure tasknames and task id nums line up correctly in prompt table
        self.prompt_table.task_id_num_to_name = task_id_num_to_name

    def add_ptuned_prompts_to_prompt_table(self):
        """
        Adds all newly p-tuned soft prompts to the prompt table 
        for inference. p-tuned soft prompts WILL NOT be further
        tuned once added to the prompt table.
        """
        for taskname in self.new_tasknames:
            tokenized_taskname = self.tokenizer.text_to_ids(taskname)
            taskname_embeddings = self.word_embeddings(torch.tensor(tokenized_taskname))
            soft_prompt_embeddings = self.prompt_encoder(taskname_embeddings)
            task_id_num = self.prompt_template[taskname]["task_id_num"]
            self.prompt_table.add_prompt_from_p_tuning_encoder(taskname, task_id_num, soft_prompt_embeddings)

    def embed_input(self, input_ids: Tensor, taskname_ids: Tensor):
        """
        Replaces the virtual tokens in the input_ids with embeddings 
        calculated from either the 'prompt_table' or 'prompt_encoder'. 
        The virtual token placeholders have the token_id 
        `self.pseudo_token_id`.

        params:
            input_ids: the input token ids
            taskname_ids: the NLP task tag token ids
        returns:
            the token embedding for the LM model.
        """
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_ids[(input_ids == self.pseudo_token_id)] = self.pad_token_id
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()

        # Get virtual token embeddings from the prompt table or prompt encoder
        if self.soft_token_source == 'prompt-table':
            virtual_token_embeddings = [self.prompt_table(task_id_num) for task_id_num in taskname_ids]
            virtual_token_embeddings = torch.stack(virtual_token_embeddings)

        elif self.soft_token_source == 'prompt-encoder':
            taskname_embeddings = self.word_embeddings(taskname_ids)
            virtual_token_embeddings = self.prompt_encoder(taskname_embeddings=taskname_embeddings)

        # Find the indicies where virtual tokens should be inserted
        virtual_token_locations = input_ids == self.pseudo_token_id

        # Create index template specifying where virtual token embeddings should be placed
        batch_size, _, embedding_size = discrete_token_embeds.shape
        virtual_token_index = virtual_token_locations.nonzero().reshape((batch_size, -1, 2))[:, :, 1][:, :, None]
        virtual_token_index = virtual_token_index.expand(batch_size, self.total_soft_tokens, embedding_size)

        # Insert virtual token embeddings where they belong amoung the discrete token embeddings
        discrete_token_embeds.scatter_(1, virtual_token_index, virtual_token_embeddings)
        input_embeds = discrete_token_embeds

        return input_embeds

    def soft_prompt_forward(self, input_ids, labels, attention_mask, position_ids, taskname_ids):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        input_embeds = self.embed_input(input_ids, taskname_ids)
        position_embeddings = self.model.model.language_model.embedding.position_embeddings(position_ids)
        encoder_input = input_embeds + position_embeddings

        # Call forward on GPT model with preprocessed embeddings
        if self.float_type == torch.float32:
            output = self.model.model(
                input_ids=None, 
                position_ids=None, 
                encoder_input=encoder_input, 
                attention_mask=attention_mask, 
                labels=labels,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                output = self.model.model(
                input_ids=None, 
                position_ids=None, 
                encoder_input=encoder_input, 
                attention_mask=attention_mask, 
                labels=labels,
            )

        return output

    def training_step(self, batch, batch_idx):
        input_ids, labels, loss_mask, attention_mask, position_ids, taskname_ids = batch
        output = self.soft_prompt_forward(input_ids, labels, attention_mask, position_ids, taskname_ids)
        output_tensor, encoder_hidden_states = output
        loss = self.model.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)

        # Reduced loss for logging.
        reduced_loss = average_losses_across_data_parallel_group([loss])

        # Cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, loss_mask, attention_mask, position_ids, taskname_ids = batch
        
        with torch.no_grad():
            output = self.soft_prompt_forward(input_ids, labels, attention_mask, position_ids, taskname_ids)
            output_tensor, encoder_hidden_states = output
            loss = self.model.loss_func(loss_mask, output_tensor)
            self.log('validation_loss', loss)

            return loss

    def validation_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def setup(self, stage=None):
        if stage == 'predict':
            return
        
        # New soft prompt init needs to happen before building datasets
        if self.soft_prompt_style == 'prompt-tuning':
            self.init_new_prompts()

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()
        self.freeze_existing_soft_prompt_params()

    def freeze_existing_soft_prompt_params(self):
        """Freeze params of existing soft prompts that should not be tuned more
        """
        # Only want new prompt tags to be tunable, leave existing prompt tags alone
        for taskname in self.prompt_table.prompt_table.keys():
            if taskname in set(self.cfg.new_tasks):
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = True
            else:
                for params in self.prompt_table.prompt_table[taskname].parameters():
                    params.requires_grad = False

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_soft_prompt_dataset(
                dataset_path=self.cfg.data.train_ds,
                batch_size=self.cfg.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_soft_prompt_dataset(
                dataset_path=self.cfg.data.validation_ds,
                batch_size=self.cfg.batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_soft_prompt_dataset(
                dataset_path=self.cfg.data.test_ds,
                batch_size=self.cfg.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def build_soft_prompt_dataset(self, dataset_path, batch_size, drop_last, shuffle, num_workers, pin_memory):
        dataset = GPTSoftPromptDataset(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            soft_token_source=self.soft_token_source,
            task_templates=self.task_templates,
            total_soft_tokens=self.total_soft_tokens,
            pseudo_token=self.pseudo_token,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True)
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.cfg.batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    @classmethod
    def list_available_models(cls):
        pass