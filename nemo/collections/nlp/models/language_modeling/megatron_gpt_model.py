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

import re
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.distributed import rank_zero_only

import torch
import torch.nn.functional as F
from apex.transformer import parallel_state, tensor_parallel
from apex.transformer.pipeline_parallel.schedules.common import build_model, _get_params_for_weight_decay_optimization
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_tuning_dataset import GPTPromptTuningDataset
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.megatron_init import (
    initialize_model_parallel_for_nemo,
    set_jit_fusion_options,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
    init_method_normal,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.utils_funcs import get_last_rank, inject_model_parallel_rank
from nemo.utils import AppState, logging


class MegatronGPTModel(NLPModel):
    """
    Megatron GPT pretraining and prompt tuning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        # buffer used during train_step for logging average loss over gradient accumulation steps
        self._reduced_loss_buffer = []

        initialize_model_parallel_for_nemo(
            world_size=trainer.world_size,
            global_rank=trainer.global_rank,
            local_rank=trainer.local_rank,
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            pipeline_model_parallel_size=cfg.get('pipeline_model_parallel_size', 1),
            micro_batch_size=cfg.get('micro_batch_size'),
            seed=self.cfg.get('seed', 1234),
        )

        set_jit_fusion_options()

        self.tokenizer = get_nmt_tokenizer(
            library=self.cfg.tokenizer.library,
            model_name=self.cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer_model", self.cfg.tokenizer.model),
            vocab_file=self.register_artifact("vocab_file", self.cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("merges_file", self.cfg.tokenizer.merge_file),
        )

        vocab_size = self.tokenizer.vocab_size

        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=vocab_size,
            make_vocab_size_divisible_by=cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
        )

        # TODO: Not sure how to use lists of modules with PTL.
        # This means we can only use pipeline parallelism without the interleaved schedule.
        self.model = build_model(model_provider_func=self.model_provider_func, wrap_with_ddp=False)[0]

        self.use_soft_prompts = False

        if self.cfg.get('use_soft_prompts', False):
            self.use_soft_prompts = True
            self.prompts_to_tune = set([])
            self.prompt_table = set([])
            self.num_prompt_tokens = cfg.get('num_prompt_tokens', 10)

            if self.cfg.get('existing_prompt_tags', None):
                self.prompt_table = set(self.cfg.existing_prompt_tags)
        self.setup_optimizer_param_groups()

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = GPTModel(
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=self.cfg.get('use_cpu_initialization', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=self.cfg.get('onnx_safe', False),
            use_soft_prompts=self.cfg.get('use_soft_prompts', False),
            num_prompt_tokens=self.cfg.get('num_prompt_tokens', 10),
            prompt_tags=self.cfg.get('existing_prompt_tags', None),
        )
        return model

    def forward(self, tokens, text_position_ids, attention_mask, labels, prompt_tags=None):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels=labels, prompt_tags=prompt_tags,)
        return output_tensor

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = _get_params_for_weight_decay_optimization([self.model])

    def training_step(self, batch, batch_idx):
        # currently our dataloaders are producing a micro-batch here,
        # but we need this to be a "global batch" which will get split
        # into micro batches by fwd/bwd function
        # also need to add fwd/bwd function for non-pipeline case
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # we zero grads here because we also call backward here
            self._optimizer.zero_grad()
            batch_for_pipeline = [tokens, labels, loss_mask, attention_mask, position_ids]
            tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]
            reduced_loss = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.model,
                forward_only=False,
                tensor_shape=tensor_shape,
            )
            if not reduced_loss:
                loss = torch.tensor([0.0]).cuda()
            else:
                loss = reduced_loss[0]['avg']
            torch.distributed.broadcast(loss, get_last_rank())

            # This keeps loss and reduced_loss consistent with non-pipeline parallel
            loss = loss.squeeze()
            reduced_loss = [loss]
        else:
            output_tensor = self(tokens, position_ids, attention_mask, labels)
            loss = self.loss_func(loss_mask, output_tensor)
            reduced_loss = average_losses_across_data_parallel_group([loss])

        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True, rank_zero_only=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr, rank_zero_only=True)
            self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
            self.log(
                'consumed_samples',
                self.compute_consumed_samples(self.trainer.global_step),
                prog_bar=True,
                rank_zero_only=True,
            )
            self._reduced_loss_buffer = []

        return loss

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            When using pipeline parallel, we run backward in the fwd/bwd function.
            No need to call it here.
        """
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            return
        else:
            super().backward(*args, **kwargs)

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing when using pipeline parallel as we are calling
            backward during the training_step.
        """
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            return

        else:
            super().optimizer_zero_grad(*args, **kwargs)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            tokens, labels, loss_mask, attention_mask, position_ids = batch
            output_tensor = model(tokens, position_ids, attention_mask, labels)

            def loss_func(output_tensor):
                loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def validation_step(self, batch, batch_idx):
        reduced_loss = None
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            batch_for_pipeline = [tokens, labels, loss_mask, attention_mask, position_ids]
            tensor_shape = [self.cfg.encoder_seq_length, self.cfg.micro_batch_size, self.cfg.hidden_size]
            reduced_loss = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.model,
                forward_only=True,
                tensor_shape=tensor_shape,
            )
        else:
            output_tensor = self(tokens, position_ids, attention_mask, labels)
            loss = self.loss_func(loss_mask, output_tensor)
            reduced_loss = average_losses_across_data_parallel_group([loss])

        return reduced_loss

    def validation_epoch_end(self, outputs):
        if self.cfg.pipeline_model_parallel_size > 1:
            if parallel_state.is_pipeline_last_stage():
                outputs = [output[0]['avg'] for output in outputs]
            else:
                # only the last pipeline parallel stages return loss
                outputs = None

        if outputs is not None:
            averaged_loss = torch.stack(outputs).mean()

        else:
            averaged_loss = torch.tensor(0.0).cuda()

        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step), rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eos_id,
            self.cfg.data.get('reset_position_ids', False),
            self.cfg.data.get('reset_attention_mask', False),
            self.cfg.data.get('eod_mask_loss', False),
        )

        return tokens, labels, loss_mask, attention_mask, position_ids

    def build_train_valid_test_datasets(self):
        if self.use_soft_prompts:
            return

        logging.info('Building GPT datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size / self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def build_prompt_tuning_dataset(self, dataset_path):
        dataset = GPTPromptTuningDataset(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            num_prompt_tokens=self.cfg.num_prompt_tokens,
            max_seq_length=self.cfg.data.get('max_seq_length', 512),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos_eos=self.cfg.data.get('add_bos_eos', True),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

        return dataset, dataloader

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        if stage == 'predict':
            return

        # inject model parallel rank into resume path
        if self.trainer.checkpoint_connector.resume_from_checkpoint_fit_path is not None:
            self.trainer.checkpoint_connector.resume_from_checkpoint_fit_path = inject_model_parallel_rank(
                self.trainer.checkpoint_connector.resume_from_checkpoint_fit_path
            )

        # TODO: consider adding a ModelPT guard to check if model is being restored.
        # allowing restored models to optionally setup datasets
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        # if not using pipeline parallel, then this call will do nothing
        self.model.initialize_word_embeddings(
            init_method=init_method_normal(self.cfg.get('init_method_std', 0.02)),
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        )

    def setup_training_data(self, cfg):
        if self.use_soft_prompts:
            if cfg.get('train_ds', None):
                self._train_ds, self._train_dl = self.build_prompt_tuning_dataset(self.cfg.data.train_ds)
            else:
                raise AttributeError('No prompt tuning train dataset was specified in the cfg file')

            # Freeze all weights except prompt embeddings
            self.prompt_tuning_freeze()

        elif hasattr(self, '_train_ds'):
            resume_checkpoint_path = self.trainer.checkpoint_connector.resume_from_checkpoint_fit_path
            if resume_checkpoint_path:
                consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            else:
                consumed_samples = 0
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if self.use_soft_prompts:
            if cfg.get('valid_ds', None):
                self._validation_ds, self._validation_dl = self.build_prompt_tuning_dataset(self.cfg.data.valid_ds)
            else:
                raise AttributeError('No prompt tuning validation dataset was specified in the cfg file')

        elif hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if self.use_soft_prompts:
            if cfg.get('test_ds', None):
                self._test_ds, self._test_dl = self.build_prompt_tuning_dataset(self.cfg.data.test_ds)
            else:
                logging.info('No prompt tuning test dataset file provided in config, skipping')

        elif hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, global_step):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        parameters = self.get_parameters()
        clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

    def prompt_tuning_freeze(self):
        """Freeze weights of word embeddings and decoder, leaving only prompt embeddings unfrozen
        """
        for param in self.model.parameters():
            param.requires_grad = False

        # Only want new prompt tags to be tunable, leave existing prompt tags alone
        for prompt_tag in self.model.language_model.prompt_table.prompt_table.keys():
            if prompt_tag in self.prompts_to_tune:
                for param in self.model.language_model.prompt_table.prompt_table[prompt_tag].parameters():
                    param.requires_grad = True
            else:
                for param in self.model.language_model.prompt_table.prompt_table[prompt_tag].parameters():
                    param.requires_grad = False

    @classmethod
    def _bucketize_gpt_inference(cls, batch, use_soft_prompts=False):
        batch_tokens, lens, tokens_to_generate, compute_logprobs = batch[:4]
        batch_size = len(batch_tokens)
        tokens_to_generate = tokens_to_generate[0]
        batch_tokens = batch_tokens.tolist()

        if use_soft_prompts:
            prompt_tags = batch[4]

        # unpad tokens
        indxs = [index for index in range(batch_size)]
        for lenn, index in zip(lens, indxs):
            batch_tokens[index] = batch_tokens[index][:lenn]

        # chunk tokens by same length
        pre_buckets, lens = [], list(set(lens.tolist()))
        for lenn in lens:
            pre_buckets.append([(tokens, index) for index, tokens in enumerate(batch_tokens) if len(tokens) == lenn])

        buckets, positions, bucket_prompt_tags = [], [], []

        # get buckets and prompts initial positions
        for bucket in pre_buckets:
            buckets.append(torch.tensor([item[0] for item in bucket]).to(device='cuda'))
            positions.append([item[1] for item in bucket])

            # bucket prompt tags identically to their corresponding examples
            if use_soft_prompts:
                bucket_prompt_tags.append([prompt_tags[item[1]] for item in bucket])

        # Flatten position list
        positions = [item for sublist in positions for item in sublist]

        # Form request
        request = {"tokens": buckets, "prompt_tags": bucket_prompt_tags}

        return request, positions, tokens_to_generate, compute_logprobs[0]

    def get_parameters(self):
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request, positions, tokens_to_generate, compute_logprobs = MegatronGPTModel._bucketize_gpt_inference(
            batch, self.use_soft_prompts
        )

        if compute_logprobs:
            response = self.compute_logprobs(request, positions)
        else:
            response = self.complete(request, positions, tokens_to_generate)

        return response

    def complete(self, request: Dict, positions: List, tokens_to_generate: int):
        """
            Autoregressively invokes language model in the inference mode
        Args:
            request: 
                * tokens: List of "buckets" with unpadded tokens of the same length
                * prompt_tags: List of "buckets" where each bucket contains the prompt_tag strings
                               specifying the prompt tag to use (optional)
            positions: List with initial prompts positions
            tokens_to_generate: int value denoting amount of tokens model should generate

        Returns:	
            response: A python list of tuples
                (text, tokens, log_probs, offsets)
                * text: string, inputted prompt + generated text by model
                * tokens: list of tokens correspond to text
                * log_probs: list of tokens log probabilities
                * offsets: list of tokens start positions in text
                
        """
        results = []
        request_tokens = request["tokens"]

        for idx, tokens in enumerate(request_tokens):

            # For prompt tuned GPT models
            if self.use_soft_prompts:
                prompt_tags = request["prompt_tags"][idx]
            else:
                prompt_tags = None

            logsoftmaxlayer = torch.nn.LogSoftmax(dim=-1)

            for i in range(tokens_to_generate + 1):
                if self.use_soft_prompts:
                    batch_size = len(tokens)
                    full_length = len(tokens[0]) + self.num_prompt_tokens

                    # Get postion ids for text after soft prompt
                    position_ids = torch.arange(
                        start=self.num_prompt_tokens, end=full_length, dtype=torch.long, device=self.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand_as(tokens).clone()

                    # Make attention mask starting with first token in soft prompt
                    attention_mask = torch.tril(
                        torch.ones((batch_size, full_length, full_length), device=self.device)
                    ).view(batch_size, 1, full_length, full_length)
                    attention_mask = attention_mask < 0.5

                else:
                    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                        data=tokens,
                        eod_token=self.tokenizer.eos_id,
                        reset_position_ids=self.cfg.get('reset_position_ids', False),
                        reset_attention_mask=self.cfg.get('reset_attention_mask', False),
                        eod_mask_loss=self.cfg.get('eod_mask_loss', False),
                    )

                # No labels during inference. Still need masks to not attend to the right
                output_tensor = self(tokens, position_ids, attention_mask, prompt_tags=prompt_tags, labels=None)
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
                log_probs, token_ids = torch.max(logsoftmaxlayer(output_tensor), dim=-1)
                reached_eos = token_ids[0, -1].item() == self.tokenizer.eos_id
                tokens = torch.cat([tokens, torch.unsqueeze(token_ids[:, -1], 1)], dim=1)

            # add to results as (text, tokens, log_probs, offsets)
            for token, prob in zip(tokens, log_probs.tolist()):
                results.append(
                    (self.tokenizer.ids_to_text(token[:-1]), self.tokenizer.ids_to_tokens(token[:-1]), prob, [0])
                )
        # offsets calculation
        for item in results:
            for index, token in enumerate(item[1]):
                if index != len(item[1]) - 1:
                    item[3].append(len(token) + item[3][-1])
        # returnprompts in order they were inputted
        response = [0 for i in range(len(positions))]
        for item, index in zip(results, positions):
            response[index] = item

        return response

    def compute_logprobs(self, request: Dict, positions: List):
        """
            Only logprobs computation without generation tokens
        Args:
            request: 
                * tokens: List of "buckets" with unpadded tokens of the same length
                * prompt_tags: List of "buckets" where each bucket contains the prompt_tag strings
                                    specifying the prompt tag to use (optional)
            positions: List with initial prompts positions
        Returns:
            response: A python list of tuples
            (text, tokens, log_probs, offsets)
            * text: string, inputted prompt + generated text by model
            * tokens: list of tokens correspond to text
            * log_probs: list of log_softmax's from output_tensor in respect to text tokens
            * offsets: list of tokens start positions in text
        """
        results = []
        request_tokens = request["tokens"]
        logsoftmaxlayer = torch.nn.LogSoftmax(dim=-1)
        for idx, tokens in enumerate(request_tokens):
            tokens_cut = tokens[:, :-1]
            # For prompt tuned GPT models
            if self.use_soft_prompts:
                prompt_tags = request["prompt_tags"][idx]
            else:
                prompt_tags = None

            if self.use_soft_prompts:
                batch_size = len(tokens_cut)
                full_length = len(tokens_cut[0]) + self.num_prompt_tokens
                # Get postion ids for text after soft prompt
                position_ids = torch.arange(
                    start=self.num_prompt_tokens, end=full_length, dtype=torch.long, device=self.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(tokens_cut).clone()
                # Make attention mask starting with first token in soft prompt
                attention_mask = torch.tril(
                    torch.ones((batch_size, full_length, full_length), device=self.device)
                ).view(batch_size, 1, full_length, full_length)
                attention_mask = attention_mask < 0.5

            else:
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=tokens_cut,
                    eod_token=self.tokenizer.eos_id,
                    reset_position_ids=self.cfg.get('reset_position_ids', False),
                    reset_attention_mask=self.cfg.get('reset_attention_mask', False),
                    eod_mask_loss=self.cfg.get('eod_mask_loss', False),
                )
            output_tensor = self(tokens_cut, position_ids, attention_mask, prompt_tags=prompt_tags, labels=None)
            output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)

            log_probs = []
            for output in output_tensor:
                probs = F.log_softmax(output, dim=1)
                probs = probs[-len(tokens_cut[0]) :]
                log_probs.append(probs.to(dtype=torch.float16))

            for token, prob in zip(tokens, log_probs):
                results.append((self.tokenizer.ids_to_text(token), self.tokenizer.ids_to_tokens(token), prob, [0]))
        # offsets calculation
        for item in results:
            for index, token in enumerate(item[1]):
                if index != len(item[1]) - 1:
                    item[3].append(len(token) + item[3][-1])

        # return prompts in order they were inputted
        response = [0 for i in range(len(positions))]
        for item, index in zip(results, positions):
            response[index] = item

        return response

    def init_prompt_from_random(self, prompt_tag):
        self.model._init_prompt_from_random(prompt_tag)
        self._add_prompt_tag(prompt_tag)

    def init_prompt_from_text(self, prompt_tag, init_text):
        init_token_ids = self.tokenizer.text_to_ids(init_text)
        self.model._init_prompt_from_text(prompt_tag, init_token_ids)
        self._add_prompt_tag(prompt_tag)

    def get_prompt_table(self):
        if hasattr(self, 'prompt_table'):
            return self.prompt_table

    def list_available_models(self):
        return None

    def _add_prompt_tag(self, prompt_tag):
        if not hasattr(self, 'prompt_table'):
            raise AttributeError('Please set "use_soft_prompts" in cfg to True')

        self.prompt_table.add(prompt_tag)
        self.prompts_to_tune.add(prompt_tag)

        # Add new prompt tag to cfg for loading prompt table at inference
        with open_dict(self.cfg):
            self.cfg.existing_prompt_tags = list(self.prompt_table)

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after
