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

import glob
import hashlib
import json
import os
from os import path
from typing import Any, Dict, List, Optional, Union

import torch
from megatron import mpu
from megatron.checkpointing import get_checkpoint_version, set_checkpoint_version
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from torch.nn.parallel import DistributedDataParallel
from transformers import TRANSFORMERS_CACHE

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules import BertModule, MegatronBertEncoder
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.utils import AppState, app_state, logging
from nemo.utils.exp_manager import configure_checkpointing
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['NLPModel']

NEMO_NLP_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_nlp_tmp")

os.makedirs(NEMO_NLP_TMP, exist_ok=True)


class NLPModel(ModelPT, Exportable):
    """Base class for NLP Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.set_world_size(trainer)
        # if self._trainer is not None:
        #     if isinstance(self._trainer.accelerator_connector._training_type_plugin, DDPPlugin):
        #         self._trainer.accelerator_connector._training_type_plugin = NLPDDPPlugin()

    @rank_zero_only
    def register_bert_model(self):
        """Adds encoder config to .nemo archive.
        """
        # check if there is an encoder, warn if not
        if self.bert_model is None:
            raise ValueError('Instantiate self.bert_model before registering it.')
        else:
            # get encoder config and create source for artifact
            if isinstance(self.bert_model, MegatronBertEncoder):
                pretrained_model_name = self.bert_model._model_name
                encoder_config_path = pretrained_model_name + '_encoder_config.json'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path)
                config_for_json = OmegaConf.to_container(self.bert_model.config)
                with open(encoder_config_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(config_for_json, indent=2, sort_keys=True) + '\n')
                self.register_artifact(encoder_config_path, encoder_config_src)
                self.cfg.language_model.config_file = encoder_config_path
            elif isinstance(self.bert_model, BertModule):
                # HuggingFace Transformer Config
                pretrained_model_name = self.bert_model.name_or_path
                # Some HF names have "/" in them so we replace with _
                pretrained_model_name = pretrained_model_name.replace("/", "_")
                encoder_config_path = pretrained_model_name + '_encoder_config.json'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path)
                self.bert_model.config.to_json_file(encoder_config_src)  # name requested by jarvis team
                self.register_artifact(encoder_config_path, encoder_config_src)
                self.cfg.language_model.config_file = encoder_config_path
            else:
                logging.info(
                    f'Registering BERT model config for {self.bert_model} is not yet supported. Please override this method if needed.'
                )

    def setup_tokenizer(self, cfg: DictConfig):
        """Instantiates tokenizer based on config and registers tokenizer artifacts.

           If model is being restored from .nemo file then the tokenizer.vocab_file will
           be used (if it exists).

           Otherwise, we will use the vocab file provided in the config (if it exists).

           Finally, if no vocab file is given (this happens frequently when using HF),
           we will attempt to extract the vocab from the tokenizer object and then register it.

        Args:
            cfg (DictConfig): Tokenizer config
        """
        vocab_file = None
        if self._is_model_being_restored():
            if os.path.exists('tokenizer.vocab_file'):
                # model is being restored from .nemo file so tokenizer.vocab_file has precedence
                vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src='tokenizer.vocab_file')

            # tokenizer.vocab_file is added to the config file and registered as artifact for .nemo file
            # during training but this file is missing for load_from_checkpoint() method call
            # it's safer to use restore_from .nemo file
            elif cfg.vocab_file and not os.path.exists(cfg.vocab_file):
                logging.warning(
                    f'tokenizer.vocab_file not found at {cfg.vocab_file}. It is recommended to use restore_from() method with .nemo file.'
                )
            else:
                vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        elif cfg.vocab_file:
            # use vocab file from config
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=vocab_file,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )
        self.tokenizer = tokenizer

        if vocab_file is None:
            # when there is no vocab file we try to get the vocab from the tokenizer and register it
            self._register_vocab_from_tokenizer(vocab_file_config_path='tokenizer.vocab_file', cfg=cfg)

    @rank_zero_only
    def _register_vocab_from_tokenizer(
        self,
        vocab_file_config_path: str = 'tokenizer.vocab_file',
        vocab_dict_config_path: str = 'tokenizer_vocab_dict.json',
        cfg: DictConfig = None,
    ):
        """Creates vocab file from tokenizer if vocab file is None.

        Args:
            vocab_file_config_path: path to the vocab_file in the config
            vocab_dict_config_path: path to the vocab_dict in the config
            cfg: tokenizer config
        """
        if self.tokenizer is None:
            raise ValueError('Instantiate self.tokenizer before registering vocab from it.')
        else:
            if isinstance(self.tokenizer, AutoTokenizer):
                # extract vocab from tokenizer
                vocab_dict = self.tokenizer.tokenizer.get_vocab()

                # for fast and slow tokenizer vocabularies compatibility
                vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

                # get hash of vocab_dict to create a unique directory to write vocab_dict and vocab_file
                m = hashlib.md5()
                if 'tokenizer_name' in cfg:
                    if cfg.tokenizer_name is not None:
                        # different pretrained models with the same vocab will have different hash
                        m.update(cfg.tokenizer_name.encode())
                # get string representation of vocab_dict
                vocab_dict_str = json.dumps(vocab_dict, sort_keys=True).encode()
                m.update(vocab_dict_str)
                vocab_dict_hash = m.hexdigest()

                hash_path = os.path.join(NEMO_NLP_TMP, vocab_dict_hash)
                os.makedirs(hash_path, exist_ok=True)

                vocab_json_src = os.path.join(hash_path, vocab_dict_config_path)

                with open(vocab_json_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(vocab_dict, indent=2, sort_keys=True) + '\n')
                self.register_artifact(config_path=vocab_dict_config_path, src=vocab_json_src)
                # create vocab file
                vocab_file_src = os.path.join(hash_path, vocab_file_config_path)
                with open(vocab_file_src, 'w', encoding='utf-8') as f:
                    for key in vocab_dict:
                        f.write(key + '\n')

                cfg.vocab_file = vocab_file_src
                self.register_artifact(config_path=vocab_file_config_path, src=vocab_file_src)
            else:
                logging.info(
                    f'Registering tokenizer vocab for {self.tokenizer} is not yet supported. Please override this method if needed.'
                )

    def _clip_gradients(self, optimizer, clip_val=None):
        """ Override of PTL Gradient Clipping.
            Enables model parallel gradient clipping from Megatron-LM.

        Args:
            optimizer ([type]): [description]
            clip_val ([type], optional): [description]. Defaults to None.
        """
        app_state = AppState()

        # get clip_val from trainer if None is provided
        if clip_val is None:
            clip_val = float(self._trainer.gradient_clip_val)

        if app_state.model_parallel_size is not None:
            model = self._trainer.get_model()
            parameters = model.parameters()
            if mpu.model_parallel_is_initialized():
                mpu.grads.clip_grad_norm(parameters=parameters, max_norm=clip_val)
            else:
                raise ValueError('Model parallel groups must be intialized to use model parallel gradient clipping.')

        else:
            return Accelerator._clip_gradients(self, optimizer, clip_val)

    def setup(self, stage: str) -> None:
        """ PTL hook that is called on all DDP processes. """

        if stage == 'fit':

            # adds self.bert_model config to .nemo file
            if hasattr(self, 'bert_model') and self.bert_model is not None:
                self.register_bert_model()

            self._trainer.checkpoint_connector = NLPCheckpointConnector(self._trainer)

            app_state = AppState()

            if app_state.model_parallel_size is not None:

                # Configure checkpointing for model parallel
                if app_state.create_checkpoint_callback:
                    # global rank 0 is configured by exp_manager
                    if not is_global_rank_zero() and app_state.data_parallel_rank == 0:
                        configure_checkpointing(
                            self._trainer,
                            app_state.log_dir,
                            app_state.checkpoint_name,
                            app_state.checkpoint_callback_params,
                        )

    # def on_pretrain_routine_start(self) -> None:
    #     """ PTL hook that is called after DDP is initialized.
    #     """

    #     app_state = AppState()

    #     if app_state.model_parallel_size is not None:

    #         if isinstance(self.bert_model, MegatronBertEncoder):

    #             if app_state.model_parallel_group is None:
    #                 self.init_model_parallel(app_state.global_rank, app_state.world_size)

    #             # mpu grad clipping needs parameters to have the attribute model_parallel
    #             parameters = self._trainer.get_model().parameters()
    #             for p in parameters:
    #                 if not hasattr(p, 'model_parallel'):
    #                     p.model_parallel = False

    #             logging.info("Replacing sampler with model parallel sampler")
    #             mp_sampler = torch.utils.data.distributed.DistributedSampler(
    #                 self._train_dl.dataset,
    #                 num_replicas=app_state.data_parallel_size,
    #                 rank=app_state.data_parallel_rank,
    #             )
    #             mp_dl = self._trainer.replace_sampler(self._train_dl, mp_sampler)
    #             self._train_dl = mp_dl

    #             # TODO: figure out how to override clip gradients again
    #             # Update PTL trainer to use our _clip_gradients
    #             # self._trainer.accelerator_backend._clip_gradients = self._clip_gradients

    #             # model parallel checkpoints need to be restored after torch.distributed is initialized
    #             if self._trainer.resume_from_checkpoint is not None:
    #                 # update path based on model parallel rank
    #                 filepath = self._trainer.resume_from_checkpoint
    #                 dirname = os.path.dirname(os.path.dirname(filepath))
    #                 basename = os.path.basename(filepath)
    #                 filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
    #                 self._trainer.resume_from_checkpoint = filepath
    #                 logging.info(f'Resuming training from checkpoint {self._trainer.resume_from_checkpoint}')
    #                 # need to set checkpoint version for megatron-lm
    #                 checkpoint_version = torch.load(self._trainer.resume_from_checkpoint).get(
    #                     'checkpoint_version', None
    #                 )
    #                 if checkpoint_version is not None:
    #                     set_checkpoint_version(checkpoint_version)
    #                 else:
    #                     logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
    #                     set_checkpoint_version(0)
    #             else:
    #                 logging.info(
    #                     f"Restoring from pretrained model parallel checkpoint: {self.bert_model._restore_path}"
    #                 )
    #                 self.bert_model.restore_weights(self.bert_model._restore_path)

    #         else:
    #             raise NotImplementedError(
    #                 f'The BERT encoder: {self.bert_model} does not support model parallelism yet.'
    #             )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if hasattr(self, "bert_model") and isinstance(self.bert_model, MegatronBertEncoder):
            checkpoint['checkpoint_version'] = get_checkpoint_version()
        return None

    # remove rank check as model parallel models need to be saved on data parallel rank 0
    # @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        app_state = AppState()
        if app_state.model_parallel_size is not None:
            if app_state.data_parallel_rank == 0:
                # update .nemo path for model parallel save
                base_path = save_path[0:-5]
                save_path = f'{base_path}_mp_rank_{app_state.model_parallel_rank:02}.nemo'
                self._default_save_to(save_path)
            else:
                return
        else:
            # Add NeMo rank check as well
            if not is_global_rank_zero():
                return
            else:
                self._default_save_to(save_path)

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
        return_config: bool = False,
    ):
        """
        Restores model instance (weights and configuration) from .nemo file.

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

            Example:
                ```
                model = nemo.collections.asr.models.EncDecCTCModel.restore_from('asr.nemo')
                assert isinstance(model, nemo.collections.asr.models.EncDecCTCModel)
                ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        global _MODEL_RESTORE_PATH

        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        if os.path.isfile(restore_path):
            _MODEL_RESTORE_PATH = os.path.abspath(os.path.expanduser(restore_path))
            return cls._default_restore_from(restore_path, override_config_path, map_location, strict, return_config)
        elif os.path.isdir(restore_path):
            app_state = AppState()
            nemo_files = glob.glob(os.path.join(restore_path, '*.nemo'))
            model_parallel_size = len(nemo_files)
            app_state.model_parallel_size = model_parallel_size
            logging.info(
                (
                    f'restore_path: {restore_path} is a directory. '
                    f'Assuming megatron model parallelism with '
                    f'model_parallel_size: {model_parallel_size}'
                )
            )
            # set world size to model parallel size for inference
            # TODO: what to do for resuming training?
            app_state.world_size = model_parallel_size

            # try to get local rank from global
            local_rank = None
            try:
                local_rank = int(os.environ['LOCAL_RANK'])
            except:
                logging.info('Global variable LOCAL_RANK not yet specified')

            if local_rank is not None:
                app_state.local_rank = local_rank
            else:
                # if local is None then we are on the main process
                local_rank = 0

            model_parallel_rank = compute_model_parallel_rank(local_rank, model_parallel_size)
            app_state.model_parallel_rank = model_parallel_rank

            restore_path = ''
            # select .nemo file based on model parallel rank
            for file in nemo_files:
                if f'mp_rank_{app_state.model_parallel_rank:02d}.nemo' in file:
                    restore_path = file

            logging.info(f'Model parallel .nemo file detected. Restoring from: {restore_path}')
            _MODEL_RESTORE_PATH = os.path.abspath(os.path.expanduser(restore_path))
            return cls._default_restore_from(restore_path, override_config_path, map_location, strict, return_config)

    @property
    def input_module(self):
        return self.bert_model

    @property
    def output_module(self):
        return self.classifier


class NLPCheckpointConnector(CheckpointConnector):
    """ Override PTL CheckpointConnector to support model parallel checkpoints from Megatron-LM.
    """

    def __init__(self, trainer):
        super().__init__(trainer)

    def save_checkpoint(self, filepath, weights_only: bool):
        """Slightly modified version of PyTorch Lightning's save_checkpoint.

        Args:
            filepath ([str]): [description]
            weights_only (bool): [description]

        Returns:
            [type]: [description]
        """
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # filepath needs to be updated to include mp_rank
            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'

            # dump states as a checkpoint dictionary object
            checkpoint = self.dump_checkpoint(weights_only)

            # each model parallel rank needs to save a copy of its model
            if app_state.data_parallel_rank == 0:
                # write the checkpoint dictionary on the file
                if self.trainer.accelerator_backend:
                    checkpoint = self.trainer.accelerator_backend.on_save(checkpoint)
                try:
                    atomic_save(checkpoint, filepath)
                except AttributeError as err:
                    if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                        del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                    rank_zero_warn(
                        'Warning, `hyper_parameters` dropped from checkpoint.' f' An attribute is not picklable {err}'
                    )
                    atomic_save(checkpoint, filepath)
        return None


class NLPDDPPlugin(DDPPlugin):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.
    """

    distributed_backend = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, sync_batchnorm, **kwargs)

    def start_training(self, trainer: 'Trainer') -> None:
        """ PTL Hook that is called after DPP is initialized. """
        app_state = AppState()

        if app_state.model_parallel_size is not None:

            if isinstance(self.lightning_module.bert_model, MegatronBertEncoder):

                if app_state.model_parallel_group is None:
                    self.init_model_parallel(app_state.global_rank, app_state.world_size)

                # mpu grad clipping needs parameters to have the attribute model_parallel
                parameters = self.lightning_module.parameters()
                for p in parameters:
                    if not hasattr(p, 'model_parallel'):
                        p.model_parallel = False

                # logging.info("Replacing sampler with model parallel sampler")
                # mp_sampler = torch.utils.data.distributed.DistributedSampler(
                #     self._train_dl.dataset,
                #     num_replicas=app_state.data_parallel_size,
                #     rank=app_state.data_parallel_rank,
                # )
                # mp_dl = self._trainer.replace_sampler(self._train_dl, mp_sampler)
                # self._train_dl = mp_dl

                # TODO: figure out how to override clip gradients again
                # Update PTL trainer to use our _clip_gradients
                # self._trainer.accelerator_backend._clip_gradients = self._clip_gradients

                # model parallel checkpoints need to be restored after torch.distributed is initialized
                if self._trainer.resume_from_checkpoint is not None:
                    # update path based on model parallel rank
                    filepath = self._trainer.resume_from_checkpoint
                    dirname = os.path.dirname(os.path.dirname(filepath))
                    basename = os.path.basename(filepath)
                    filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
                    self._trainer.resume_from_checkpoint = filepath
                    logging.info(f'Resuming training from checkpoint {self._trainer.resume_from_checkpoint}')
                    # need to set checkpoint version for megatron-lm
                    checkpoint_version = torch.load(self._trainer.resume_from_checkpoint).get(
                        'checkpoint_version', None
                    )
                    if checkpoint_version is not None:
                        set_checkpoint_version(checkpoint_version)
                    else:
                        logging.warning('Megatron-lm checkpoint version not found. Setting checkpoint_version to 0.')
                        set_checkpoint_version(0)
                else:
                    logging.info(
                        f"Restoring from pretrained model parallel checkpoint: {self.bert_model._restore_path}"
                    )
                    self.bert_model.restore_weights(self.bert_model._restore_path)

            else:
                raise NotImplementedError(
                    f'The BERT encoder: {self.bert_model} does not support model parallelism yet.'
                )

        return super().start_training(trainer)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to True.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info("Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group,
            )

        else:
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model), device_ids=device_ids, **self._ddp_kwargs,
            )

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            if torch.distributed.is_initialized():
                mpu.initialize_model_parallel(app_state.model_parallel_size)
                app_state.model_parallel_group = mpu.get_model_parallel_group()
                app_state.data_parallel_group = mpu.get_data_parallel_group()
                app_state.model_parallel_rank = torch.distributed.get_rank(group=app_state.model_parallel_group)
                app_state.data_parallel_rank = torch.distributed.get_rank(group=app_state.data_parallel_group)
                logging.info(f'mp_rank: {app_state.model_parallel_rank}')
                logging.info(f'dp_rank: {app_state.data_parallel_rank}')

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:

            distributed_sampler_kwargs = dict(
                num_replicas=(app_state.data_parallel_size), rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super().distributed_sampler_kwargs
