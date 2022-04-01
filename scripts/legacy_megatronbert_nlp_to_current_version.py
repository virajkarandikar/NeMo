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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

import argparse
import logging
import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.core import ModelPT
from nemo.core.classes import Exportable, typecheck
from nemo.core.config import TrainerConfig

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Export NLP models pretrained with Megatron Bert on NeMo < 1.5.0 to current version ",
    )
    parser.add_argument("source", help="Source .nemo file")
    parser.add_argument("out", help="Location to write result to")
    parser.add_argument("--autocast", action="store_true", help="Use autocast when exporting")
    parser.add_argument(
        "--megatron-checkpoint",
        type=str,
        help="Path of the MegatronBert nemo checkpoint converted from MegatronLM using megatron_lm_ckpt_to_nemo.py file (Not NLP model checkpoint)",
    )
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    parser.add_argument("--device", default="cuda", help="Device to export for")
    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level

    logger = logging.getLogger(__name__)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    logging.basicConfig(level=loglevel, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Logging level set to {}".format(loglevel))

    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    out = args.out

    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        gpus=1,
        accelerator="ddp",
        num_nodes=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        checkpoint_callback=False,
    )
    trainer = pl.Trainer(cfg_trainer)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    autocast = nullcontext
    try:
        with torch.inference_mode():
            # If the megatron based NLP model was trained on NeMo < 1.5, then we need to update the lm_checkpoint on the model config
            if args.megatron_checkpoint:
                model_cfg = ModelPT.restore_from(
                    restore_path=nemo_in, trainer=trainer, megatron_legacy=True, return_config=True
                )
                OmegaConf.set_struct(model_cfg, True)
                with open_dict(model_cfg):
                    model_cfg.language_model.lm_checkpoint = args.megatron_checkpoint
                model = ModelPT.restore_from(
                    restore_path=nemo_in, trainer=trainer, megatron_legacy=True, override_config_path=model_cfg,
                )
            else:
                logging.error("Megatron checkpoint path must be provided if megatron_legacy is selected")
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.cfg.target, nemo_in))

    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.cfg.target))
        sys.exit(1)
    typecheck.set_typecheck_enabled(enabled=False)

    try:
        model = model.to(device=args.device)
        model.eval()
        if args.autocast:
            autocast = torch.cuda.amp.autocast
        with autocast(), torch.inference_mode():
            model.save_to(out)
    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.cfg.target
            )
        )
        raise e

    logging.info("Successfully exported to {}".format(out))

    del model


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
