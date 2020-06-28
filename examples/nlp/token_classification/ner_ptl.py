"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl

from transformers import BertModel
from nemo.collections.nlp.data.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.backends.pytorch.common.parts import MultiLayerPerceptron
from nemo.collections.nlp.utils.functional_utils import gelu
from nemo.collections.nlp.utils.transformer_utils import transformer_weights_init
from nemo.collections.nlp.data import BertTokenClassificationDataset
from typing import Optional, Dict

ACT2FN = {"gelu": gelu, "relu": nn.functional.relu}

class NERModel(pl.LightningModule):
    """
    Sample model to show how to define a template.

    Example:

        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     batch_size=2,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001 * 8,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     hidden_dim=1000,
        ... )
        >>> model = LightningTemplateModel(**params)
    """

    def __init__(self,
                 num_classes,
                 pretrained_model_name='bert-base-cased',
                 activation='relu',
                 log_softmax=True,
                 dropout=0.0,
                 use_transformer_pretrained=True
                 ):
        # init superclass
        super().__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self.classifier = TokenClassifier(hidden_size=self.hidden_size,num_classes=num_classes, activation=activation, log_softmax=log_softmax, dropout=dropout, use_transformer_pretrained=use_transformer_pretrained)

        self.loss = nn.CrossEntropyLoss()
        # This will be set by setup_training_datai
        self.__train_dl = None
        # This will be set by setup_validation_data
        self.__val_dl = None
        # This will be set by setup_test_data
        self.__test_dl = None
        # This will be set by setup_optimization
        self.__optimizer = None

    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(input_ids, token_type_ids, attention_mask)[0]
        logits = self.classifier(hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        #TODO replace with loss module
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)
        loss = self.loss(logits_flatten, labels_flatten)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)
        val_loss = self.loss(logits_flatten, labels_flatten)

        tensorboard_logs = {'val_loss': val_loss}
        # TODO - add eval - callback?
        # labels_hat = torch.argmax(y_hat, dim=1)
        # n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, 'log': tensorboard_logs} #, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        # tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss} #, 'log': tensorboard_logs}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
    #     tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
    #     return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, data_dir, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        text_file = os.path.join(data_dir, 'text_train.txt')
        labels_file = os.path.join(data_dir, 'labels_train.txt')
        self.__train_dl = self.__setup_dataloader_ner(text_file, labels_file)

    def setup_validation_data(self, data_dir, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        text_file = os.path.join(data_dir, 'text_dev.txt')
        labels_file = os.path.join(data_dir, 'labels_dev.txt')
        self.__val_dl = self.__setup_dataloader_ner(text_file, labels_file)

    # def setup_test_data(self, test_data_layer_params: Optional[Dict]):
    #     if 'shuffle' not in test_data_layer_params:
    #         test_data_layer_params['shuffle'] = False
    #     self.__test_dl = self.__setup_dataloader_from_config(config=test_data_layer_params)

    def setup_optimization(self, optim_params: Optional[Dict], optimizer='adam'):
        if optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.parameters(), lr=optim_params['lr'])
        else:
            raise ValueError (f'TODO {optimizer}')

    def __setup_dataloader_ner(self, text_file,
        label_file,
        max_seq_length=128,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        shuffle = False,
        batch_size = 64,
        num_workers = 0,
    ):

        dataset = BertTokenClassificationDataset(
            text_file= text_file,
            label_file= label_file,
            max_seq_length= max_seq_length,
            tokenizer= self.tokenizer,
            num_samples= num_samples,
            pad_label= pad_label,
            label_ids= label_ids,
            ignore_extra_tokens= ignore_extra_tokens,
            ignore_start_end= ignore_start_end,
            use_cache= use_cache)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def configure_optimizers(self):
        return self.__optimizer

    def train_dataloader(self):
        return self.__train_dl

    def val_dataloader(self):
        return self.__val_dl

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser


class TokenClassifier(nn.Module):
    def __init__(
        self,
            hidden_size: object,
            num_classes: object,
            activation: object = 'relu',
            log_softmax: object = True,
            dropout: object = 0.0,
            use_transformer_pretrained: object = True,
    ) -> object:
        super().__init__()
        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=1, activation=activation, log_softmax=log_softmax
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits



if __name__ == '__main__':
    ner_model = NERModel(num_classes=9)
    ner_model.setup_training_data(data_dir='/home/ebakhturina/data/ner/conell/', train_data_layer_params={'shuffle':True})
    ner_model.setup_validation_data(data_dir='/home/ebakhturina/data/ner/conell/', val_data_layer_params={'shuffle':False})
    ner_model.setup_optimization(optim_params={'lr': 0.0003})
    trainer = pl.Trainer(
        val_check_interval=35, amp_level='O1', precision=16, gpus=2, max_epochs=123, distributed_backend='ddp')
    trainer.fit(ner_model)

