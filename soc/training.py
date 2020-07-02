import multiprocessing
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Callable, List, Any
from .typing import SocSeqBatch, SocBatch
from .models import make_model
from .datasets import make_dataset

CollateFnType = Callable[[List[Any]], Any]


class Runner(pl.LightningModule):
    def __init__(self, config):
        super(Runner, self).__init__()
        self.hparams = config

        # Data
        self.dataset = make_dataset(config)
        self.hparams['data_input_size'] = self.dataset.get_input_size()
        self.hparams['data_output_size'] = self.dataset.get_output_size()

        self.model = make_model(self.hparams)

        if self.hparams['loss_name'] == 'mse':
            self.loss_f = F.mse_loss
        else:
            raise Exception('Unknown loss function {}'.format(config['loss_name']))

    def configure_optimizers(self):
        if self.hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        else:
            raise Exception('Unknown optimizer {}'.format(self.hparams['optimizer']))

        if self.hparams['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001
            )

            return optimizer, scheduler

        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        training_type = self.dataset.get_training_type()
        if training_type == 'supervised_seq':
            loss = train_on_supervised_seq_batch(batch, self.model, self.loss_f)
        elif training_type == 'supervised_forward':
            loss = train_on_supervised_forward_batch(batch, self.model, self.loss_f)
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(training_type)
            )

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=multiprocessing.cpu_count() - 1,
            collate_fn=self.dataset.get_collate_fn(),
            drop_last=True,
            pin_memory=device != "cpu"
        )

        return dataloader


def train_on_supervised_seq_batch(
        batch: SocSeqBatch, model: Module, loss_f: Callable
) -> torch.Tensor:
    """
        This function apply an batch update to the model.

        The dataloader is expected to to provide a tuple of x and y values
    """
    x = batch[0]
    y_true = batch[1]
    mask = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x)
    y_preds_raw = outputs[0]
    y_preds = y_preds_raw * mask

    bs, seq_len, C, H, W = y_true.shape

    loss = loss_f(y_preds.reshape(bs * seq_len, C, W, H), y_true.reshape(bs * seq_len, C, W, H))

    return loss


def train_on_supervised_forward_batch(
        batch: SocBatch, model: Module, loss_f: Callable
) -> torch.Tensor:
    """
        This function apply an batch update to the model.
    """
    x = batch[0]
    y_true = batch[1]

    y_preds = model(x)

    loss = loss_f(y_preds, y_true)

    return loss
