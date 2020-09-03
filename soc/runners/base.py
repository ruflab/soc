import multiprocessing
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningModule
from typing import Union, Tuple, Optional, Callable
from ..typing import SocDataMetadata
from ..datasets import make_dataset
from ..models import make_model


class SOCRunner(LightningModule):
    """
        A runner represent a training pipeline.
        It contains everything from the dataset to the optimizer.

        Args:
            - config: Hyper parameters configuration
    """
    output_metadata: Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]
    val_dataset: Optional[Dataset] = None
    train_dataset: Dataset
    collate_fn: Optional[Callable]

    def __init__(self, config):
        super(SOCRunner, self).__init__()
        self.hparams = config
        self.val_dataset = None

    def prepare_data(self):
        # Download data here if needed
        pass

    def setup(self, stage):
        self.train_dataset, self.val_dataset = self.setup_dataset(self.hparams)

        self.output_metadata = self.train_dataset.get_output_metadata()
        self.collate_fn = self.train_dataset.get_collate_fn()
        self.hparams.model['data_input_size'] = self.train_dataset.get_input_size()
        self.hparams.model['data_output_size'] = self.train_dataset.get_output_size()

        if self.val_dataset is None:
            self.train_dataset, self.val_dataset = self.split_dataset(self.train_dataset)

        self.model = make_model(self.hparams.model)

    def setup_dataset(self, hparams):
        """This function purpose is mainly to be overrided for tests"""
        train_dataset = make_dataset(hparams.dataset)
        if 'val_dataset' in hparams and hparams['val_dataset'] is not None:
            val_dataset = make_dataset(hparams.val_dataset)
        else:
            val_dataset = None

        return train_dataset, val_dataset

    def split_dataset(self, dataset, percent: float = 0.9):
        ds_len = len(dataset)
        train_len = min(round(percent * ds_len), ds_len - 1)
        val_len = ds_len - train_len
        soc_train, soc_val = random_split(dataset, [train_len, val_len], torch.Generator())

        return soc_train, soc_val

    def train_dataloader(self):
        # Ho my god! -_- overfit_batches is broken
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/2311
        ds_params = self.hparams.dataset
        shuffle = ds_params['shuffle'] if 'shuffle' in ds_params else True

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=shuffle,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams['batch_size'],
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        return dataloader

    def configure_optimizers(self):
        parameters = self.get_parameters()

        if self.hparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                parameters, lr=self.hparams['lr'], weight_decay=self.hparams.weight_decay
            )
        elif self.hparams['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.hparams['lr'],
                weight_decay=self.hparams.weight_decay,
                amsgrad=self.hparams.amsgrad
            )
        else:
            raise Exception('Unknown optimizer {}'.format(self.hparams['optimizer']))

        if self.hparams['scheduler'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001
            )

            return optimizer, scheduler
        elif self.hparams['scheduler'] == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams['lr'],
                max_lr=10 * self.hparams['lr'],
            )

            return optimizer, scheduler

        return optimizer

    def get_parameters(self):
        return self.model.parameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        def _mean(res, key):
            if key == 'playerressources_post_trade_acc':
                all_val = [x[key] for x in res if x[key] != -1]
            else:
                all_val = [x[key] for x in res]
            if len(all_val) == 0:
                return -1.
            return torch.stack(all_val).mean()

        logs = {}
        for key in outputs[0].keys():
            logs[key] = _mean(outputs, key)

        final_dict = {'val_accuracy': logs['val_accuracy'], 'log': logs}

        return final_dict
