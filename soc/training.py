import multiprocessing
import os
import torch
import pprint
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from typing import Callable, List, Any, Dict
from .typing import SocSeqBatch, SocBatch, SocConfig
from .models import make_model
from .datasets import make_dataset

CollateFnType = Callable[[List[Any]], Any]


class Runner(pl.LightningModule):
    def __init__(self, config: SocConfig):
        super(Runner, self).__init__()
        self.hparams = config

        if self.hparams['loss_name'] == 'mse':
            self.loss_f = F.mse_loss
        else:
            raise Exception('Unknown loss function {}'.format(self.hparams['loss_name']))

    def prepare_data(self):
        # Download data here if needed
        pass

    def setup(self, stage):
        dataset = self.setup_dataset()

        ds_len = len(dataset)
        train_len = min(round(0.9 * ds_len), ds_len - 1)
        val_len = ds_len - train_len
        soc_train, soc_val = random_split(dataset, [train_len, val_len])
        self.train_dataset = soc_train
        self.val_dataset = soc_val
        self.training_type = dataset.get_training_type()
        self.collate_fn = dataset.get_collate_fn()
        self.hparams['data_input_size'] = dataset.get_input_size()
        self.hparams['data_output_size'] = dataset.get_output_size()

        self.model = make_model(self.hparams)

    def setup_dataset(self):
        """This function purpose is mainly to be overrided for tests"""
        dataset = make_dataset(self.hparams)

        return dataset

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams['batch_size'],
            num_workers=multiprocessing.cpu_count(),
            collate_fn=self.collate_fn,
        )

        return dataloader

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
        if self.training_type == 'supervised_seq':
            loss = train_on_supervised_seq_batch(batch, self.model, self.loss_f)
        elif self.training_type == 'supervised_forward':
            loss = train_on_supervised_forward_batch(batch, self.model, self.loss_f)
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(self.training_type)
            )

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        if self.training_type == 'supervised_seq':
            loss = train_on_supervised_seq_batch(batch, self.model, self.loss_f)
        elif self.training_type == 'supervised_forward':
            loss = train_on_supervised_forward_batch(batch, self.model, self.loss_f)
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(self.training_type)
            )

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


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

    y_preds_reshaped = y_preds.reshape(bs * seq_len, C, W, H)
    y_true_reshaped = y_true.reshape(bs * seq_len, C, W, H)

    losses_dict = compute_losses_on_state_seq(loss_f, y_preds_reshaped, y_true_reshaped)
    losses = list(losses_dict.values())
    loss = torch.mean(torch.stack(losses))

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


def compute_losses_on_state_seq(
        loss_f: Callable, y_preds_seq_t: torch.Tensor, y_true_seq_t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    loss_properties = loss_f(y_preds_seq_t[:, 0:9], y_true_seq_t[:, 0:9])

    loss_pieces = loss_f(y_preds_seq_t[:, 9:81], y_true_seq_t[:, 9:81])

    loss_public_infos = loss_f(y_preds_seq_t[:, 81:], y_true_seq_t[:, 81:])
    losses = {
        'loss_properties': loss_properties,
        'loss_pieces': loss_pieces,
        'loss_public_infos': loss_public_infos,
    }

    return losses


def train(config: SocConfig):
    # Misc part
    if config['generic']['verbose'] is True:
        import copy
        tmp_config = copy.deepcopy(config)
        if "gpus" in tmp_config['trainer']:
            del tmp_config['trainer']["gpus"]
        if "tpu_cores" in tmp_config['trainer']:
            del tmp_config['trainer']["tpu_cores"]
        pprint.pprint(tmp_config)

    pl.seed_everything(config['generic']['seed'])

    # Runner part
    runner = Runner(config['generic'])

    # Trainer part
    config['trainer']['deterministic'] = True
    # config['trainer'][' distributed_backend'] = 'dp'

    if 'default_root_dir' not in config['trainer'] or config['trainer']['default_root_dir'] is None:
        cfd = os.path.dirname(os.path.realpath(__file__))
        default_results_dir = os.path.join(cfd, '..', 'scripts', 'results')
        config['trainer']['default_root_dir'] = default_results_dir

    save_top_k = 1
    if 'other' in config:
        if 'save_top_k' in config['other']:
            save_top_k = config['other']['save_top_k']
    checkpoint_callback = ModelCheckpoint(
        filepath=config['trainer']['default_root_dir'],
        verbose=True,
        save_top_k=save_top_k,
        monitor='val_loss',
        mode='min',
    )
    config['trainer']['checkpoint_callback'] = checkpoint_callback

    if 'logger' in config['trainer']:
        if config['trainer']['logger'] == 'neptune':
            config['trainer']['logger'] = NeptuneLogger(
                api_key=os.environ['NEPTUNE_API_TOKEN'],
                project_name=os.environ['NEPTUNE_PROJECT_NAME'],
                params=config['generic'],
            )

    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(runner)
