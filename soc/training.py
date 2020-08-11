import multiprocessing
import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.nn import Module
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from typing import Callable, List, Any, Dict, Optional
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from .typing import SocSeqBatch, SocBatch, SocBatchMultipleOut, SocDataMetadata
from .typing import SocSeqPolicyBatch
from .models import make_model
from .datasets import make_dataset
from . import val
from .losses import compute_losses
from .val import compute_accs

CollateFnType = Callable[[List[Any]], Any]


@dataclass
class GenericConfig:
    seed: int = 1
    verbose: bool = False
    dataset: Any = MISSING
    model: Any = MISSING
    lr: float = 3e-3
    optimizer: str = 'adam'
    scheduler: Optional[str] = None
    batch_size: int = 32
    weight_decay: Optional[float] = 0.


@dataclass
class SocConfig:
    defaults: List[Any] = MISSING

    generic: GenericConfig = GenericConfig()
    trainer: Any = MISSING
    other: Any = MISSING


class Runner(pl.LightningModule):
    """
        A runner represent a training pipeline.
        It contains everything from the dataset to the optimizer.

        Args:
            - config: Hyper parameters configuration
    """
    def __init__(self, config):
        super(Runner, self).__init__()
        self.hparams = config

    def prepare_data(self):
        # Download data here if needed
        pass

    def setup(self, stage):
        dataset = self.setup_dataset()

        self.train_dataset, self.val_dataset = self.split_dataset(dataset)
        self.training_type = dataset.get_training_type()
        self.metadata = dataset.get_output_metadata()
        self.collate_fn = dataset.get_collate_fn()
        self.hparams.model['data_input_size'] = dataset.get_input_size()
        self.hparams.model['data_output_size'] = dataset.get_output_size()

        self.model = make_model(self.hparams.model)

    def split_dataset(self, dataset, percent: float = 0.9):
        ds_len = len(dataset)
        train_len = min(round(percent * ds_len), ds_len - 1)
        val_len = ds_len - train_len
        soc_train, soc_val = random_split(dataset, [train_len, val_len])

        return soc_train, soc_val

    def setup_dataset(self):
        """This function purpose is mainly to be overrided for tests"""
        dataset = make_dataset(self.hparams.dataset)

        return dataset

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
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams['lr'],
                weight_decay=self.hparams.weight_decay
            )
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
            train_dict = train_on_supervised_seq_batch(batch, self.model, self.metadata)
        elif self.training_type == 'supervised_seq_policy':
            train_dict = train_on_supervised_seq_policy_batch(batch, self.model, self.metadata)
        elif self.training_type == 'supervised_forward':
            train_dict = train_on_supervised_forward_batch(batch, self.model, self.metadata)
        elif self.training_type == 'resnet18policy':
            train_dict = train_on_resnet18policy_batch(batch, self.model, self.metadata)
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(self.training_type)
            )

        final_dict = {'loss': train_dict['train_loss'], 'log': train_dict}

        return final_dict

    def validation_step(self, batch, batch_idx):
        if self.training_type == 'supervised_seq':
            val_dict = val_on_supervised_seq_batch(batch, self.model, self.metadata)
        elif self.training_type == 'supervised_seq_policy':
            val_dict = val_on_supervised_seq_policy_batch(batch, self.model, self.metadata)
        elif self.training_type == 'supervised_forward':
            val_dict = val_on_supervised_forward_batch(batch, self.model, self.metadata)
        elif self.training_type == 'resnet18policy':
            val_dict = val_on_resnet18policy_batch(batch, self.model, self.metadata)
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(self.training_type)
            )

        return val_dict

    def validation_epoch_end(self, outputs):
        def _mean(res, key):
            return torch.stack([x[key] for x in res]).mean()

        logs = {}
        for k in outputs[0].keys():
            logs[k] = _mean(outputs, k)

        final_dict = {'val_accuracy': logs['val_accuracy'], 'log': logs}

        return final_dict


def train_on_supervised_seq_batch(batch: SocSeqBatch, model: Module,
                                  metadata: SocDataMetadata) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y, mask) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """
    x = batch[0]
    y_true = batch[1]
    mask = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x)
    y_logits_raw = outputs[0]
    y_logits = y_logits_raw * mask

    train_dict = compute_losses(metadata, y_logits, y_true)

    loss = torch.tensor(0., device=y_logits.device)
    for k, l in train_dict.items():
        loss += l
    train_dict['train_loss'] = loss

    return train_dict


def val_on_supervised_seq_batch(
    batch: SocSeqBatch,
    model: Module,
    metadata: SocDataMetadata,
) -> Dict[str, torch.Tensor]:
    """This function computes the validation loss and accuracy of the model."""

    x = batch[0]
    y_true = batch[1]
    mask = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x)
    y_logits_raw = outputs[0]
    y_logits = y_logits_raw * mask

    val_dict = compute_accs(metadata, y_logits, y_true)

    one_meta = {'piecesonboard_one_mean': metadata['piecesonboard']}
    if 'actions' in metadata.keys():
        one_meta['actions_one_mean'] = metadata['actions']
    val_dict.update(val.get_stats(one_meta, torch.round(y_logits), 1))

    val_acc = torch.tensor(0., device=y_logits.device)
    for k, acc in val_dict.items():
        val_acc += acc
    val_acc /= len(val_dict)

    val_dict['val_accuracy'] = val_acc

    return val_dict


def train_on_supervised_seq_policy_batch(
    batch: SocSeqPolicyBatch, model: Module, metadata: List[SocDataMetadata]
) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y, mask) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """
    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    mask_spatial, mask_linear, mask_action = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x_seq)
    y_spatial_s_logits_seq_raw, y_s_logits_seq_raw, y_a_logits_seq_raw = outputs[0]
    y_spatial_s_logits_seq = y_spatial_s_logits_seq_raw * mask_spatial
    y_s_logits_seq = y_s_logits_seq_raw * mask_linear
    y_a_logits_seq = y_a_logits_seq_raw * mask_action

    spatial_metadata, linear_metadata, actions_metadata = metadata

    train_dict = {}
    train_dict.update(
        compute_losses(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq)
    )
    train_dict.update(compute_losses(linear_metadata, y_s_logits_seq, y_s_true_seq))
    train_dict.update(compute_losses(actions_metadata, y_a_logits_seq, y_a_true_seq))

    loss = torch.tensor(0., device=y_spatial_s_logits_seq.device)
    for k, l in train_dict.items():
        loss += l

    train_dict['train_loss'] = loss

    return train_dict


def val_on_supervised_seq_policy_batch(
    batch: SocSeqPolicyBatch, model: Module, metadata: List[SocDataMetadata]
) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y, mask) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """
    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    mask_spatial, mask_linear, mask_action = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x_seq)
    y_spatial_s_logits_seq_raw, y_s_logits_seq_raw, y_a_logits_seq_raw = outputs[0]
    y_spatial_s_logits_seq = y_spatial_s_logits_seq_raw * mask_spatial
    y_s_logits_seq = y_s_logits_seq_raw * mask_linear
    y_a_logits_seq = y_a_logits_seq_raw * mask_action

    spatial_metadata, linear_metadata, actions_metadata = metadata

    val_dict = {}
    val_dict.update(compute_accs(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq))
    val_dict.update(compute_accs(linear_metadata, y_s_logits_seq, y_s_true_seq))
    val_dict.update(compute_accs(actions_metadata, y_a_logits_seq, y_a_true_seq))

    one_meta = {'piecesonboard_one_mean': spatial_metadata['piecesonboard']}
    val_dict.update(val.get_stats(one_meta, torch.round(torch.sigmoid(y_spatial_s_logits_seq)), 1))

    val_acc = torch.tensor(0., device=y_spatial_s_logits_seq.device)
    for k, acc in val_dict.items():
        val_acc += acc
    val_acc /= len(val_dict)

    val_dict['val_accuracy'] = val_acc

    return val_dict


def train_on_supervised_forward_batch(batch: SocBatch, model: Module,
                                      metadata: SocDataMetadata) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """
    x = batch[0]
    y_true = batch[1]

    y_logits = model(x)

    train_dict = compute_losses(metadata, y_logits, y_true)

    loss = torch.tensor(0., device=y_logits.device)
    for k, l in train_dict.items():
        loss += l
    train_dict['train_loss'] = loss

    return train_dict


def val_on_supervised_forward_batch(
    batch: SocBatch,
    model: Module,
    metadata: SocDataMetadata,
) -> Dict[str, torch.Tensor]:
    """This function computes the validation loss and accuracy of the model."""

    x = batch[0]
    y_true = batch[1]

    y_logits = model(x)

    val_dict = compute_accs(metadata, y_logits, y_true)

    if 'mean_piecesonboard' in metadata.keys():
        prefix = 'mean_'
    else:
        prefix = ''
    one_meta = {'piecesonboard_one_mean': metadata[prefix + 'piecesonboard']}
    if 'actions' in metadata.keys() or 'mean_actions' in metadata.keys():
        one_meta['actions_one_mean'] = metadata[prefix + 'actions']
    val_dict.update(val.get_stats(one_meta, torch.round(y_logits), 1))

    val_acc = torch.tensor(0., device=y_logits.device)
    for k, acc in val_dict.items():
        val_acc += acc
    val_acc /= len(val_dict)

    val_dict['val_accuracy'] = val_acc

    return val_dict


def train_on_resnet18policy_batch(
    batch: SocBatchMultipleOut,
    model: Module,
    metadata: List[SocDataMetadata],
) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """
    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = model(x_seq)

    spatial_metadata, linear_metadata, actions_metadata = metadata

    train_dict = {}
    train_dict.update(
        compute_losses(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq)
    )
    train_dict.update(compute_losses(linear_metadata, y_s_logits_seq, y_s_true_seq))
    train_dict.update(compute_losses(actions_metadata, y_a_logits_seq, y_a_true_seq))

    loss = torch.tensor(0., device=y_spatial_s_logits_seq.device)
    for k, l in train_dict.items():
        loss += l

    train_dict['train_loss'] = loss

    return train_dict


def val_on_resnet18policy_batch(
    batch: SocBatchMultipleOut,
    model: Module,
    metadata: List[SocDataMetadata],
) -> Dict[str, torch.Tensor]:
    """
        This function computes the validation loss and accuracy of the model.

        Args:
            - batch: (x, y) batch of data
            - model: (Module) the model
            - metadata: (Dict) metadata to compute losses
    """

    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = model(x_seq)

    spatial_metadata, linear_metadata, actions_metadata = metadata

    val_dict = {}
    val_dict.update(compute_accs(spatial_metadata, y_spatial_s_logits_seq, y_spatial_s_true_seq))
    val_dict.update(compute_accs(linear_metadata, y_s_logits_seq, y_s_true_seq))
    val_dict.update(compute_accs(actions_metadata, y_a_logits_seq, y_a_true_seq))

    one_meta = {'piecesonboard_one_mean': spatial_metadata['piecesonboard']}
    val_dict.update(val.get_stats(one_meta, torch.round(torch.sigmoid(y_spatial_s_logits_seq)), 1))

    val_acc = torch.tensor(0., device=y_spatial_s_logits_seq.device)
    for k, acc in val_dict.items():
        val_acc += acc
    val_acc /= len(val_dict)

    val_dict['val_accuracy'] = val_acc

    return val_dict


def train(config: DictConfig) -> Runner:
    # Misc part
    if config['generic']['verbose'] is True:
        print(config.pretty())

    pl.seed_everything(config['generic']['seed'])

    # Runner part
    runner = Runner(config['generic'])

    ###
    # LR finder
    # The prepare_data/setup does not work well with the lr finder
    # To handle the situation we manually search for it before the training
    # The situation is being handled:
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2485
    ###
    if "auto_lr_find" in config['trainer'] and config['trainer']['auto_lr_find'] is True:
        del config['trainer']['auto_lr_find']
        tmp_trainer = pl.Trainer(**config['trainer'])
        runner.prepare_data()
        runner.setup('lr_finder')
        lr_finder = tmp_trainer.lr_find(runner)
        # fig = lr_finder.plot(suggest=True)
        new_lr = lr_finder.suggestion()
        config['generic']['lr'] = new_lr
        runner = Runner(config['generic'])

        if config['generic'].get('verbose', False) is True:
            print('Learning rate found: {}'.format(new_lr))

    config['trainer']['deterministic'] = True
    # config['trainer'][' distributed_backend'] = 'dp'

    if 'default_root_dir' not in config['trainer'] or config['trainer']['default_root_dir'] is None:
        cfd = os.path.dirname(os.path.realpath(__file__))
        default_results_dir = os.path.join(cfd, '..', 'scripts', 'results')
        config['trainer']['default_root_dir'] = default_results_dir

    ###
    # Checkpointing
    ###
    save_top_k = 1
    if 'other' in config:
        if 'save_top_k' in config['other']:
            save_top_k = config['other']['save_top_k']
    checkpoint_callback = ModelCheckpoint(
        filepath=config['trainer']['default_root_dir'],
        verbose=True,
        save_top_k=save_top_k,
        save_last=True,
        monitor='val_accuracy',
        mode='max',
    )
    config['trainer']['checkpoint_callback'] = checkpoint_callback

    ###
    # Logging
    ###
    if 'logger' in config['trainer']:
        if config['trainer']['logger'] == 'neptune':
            config['trainer']['logger'] = NeptuneLogger(
                api_key=os.environ['NEPTUNE_API_TOKEN'],
                project_name=os.environ['NEPTUNE_PROJECT_NAME'],
                params=config['generic'],
            )

    # ###
    # # Early stopping
    # # It is breaking neptune logging somehow, it seems that it overrides by 1 the current timestep
    # ###
    # early_stop_callback = EarlyStopping(
    #     monitor='val_accuracy', min_delta=0.00, patience=10, verbose=False, mode='max'
    # )
    # config['trainer']['early_stop_callback'] = early_stop_callback

    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(runner)

    return runner
