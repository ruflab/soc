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
from .typing import SocSeqBatch, SocBatch, SocConfig, SocBatchMultipleOut, SocDataMetadata
from .models import make_model
from .datasets import make_dataset
from .datasets import utils as ds_utils
from . import val

CollateFnType = Callable[[List[Any]], Any]


class Runner(pl.LightningModule):
    """
        A runner represent a training pipeline.
        It contains everything from the dataset to the optimizer.

        Args:
            - config: (SocConfig) Hyper paramete configuration
    """
    def __init__(self, config: SocConfig):
        super(Runner, self).__init__()
        self.hparams = config

        if self.hparams['loss_name'] == 'mse':
            self.loss_f = F.mse_loss
        elif self.hparams['loss_name'] == 'resnet18policy_loss':
            self.mse_loss_f = F.mse_loss
            self.ce_loss_f = F.cross_entropy
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
        self.metadata = dataset.get_output_metadata()
        self.collate_fn = dataset.get_collate_fn()
        self.hparams['data_input_size'] = dataset.get_input_size()
        self.hparams['data_output_size'] = dataset.get_output_size()

        self.model = make_model(self.hparams)

    def setup_dataset(self):
        """This function purpose is mainly to be overrided for tests"""
        dataset = make_dataset(self.hparams)

        return dataset

    def train_dataloader(self):
        # Ho my god! overfit_batches is broken
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/2311
        shuffle = self.hparams['shuffle_dataset'] if 'shuffle_dataset' in self.hparams else True

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
            train_dict = train_on_supervised_seq_batch(batch, self.model, self.loss_f)
        elif self.training_type == 'supervised_forward':
            train_dict = train_on_supervised_forward_batch(batch, self.model, self.loss_f)
        elif self.training_type == 'resnet18policy':
            train_dict = train_on_resnet18policy_batch(
                batch, self.model, self.mse_loss_f, self.ce_loss_f, self.metadata
            )
        else:
            raise Exception(
                "No training process exist for this training type: {}".format(self.training_type)
            )

        final_dict = {'loss': train_dict['train_loss'], 'log': train_dict}

        return final_dict

    def validation_step(self, batch, batch_idx):
        if self.training_type == 'supervised_seq':
            val_dict = val_on_supervised_seq_batch(batch, self.model, self.loss_f, self.metadata)
        elif self.training_type == 'supervised_forward':
            val_dict = val_on_supervised_forward_batch(
                batch, self.model, self.loss_f, self.metadata
            )
        elif self.training_type == 'resnet18policy':
            val_dict = val_on_resnet18policy_batch(
                batch, self.model, self.mse_loss_f, self.ce_loss_f, self.metadata
            )
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
                                  loss_f: Callable) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y, mask) batch of data
            - model: (Module) the model
            - loss_f: (Callable) the loss function to apply
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

    loss = loss_f(y_preds_reshaped, y_true_reshaped)

    train_dict = {'train_loss': loss}

    return train_dict


def val_on_supervised_seq_batch(
    batch: SocSeqBatch,
    model: Module,
    loss_f: Callable,
    metadata: SocDataMetadata = {},
) -> Dict[str, torch.Tensor]:
    """This function computes the validation loss and accuracy of the model."""

    x = batch[0]
    y_true = batch[1]
    mask = batch[2]

    # We assume the model outputs a tuple where the first element
    # is the actual predictions
    outputs = model(x)
    y_preds_raw = outputs[0]
    y_preds = y_preds_raw * mask
    # Actions and state are represented as ints
    # We can compute a simple decision boundary by using the round function
    y_preds_int = torch.round(y_preds)

    bs, seq_len, C, H, W = y_true.shape
    y_preds_reshaped = y_preds.reshape(bs * seq_len, C, W, H)
    y_preds_int_reshaped = y_preds_int.reshape(bs * seq_len, C, W, H)
    y_true_reshaped = y_true.reshape(bs * seq_len, C, W, H)

    loss = loss_f(y_preds_reshaped, y_true_reshaped)

    dtype = loss.dtype
    acc = (y_preds_int == y_true).type(dtype).mean()  # type: ignore
    val_dict = val.get_stats(metadata, y_preds_int_reshaped, y_true_reshaped)
    val_dict.update({
        'val_accuracy': acc,
    })

    return val_dict


def train_on_supervised_forward_batch(batch: SocBatch, model: Module,
                                      loss_f: Callable) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y) batch of data
            - model: (Module) the model
            - loss_f: (Callable) the loss function to apply
    """
    x = batch[0]
    y_true = batch[1]

    y_preds = model(x)

    loss = loss_f(y_preds, y_true)

    train_dict = {'train_loss': loss}

    return train_dict


def val_on_supervised_forward_batch(
    batch: SocBatch,
    model: Module,
    loss_f: Callable,
    metadata: SocDataMetadata = {},
) -> Dict[str, torch.Tensor]:
    """This function computes the validation loss and accuracy of the model."""

    x = batch[0]
    y_true = batch[1]

    y_preds = model(x)
    # Actions and state are represented as ints
    # We can compute a simple decision boundary by using the round function
    y_preds_int = torch.round(y_preds)

    loss = loss_f(y_preds, y_true)

    dtype = loss.dtype
    acc = (y_preds_int == y_true).type(dtype).mean()  # type: ignore
    val_dict = val.get_stats(metadata, y_preds_int, y_true)

    one_meta = {'pieces_one_mean': metadata['pieces']}
    if 'actions' in metadata.keys():
        one_meta['actions_one_mean'] = metadata['actions']
    val_dict.update(val.get_stats(one_meta, y_preds_int, 1))

    val_dict.update({
        'val_accuracy': acc,
    })

    return val_dict


def train_on_resnet18policy_batch(
    batch: SocBatchMultipleOut,
    model: Module,
    state_loss_f: Callable,
    action_loss_f: Callable,
    metadata: SocDataMetadata = {},
) -> Dict[str, torch.Tensor]:
    """
        This function apply an batch update to the model.

        Args:
            - batch: (x, y) batch of data
            - model: (Module) the model
            - state_loss_f: (Callable) the loss function for states
            - ation_loss_f: (Callable) the loss function for states
    """
    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = model(x_seq)

    bs, S_h, C, H, W = x_seq.shape
    _, S_f, C_s = y_s_true_seq.shape
    _, _, C_a = y_a_true_seq.shape
    y_s_logits = y_s_logits_seq.view(-1, C_s)
    y_s_true = y_s_true_seq.view(-1, C_s)
    y_a_logits = y_a_logits_seq.view(-1, C_a)
    y_a_true = torch.argmax(y_a_true_seq.view(-1, C_a), dim=1)

    start_i, end_i = metadata['map']
    spatial_state_loss = state_loss_f(
        y_spatial_s_logits_seq[:, :, start_i:end_i], y_spatial_s_true_seq[:, :, start_i:end_i]
    )
    start_i, end_i = metadata['robber']
    spatial_state_loss += F.cross_entropy(
        y_spatial_s_logits_seq[:, :, start_i].reshape(bs * S_f, -1),
        torch.argmax(y_spatial_s_true_seq[:, :, start_i].reshape(bs * S_f, -1), dim=1)
    )
    start_i, end_i = metadata['pieces']
    y_spatial_pieces_true = y_spatial_s_true_seq[:, :, start_i:end_i]
    # # We compute a mask to subsample zero values
    # loss_mask = torch.zeros_like(
    #     y_spatial_pieces_true, device=y_spatial_s_true_seq.device, requires_grad=False
    # )
    # loss_mask[y_spatial_pieces_true > 0] = 1
    # random_c = torch.randperm(loss_mask.shape[2])[:2]
    # loss_mask[:, :, random_c] = 1
    spatial_state_loss += F.binary_cross_entropy_with_logits(
        y_spatial_s_logits_seq[:, :, start_i:end_i], y_spatial_pieces_true
    )
    state_loss = state_loss_f(y_s_logits, y_s_true)
    action_loss = action_loss_f(y_a_logits, y_a_true)

    loss = spatial_state_loss + state_loss + action_loss

    train_dict = {
        'train_loss': loss,
        'train_spatial_state_loss': spatial_state_loss,
        'train_state_loss': state_loss,
        'train_action_loss': action_loss,
    }

    return train_dict


def val_on_resnet18policy_batch(
    batch: SocBatchMultipleOut,
    model: Module,
    state_loss_f: Callable,
    action_loss_f: Callable,
    metadata: SocDataMetadata = {},
) -> Dict[str, torch.Tensor]:
    """This function computes the validation loss and accuracy of the model."""

    x_seq = batch[0]
    y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]

    y_spatial_s_logits_seq, y_s_logits_seq, y_a_logits_seq = model(x_seq)

    bs, S_h, C, H, W = x_seq.shape
    _, _, C_ss, _, _ = y_spatial_s_true_seq.shape
    _, S_f, C_s = y_s_true_seq.shape
    _, _, C_a = y_a_true_seq.shape
    dtype = x_seq.dtype
    y_s_logits = y_s_logits_seq.view(-1, C_s)
    y_s_true = y_s_true_seq.view(-1, C_s)
    y_a_logits = y_a_logits_seq.view(-1, C_a)
    y_a_true = torch.argmax(y_a_true_seq.view(-1, C_a), dim=1)

    # Actions and state are represented as ints
    # We can compute a simple decision boundary by using the round function
    y_spatial_s_preds_seq = torch.zeros_like(y_spatial_s_logits_seq)
    start_i, end_i = metadata['map']
    y_spatial_s_preds_seq[:, :, start_i:start_i + 1] = ds_utils.unnormalize_hexlayout(
        y_spatial_s_logits_seq[:, :, start_i:start_i + 1]
    )
    y_spatial_s_true_seq[:, :, start_i:start_i + 1] = ds_utils.unnormalize_hexlayout(
        y_spatial_s_true_seq[:, :, start_i:start_i + 1]
    )
    y_spatial_s_preds_seq[:, :, start_i + 1:end_i] = ds_utils.unnormalize_numberlayout(
        y_spatial_s_logits_seq[:, :, start_i + 1:end_i]
    )
    y_spatial_s_true_seq[:, :, start_i + 1:end_i] = ds_utils.unnormalize_numberlayout(
        y_spatial_s_true_seq[:, :, start_i + 1:end_i]
    )

    start_i, _ = metadata['robber']
    rob_pred_idx = torch.argmax(y_spatial_s_logits_seq[:, :, start_i].view(bs * S_f, -1), dim=1)
    line_select = torch.arange(bs * S_f, dtype=torch.long)
    tmp = y_spatial_s_preds_seq.view(bs * S_f, C_ss, -1)
    tmp[line_select, start_i, rob_pred_idx] = 1
    y_spatial_s_preds_seq = tmp.view(y_spatial_s_preds_seq.shape)

    start_i, end_i = metadata['pieces']
    # Pieces corresponds to a binary predictions
    y_spatial_s_preds_seq[:, :, start_i:end_i] = torch.round(
        torch.sigmoid(y_spatial_s_logits_seq[:, :, start_i:end_i])
    )
    y_s_preds_int = torch.round(y_s_logits)
    action_preds_int = torch.argmax(y_a_logits, dim=-1)

    y_s_spatial_eq = (y_spatial_s_preds_seq == y_spatial_s_true_seq)
    spatial_state_acc = y_s_spatial_eq.type(dtype).mean()  # type: ignore
    state_acc = (y_s_preds_int == y_s_true).type(dtype).mean()  # type: ignore
    action_acc = (action_preds_int == y_a_true).type(dtype).mean()  # type: ignore
    acc = (spatial_state_acc + state_acc + action_acc) / 3

    val_dict = val.get_stats(metadata, y_spatial_s_preds_seq, y_spatial_s_true_seq)
    one_meta = {'pieces_one_mean': metadata['pieces']}
    val_dict.update(val.get_stats(one_meta, y_spatial_s_preds_seq, 1))

    val_dict.update({
        'val_accuracy': acc,
        'val_spatial_state_accuracy': spatial_state_acc,
        'val_state_accuracy': state_acc,
        'val_action_accuracy': action_acc,
    })

    return val_dict


def train(config: SocConfig) -> Runner:
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
        mode='min',
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
