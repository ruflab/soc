import math
import torch
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Any, Dict
from .typing import SocSeqBatch, SocBatch

CollateFnType = Callable[[List[Any]], Any]


def train_on_dataset(
        config: Dict,
        dataset: Dataset,
        model: Module,
        loss_f: Callable,
        optimizer: Optimizer,
        scheduler: object = None,
        collate_fn: CollateFnType = None,
        training_type: str = 'supervised',
        callbacks: dict = {},
) -> Module:
    if config['verbose']:
        print('Launching training')

    batch_size = config['batch_size']
    n_epochs = config['n_epochs']

    if collate_fn is None:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True
        )

    dataset_size = len(dataset)
    n_batchs = math.floor(dataset_size / batch_size)

    if 'start_training_callback' in callbacks:
        callbacks['start_training_callback']()

    for i_epoch in tqdm(range(n_epochs)):
        if 'start_epoch_callback' in callbacks:
            callbacks['start_epoch_callback'](i_epoch)

        for i_batch, batch in enumerate(dataloader):

            if 'start_batch_callback' in callbacks:
                callbacks['start_batch_callback'](i_epoch, i_batch, n_batchs)

            if training_type == 'supervised_seq':
                loss = train_on_supervised_seq_batch(batch, model, loss_f, optimizer)
            elif training_type == 'supervised_forward':
                loss = train_on_supervised_forward_batch(batch, model, loss_f, optimizer)
            else:
                raise Exception(
                    "No training process exist for this training type: {}".format(training_type)
                )

            if 'end_batch_callback' in callbacks:
                callbacks['end_batch_callback'](i_epoch, i_batch, n_batchs, loss)

        # Validation and scheduler step should appear here
        if 'end_epoch_callback' in callbacks:
            callbacks['end_epoch_callback'](i_epoch)

    if 'end_training_callback' in callbacks:
        callbacks['end_training_callback']()

    return model


def train_on_supervised_seq_batch(
        batch: SocSeqBatch, model: Module, loss_f: Callable, optimizer: Optimizer, scheduler=None
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

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_on_supervised_forward_batch(
        batch: SocBatch, model: Module, loss_f: Callable, optimizer: Optimizer, scheduler=None
) -> torch.Tensor:
    """
        This function apply an batch update to the model.
    """
    x = batch[0]
    y_true = batch[1]

    y_preds = model(x)

    loss = loss_f(y_preds, y_true)

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
