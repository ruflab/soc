import math
import torch
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Any, Tuple, Dict

CollateFnType = Callable[[List[Any]], Any]


def train_on_dataset(
        config: Dict,
        dataset: Dataset,
        model: Module,
        loss_f: Callable,
        optimizer: Optimizer,
        scheduler: object = None,
        collate_fn: CollateFnType = None,
        callbacks: dict = {}
) -> Module:
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

            loss = train_on_batch(batch, model, loss_f, optimizer)

            if 'end_batch_callback' in callbacks:
                callbacks['end_batch_callback'](i_epoch, i_batch, n_batchs, loss)

        if 'end_epoch_callback' in callbacks:
            callbacks['end_epoch_callback'](i_epoch)

    if 'end_training_callback' in callbacks:
        callbacks['end_training_callback']()

    return model


def train_on_batch(
        batch: Tuple, model: Module, loss_f: Callable, optimizer: Optimizer
) -> torch.Tensor:
    batched_states_seq = batch[0]
    batched_actions_seq = batch[1]

    # TODO: make it real
    final_inputs = torch.cat([
        batched_states_seq[0],
        batched_actions_seq[0], ], dim=-1)
    true_preds = batched_states_seq[1]

    preds, _ = model(final_inputs)
    preds = preds[:, :-1, :]

    n_preds_features = preds.shape[-1]

    loss = loss_f(preds.reshape(-1, n_preds_features), true_preds.reshape(-1, n_preds_features))

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
