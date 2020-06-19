import math
import torch
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Any, Tuple

CollateFnType = Callable[[List[Any]], Any]


def train_on_dataset(
        dataset: Dataset,
        model: Module,
        loss_f: Callable,
        optimizer: Optimizer,
        n_epochs: int = 100,
        batch_size: int = 32,
        collate_fn: CollateFnType = None,
        callbacks: dict = {}
) -> Module:
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

    for i_epoch in tqdm(range(n_epochs)):
        for i_batch, batch in enumerate(dataloader):

            loss = train_on_batch(batch, model, loss_f, optimizer)

            if 'end_batch_callback' in callbacks:
                callbacks['end_batch_callback'](i_epoch, i_batch, n_batchs, loss)

    return model


def train_on_batch(
        batch: Tuple, model: Module, loss_f: Callable, optimizer: Optimizer
) -> torch.Tensor:
    batched_states_seq = batch[0]
    batched_actions_seq = batch[1]

    # TODO: make it real
    final_inputs = torch.cat([
        batched_states_seq[0],
        batched_actions_seq[0],
    ], dim=-1)
    true_preds = batched_states_seq[1]

    preds, _ = model(final_inputs)
    preds = preds[:, :-1, :]

    n_preds_features = preds.shape[-1]

    loss = loss_f(preds.reshape(-1, n_preds_features), true_preds.reshape(-1, n_preds_features))

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
