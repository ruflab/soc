import os
import re
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import random
import numpy as np
from typing import Tuple, Dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif not os.path.isdir(folder):
        raise Exception('The path provided ({}) exist and is not a folder, aborting'.format(folder))


def get_latest_checkpoint(folder: str) -> str:
    files = sorted([f for f in os.listdir(folder) if re.match('ckpt_', f)])
    if len(files) > 0:
        return os.path.join(folder, files[-1])

    return ''


def save(config: Dict, model: Module, optim: Optimizer, scheduler=None):
    folder = config['results_d']
    check_folder(folder)

    config_file = os.path.join(folder, 'config.pt')
    torch.save(config, config_file)

    last_ckpt_file = get_latest_checkpoint(folder)
    if last_ckpt_file == '':
        current_iter = 0
    else:
        current_iter = int(last_ckpt_file[-8:-3]) + 1

    ckpt = {'model': model.state_dict(), 'optimizer': optim.state_dict(), 'scheduler': None}
    if scheduler is not None:
        ckpt['scheduler'] = scheduler.state_dict()

    ckpt_file = os.path.join(folder, 'ckpt_{0:05d}.pt'.format(current_iter))
    torch.save(ckpt, ckpt_file)


def load(folder: str) -> Tuple:
    config_file = os.path.join(folder, 'config.pt')
    last_checkpoints_file = get_latest_checkpoint(folder)

    config = torch.load(config_file)
    ckpts = torch.load(last_checkpoints_file)

    return config, ckpts


def pad_collate_fn(inputs) -> Tuple:
    """
        Pad the different inputs

        inputs is a list of (state_seq, actions_seq)
    """
    batch_states_seq = []
    batch_actions_seq = []
    for t in inputs:
        states_seq, actions_seq = t

        batch_states_seq.append(torch.tensor(states_seq))
        batch_actions_seq.append(torch.tensor(actions_seq))

    batch_states_seq_t = torch.nn.utils.rnn.pad_sequence(batch_states_seq, batch_first=True)
    batch_actions_seq_t = torch.nn.utils.rnn.pad_sequence(batch_actions_seq, batch_first=True)

    return batch_states_seq_t, batch_actions_seq_t
