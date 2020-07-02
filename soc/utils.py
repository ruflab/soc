import argparse
import os
import re
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import random
import numpy as np
from typing import Tuple, Dict
from .models import get_model_class


def soc_tuple(s: str):
    try:
        t = tuple(map(int, s.split(',')))
        return t
    except Exception:
        raise argparse.ArgumentTypeError("Coordinates must be x,y,z")


def build_config(default_config: Dict, cli_config: Dict) -> Dict:
    """
        Generate a config based on the following rules:
            1. it loads the default configuration
            2. if provided, it loads a configuration file to replace the default
            3. Any parameters directly provided in the cli override values
            4. if a result folder is specified which already contains a configuration file and the
            user ask to restart the training with the load option. Then the whole configuration
            is actually loaded from that path
    """
    config = default_config

    if 'config' in cli_config.keys():
        if os.path.isfile(cli_config['config']):
            loaded_config = torch.load(cli_config['config'])

            config.update(loaded_config)

    if 'default_model' in config and config['default_model'] is True:
        default_model_config = get_model_class(config['model']).get_default_conf()
        config.update(default_model_config)

    config.update(cli_config)

    check_folder(config['results_d'])
    if 'load' in config and config['load'] is True:
        config, checkpoints = load(config['results_d'])
        config['checkpoints'] = checkpoints

    return config


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
