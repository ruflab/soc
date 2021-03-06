import argparse
import os
import torch
from typing import Dict
from .models import get_model_class


def soc_tuple(s: str) -> tuple:
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

    # check_folder(config['results_d'])
    # if 'load' in config and config['load'] is True:
    #     config, checkpoints = load(config['results_d'])
    #     config['checkpoints'] = checkpoints

    return config


def check_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif not os.path.isdir(folder):
        raise Exception('The path provided ({}) exist and is not a folder, aborting'.format(folder))
