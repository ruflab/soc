import os
import torch
import argparse
import random
import time
from tqdm import tqdm
from soc import utils
from soc.training import train_on_dataset, instantiate_training_params
from soc.models import get_models_list
from soc.datasets import get_datasets_list

cfd = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def end_batch_callback(i_epoch: int, i_batch: int, n_batchs: int, loss: float):
    if i_batch % max(n_batchs // 100, 1) == 0:
        tqdm.write(
            "Epoch: {}, {}%, loss: {}".format(i_epoch, round(i_batch / n_batchs * 100), loss)
        )


if __name__ == "__main__":
    default_results_d = os.path.join(cfd, 'results', str(int(time.time() * 1000000)))
    default_config = {
        # Generics
        'config': None,
        'results_d': default_results_d,
        'load': False,
        'seed': random.randint(0, 100),
        'verbose': True,
        # Data
        'dataset': 'SocPreprocessedSeqSAToSDataset',
        'history_length': 8,
        'future_length': 1,
        'no_db': False,  # Used for testing
        # Model
        'model': 'ConvLSTM',
        'defaul_model': False,
        'h_chan_dim': 32,
        'kernel_size': (3, 3),
        'strides': (3, 3),
        'paddings': (1, 1),
        'num_layers': 2,
        'loss_name': 'mse',
        'lr': 1e-3,
        'optimizer': 'adam',
        'scheduler': '',
        'n_epochs': 100,
        'batch_size': 4
    }

    parser = argparse.ArgumentParser(description='Training configuration')

    # Generic HP
    parser.add_argument(
        '--config', '-c', type=str, default=argparse.SUPPRESS, help='A configuration file'
    )
    parser.add_argument('--seed', '-s', type=int, default=argparse.SUPPRESS, help='The seed')
    parser.add_argument(
        '--results_d', type=str, default=argparse.SUPPRESS, help='The result folder'
    )
    parser.add_argument(
        '--load',
        type=bool,
        default=argparse.SUPPRESS,
        help='Should we try to load a config from the result folder'
    )
    parser.add_argument(
        '--verbose', type=bool, default=argparse.SUPPRESS, help='Should we print many things'
    )

    # Data HP
    parser.add_argument(
        '--dataset',
        '-d',
        choices=get_datasets_list(),
        default=argparse.SUPPRESS,
        help='The dataset name'
    )
    parser.add_argument(
        '--history_length',
        type=int,
        default=argparse.SUPPRESS,
        help='History length for the feedforward training pipeline'
    )
    parser.add_argument(
        '--future_length',
        type=int,
        default=argparse.SUPPRESS,
        help='Future length for the feedforward training pipeline'
    )

    # Model HP
    parser.add_argument(
        '--model',
        choices=get_models_list(),
        default=argparse.SUPPRESS,
        help='The architecture name'
    )
    parser.add_argument(
        '--defaul_model',
        type=bool,
        default=argparse.SUPPRESS,
        help='Load the default parameters of the model'
    )
    parser.add_argument(
        '--h_chan_dim',
        type=int,
        nargs='+',
        default=argparse.SUPPRESS,
        help='List of hidden channels per layer'
    )
    parser.add_argument(
        '--kernel_size',
        type=utils.soc_tuple,
        nargs='+',
        default=argparse.SUPPRESS,
        help='List of Kernel size per layer'
    )
    parser.add_argument(
        '--strides',
        type=utils.soc_tuple,
        nargs='+',
        default=argparse.SUPPRESS,
        help='List of Kernel size per layer'
    )
    parser.add_argument(
        '--paddings',
        type=utils.soc_tuple,
        nargs='+',
        default=argparse.SUPPRESS,
        help='List of Kernel size per layer'
    )
    parser.add_argument(
        '--num_layers', type=int, default=argparse.SUPPRESS, help='Number of layers'
    )

    # Training HP
    parser.add_argument(
        '--loss_name', '--ln', type=str, default=argparse.SUPPRESS, help='The loss name'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=argparse.SUPPRESS,
        help='the learning rate (the default depend on the optimizer)',
    )
    parser.add_argument(
        '--optimizer',
        '--opt',
        type=str,
        default=argparse.SUPPRESS,
        help='Optimizer name', )
    parser.add_argument(
        '--scheduler',
        '--sch',
        type=str,
        default=argparse.SUPPRESS,
        help='Scheduler name', )
    parser.add_argument(
        '--n_epochs',
        '--ne',
        type=int,
        default=argparse.SUPPRESS,
        help='Number of epochs', )
    parser.add_argument(
        '--batch_size',
        '--bs',
        type=int,
        default=argparse.SUPPRESS,
        help='Batch size', )

    ###
    #  Loading configuration
    ###
    cli_config = vars(parser.parse_args())
    config = utils.build_config(default_config, cli_config)

    ###
    # Configuring the app
    ###
    # Generics
    utils.set_seed(config['seed'])

    training_params = instantiate_training_params(config)
    training_params['callbacks'] = {'end_batch_callback': end_batch_callback}

    training_params['model'].to(device)
    final_model = train_on_dataset(**training_params)

    utils.save(config, final_model, training_params['optimizer'])
