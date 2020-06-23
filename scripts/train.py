import os
import torch
import argparse
import random
import time
from tqdm import tqdm
import soc
from soc.training import train_on_dataset
from soc.models import make_model, get_models_list
from soc import utils

cfd = os.path.dirname(os.path.realpath(__file__))
default_results_d = os.path.join(cfd, 'results', str(int(time.time() * 1000000)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def end_batch_callback(i_epoch: int, i_batch: int, n_batchs: int, loss: float):
    if i_batch % (n_batchs // 4) == 0:
        tqdm.write(
            "Epoch: {}, {}%, loss: {}".format(i_epoch, round(i_batch / n_batchs * 100), loss)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')

    # Generic HP
    parser.add_argument('--config', '-c', type=str, help='A configuration file')
    parser.add_argument('--seed', '-s', type=int, default=random.randint(0, 100), help='The seed')
    parser.add_argument(
        '--results_d', type=str, default=default_results_d, help='The result folder'
    )
    parser.add_argument(
        '--load',
        type=bool,
        default=False,
        help='Should we try to load a config from the result folder'
    )

    # Data HP
    parser.add_argument(
        '--dataset_name', '--dn', type=str, default='socpsqlseq', help='The dataset name'
    )

    # Model HP
    parser.add_argument(
        '--arch', choices=get_models_list(), default='ConvLSTM', help='The architecture name'
    )
    parser.add_argument(
        '--in_chan_dim', type=list, default=245 + 17, help='Number of input channels'
    )
    parser.add_argument(
        '--h_chan_dim', type=list, default=[300, 245], help='List of hidden channels per layer'
    )
    parser.add_argument(
        '--kernel_size', type=list, default=[(3, 3), (3, 3)], help='List of Kernel size per layer'
    )
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')

    # Training HP
    parser.add_argument('--loss_name', '--ln', type=str, default='mse', help='The loss name')
    parser.add_argument(
        '--lr',
        type=float,
        default=-1,
        help='the learning rate (the default depend on the optimizer)',
    )
    parser.add_argument(
        '--optimizer',
        '--opt',
        type=str,
        default='adam',
        help='Optimizer name', )
    parser.add_argument(
        '--scheduler',
        '--sch',
        type=str,
        default='',
        help='Scheduler name', )
    parser.add_argument(
        '--n_epochs',
        '--ne',
        type=int,
        default=100,
        help='Number of epochs', )
    parser.add_argument(
        '--batch_size',
        '--bs',
        type=int,
        default=4,
        help='Batch size', )

    config = vars(parser.parse_args())

    if config['config'] is not None:
        if os.path.isfile(config['config']):
            config = torch.load(config['config'])

    # Generics
    utils.set_seed(config['seed'])
    utils.check_folder(config['results_d'])
    should_load = config['load']
    if should_load:
        config, checkpoints = utils.load(config['results_d'])

    # Data
    if config['dataset_name'] == 'socpsqlseq':
        dataset = soc.SocPSQLSeqDataset()
    else:
        raise Exception('Unknown dataset {}'.format(config['dataset_name']))
    config['input_size'] = dataset.get_input_size()
    collate_fn = dataset.get_collate_fn()

    # Model
    model = make_model(config)
    if should_load and 'model' in checkpoints.keys():
        model.load_state_dict(checkpoints['model'])
    model.to(device)

    # Training
    if config['loss_name'] == 'mse':
        loss_f = torch.nn.MSELoss()
    else:
        raise Exception('Unknown loss function {}'.format(config['loss_name']))

    if config['lr'] == -1:
        lr = 1e-3
    else:
        lr = config['lr']

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise Exception('Unknown optimizer {}'.format(config['optimizer']))
    if should_load and 'optimizer' in checkpoints.keys():
        optimizer.load_state_dict(checkpoints['optimizer'])

    scheduler = None
    if config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001
        )
    if scheduler is not None and should_load and 'scheduler' in checkpoints.keys():
        scheduler.load_state_dict(checkpoints['scheduler'])

    callbacks = {'end_batch_callback': end_batch_callback}

    final_model = train_on_dataset(
        config, dataset, model, loss_f, optimizer, collate_fn=collate_fn, callbacks=callbacks
    )

    utils.save(config, final_model, optimizer)
