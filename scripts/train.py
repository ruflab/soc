import os
import random
import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from soc.training import Runner
from soc import datasets, models

cfd = os.path.dirname(os.path.realpath(__file__))


def main(config):
    runner = Runner(config['generic'])

    seed_everything(config['generic']['seed'])
    config['trainer']['deterministic'] = True
    # config['trainer'][' distributed_backend'] = 'dp'

    if config['trainer']['default_root_dir'] is None:
        config['trainer']['default_root_dir'] = os.path.join(cfd, 'results')
    checkpoint_callback = ModelCheckpoint(
        filepath=config['trainer']['default_root_dir'],
        save_top_k=0,
        verbose=True,
        monitor='train_loss',
        mode='min',
        prefix=''
    )
    config['trainer']['checkpoint_callback'] = checkpoint_callback

    trainer = Trainer(**config['trainer'])
    trainer.fit(runner)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')

    parser.add_argument('--seed', '-s', type=int, default=random.randint(0, 100), help='The seed')
    parser.add_argument(
        '--verbose', type=bool, default=argparse.SUPPRESS, help='Should we print many things'
    )
    parser.add_argument(
        '--dataset',
        '-d',
        choices=datasets.get_datasets_list(),
        default='SocPreprocessedSeqSAToSDataset',
        help='The dataset name'
    )
    parser.add_argument(
        '--model', choices=models.get_models_list(), default='ConvLSTM', help='The model name'
    )
    parser.add_argument('--loss_name', '--ln', type=str, default='mse', help='The loss name')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
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
        default=None,
        help='Scheduler name', )
    parser.add_argument(
        '--batch_size',
        '--bs',
        type=int,
        default=8,
        help='Batch size', )

    temp_args, _ = parser.parse_known_args()
    temp_config = vars(temp_args)

    model_class = models.get_model_class(temp_config)
    parser = model_class.add_argparse_args(parser)

    dataset_class = datasets.get_dataset_class(temp_config)
    parser = dataset_class.add_argparse_args(parser)

    # Ro resume training, use resume_from_checkpoint arg
    trainer_parser = Trainer.add_argparse_args(
        argparse.ArgumentParser(description='Training configuration')
    )
    config = {
        'generic': vars(parser.parse_known_args()[0]),
        'trainer': vars(trainer_parser.parse_known_args()[0])
    }

    main(config)
