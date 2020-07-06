import random
import argparse
import json
from pytorch_lightning import Trainer
from soc import datasets, models, training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training configuration')

    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default=None,
        help='A config to load (override all cli parameters)'
    )
    parser.add_argument('--seed', '-s', type=int, default=random.randint(0, 100), help='The seed')
    parser.add_argument('--verbose', type=bool, default=False, help='Should we print many things')
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
    if temp_config['config'] is not None:
        with open(temp_config['config'], 'r') as f:
            config = json.load(f)
    else:
        model_class = models.get_model_class(temp_config)
        parser = model_class.add_argparse_args(parser)

        dataset_class = datasets.get_dataset_class(temp_config)
        parser = dataset_class.add_argparse_args(parser)

        # To resume training, use resume_from_checkpoint arg
        trainer_parser = Trainer.add_argparse_args(
            argparse.ArgumentParser(description='Training configuration')
        )
        config = {
            'generic': vars(parser.parse_known_args()[0]),
            'trainer': vars(trainer_parser.parse_known_args()[0])
        }

    training.train(config)
