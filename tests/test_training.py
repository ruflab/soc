import os
import time
# import shutil
import unittest
import pandas as pd
from unittest.mock import MagicMock
from soc import utils
from soc.training import train_on_dataset

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):

    df_states: pd.DataFrame
    df_actions: pd.DataFrame

    obs_files = [
        os.path.join(fixture_dir, 'small_obsgamestates_100.csv'),
        os.path.join(fixture_dir, 'small_obsgamestates_101.csv'),
    ]
    actions_files = [
        os.path.join(fixture_dir, 'small_gameactions_100.csv'),
        os.path.join(fixture_dir, 'small_gameactions_101.csv'),
    ]

    _get_states_from_db_se_f = None
    _get_actions_from_db_se_f = None

    @classmethod
    def setUpClass(cls):
        cls.states = [pd.read_csv(file) for file in cls.obs_files]
        cls.actions = [pd.read_csv(file) for file in cls.actions_files]

    def setUp(self):
        self.folder = os.path.join(fixture_dir, str(int(time.time() * 1000000)))

    def test_training_socseqsas_convlstm(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return actions[idx]

        config = {
            # Generics
            'config': None,
            'results_d': self.folder,
            'load': False,
            'seed': 1,
            'verbose': False,
            # Data
            'dataset': 'SocPSQLSeqSAToSDataset',
            'no_db': True,
            # Model
            'arch': 'ConvLSTM',
            'defaul_arch': False,
            'h_chan_dim': [150, 150],
            'kernel_size': [(3, 3), (3, 3)],
            'strides': [(3, 3), (3, 3)],
            'num_layers': 2,
            'loss_name': 'mse',
            'lr': 1e-3,
            'optimizer': 'adam',
            'scheduler': '',
            'n_epochs': 1,
            'batch_size': 2
        }

        utils.set_seed(config['seed'])

        training_params = utils.instantiate_training_params(config)

        training_params['dataset']._get_states_from_db = MagicMock(
            side_effect=_get_states_from_db_se_f
        )
        training_params['dataset']._get_actions_from_db = MagicMock(
            side_effect=_get_actions_from_db_se_f
        )
        training_params['dataset']._get_length = MagicMock(return_value=2)

        train_on_dataset(**training_params)

    def test_training_socseqsas_conv3dmodel(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return actions[idx]

        config = {
            # Generics
            'config': None,
            'results_d': self.folder,
            'load': False,
            'seed': 1,
            'verbose': False,
            # Data
            'dataset': 'SocPSQLSeqSAToSDataset',
            'no_db': True,
            # Model
            'arch': 'Conv3dModel',
            'defaul_arch': False,
            'h_chan_dim': 64,
            'kernel_size': (3, 3, 3),
            'strides': (1, 1, 1),
            'paddings': (1, 1, 1, 1, 2, 0),
            'num_layers': 2,
            'loss_name': 'mse',
            'lr': 1e-3,
            'optimizer': 'adam',
            'scheduler': '',
            'n_epochs': 1,
            'batch_size': 2
        }

        utils.set_seed(config['seed'])

        training_params = utils.instantiate_training_params(config)
        training_params['dataset']._get_states_from_db = MagicMock(
            side_effect=_get_states_from_db_se_f
        )
        training_params['dataset']._get_actions_from_db = MagicMock(
            side_effect=_get_actions_from_db_se_f
        )
        training_params['dataset']._get_length = MagicMock(return_value=2)

        train_on_dataset(**training_params)

    def test_training_socforward_resnet(self):
        states = self.states
        actions = self.actions

        def _get_states_from_db_se_f(
                table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
                table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        config = {
            # Generics
            'config': None,
            'results_d': self.folder,
            'load': False,
            'seed': 1,
            'verbose': False,
            # Data
            'dataset': 'SocPSQLForwardDataset',
            'history_length': 2,
            'future_length': 1,
            'no_db': True,
            'first_index': 0,
            # Model
            'arch': 'resnet18',
            'defaul_arch': False,
            'h_chan_dim': 64,
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'paddings': (1, 1),
            'num_layers': 2,
            'loss_name': 'mse',
            'lr': 1e-3,
            'optimizer': 'adam',
            'scheduler': '',
            'n_epochs': 1,
            'batch_size': 2
        }

        utils.set_seed(config['seed'])

        training_params = utils.instantiate_training_params(config)
        training_params['dataset']._get_states_from_db = MagicMock(
            side_effect=_get_states_from_db_se_f
        )
        training_params['dataset']._get_actions_from_db = MagicMock(
            side_effect=_get_actions_from_db_se_f
        )
        training_params['dataset']._get_length = MagicMock(return_value=2)
        training_params['dataset']._get_nb_steps = MagicMock(return_value=[9, 9])

        train_on_dataset(**training_params)
