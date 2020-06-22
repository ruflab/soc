import os
import shutil
import unittest
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
import numpy as np
from unittest.mock import MagicMock
from soc import SocPSQLDataset
from soc import utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):

    df_states: pd.DataFrame
    df_actions: pd.DataFrame

    obs_files = [
        os.path.join(fixture_dir, 'obsgamestates_100.csv'),
        os.path.join(fixture_dir, 'obsgamestates_101.csv'),
    ]
    actions_files = [
        os.path.join(fixture_dir, 'gameactions_100.csv'),
        os.path.join(fixture_dir, 'gameactions_101.csv'),
    ]

    _get_states_from_db_se_f = None
    _get_actions_from_db_se_f = None

    @classmethod
    def setUpClass(cls):
        states = [pd.read_csv(file) for file in cls.obs_files]
        actions = [pd.read_csv(file) for file in cls.actions_files]

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return actions[idx]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(fixture_dir, 'test_save_load'))

    def test_data_loading_pipeline(self):
        dataset = SocPSQLDataset(no_db=True)

        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_length = MagicMock(return_value=2)

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=utils.pad_collate_fn)

        x = next(iter(dataloader))

        assert len(x) == 2
        assert x[0].shape == (2, 297, 245, 7, 7)
        assert x[1].shape == (2, 297, 17, 7, 7)

    def test_save_load(self):
        tmp_folder = os.path.join(fixture_dir, 'test_save_load', str(int(time.time() * 1000000)))
        config = {'a': 2, 'b': 'test', 'results_d': tmp_folder}
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        optim = torch.optim.SGD(model.parameters(), 1.)
        for i in range(3):
            out = model(torch.tensor([1., 1.]))
            out.backward()
            optim.step()
            utils.save(config, model, optim)

        utils.save(config, model, optim)
        time.sleep(0.1)  # Make sure we take the time to save
        config2, checkpoints = utils.load(tmp_folder)

        model2 = torch.nn.Sequential(torch.nn.Linear(2, 1))
        model2.load_state_dict(checkpoints['model'])

        assert config2['a'] == 2
        assert config2['b'] == 'test'
        np.testing.assert_array_equal(
            model2.state_dict()['0.weight'], model.state_dict()['0.weight']
        )
        np.testing.assert_array_equal(model2.state_dict()['0.bias'], model.state_dict()['0.bias'])
