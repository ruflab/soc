import os
import unittest
import pandas as pd
from torch.utils.data import DataLoader
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

    def setUp(self):
        states = [pd.read_csv(file) for file in self.obs_files]
        actions = [pd.read_csv(file) for file in self.actions_files]

        def _get_states_from_db_se_f(idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(idx: int) -> pd.DataFrame:
            return actions[idx]

        self._get_states_from_db_se_f = _get_states_from_db_se_f
        self._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_loading_pipeline(self):
        dataset = SocPSQLDataset(no_db=True)

        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_length = MagicMock(return_value=2)

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=utils.pad_collate_fn)

        x = next(iter(dataloader))

        assert len(x) == 2
        assert x[0].shape == (2, 297, 7, 7, 245)
        assert x[1].shape == (2, 297, 7, 7, 17)
