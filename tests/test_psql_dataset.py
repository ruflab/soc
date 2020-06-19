import os
import unittest
import pandas as pd
from unittest.mock import MagicMock
from soc import SocPSQLDataset

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestSocPSQLDataset(unittest.TestCase):

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

    def test_soc_psql_dataset(self):
        dataset = SocPSQLDataset(no_db=True)

        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)

        seqs = dataset[0]
        assert len(seqs) == 2
        assert len(seqs[0]) == 297
        assert seqs[0][0].shape == (7, 7, 245)

        seqs = dataset[1]
        assert len(seqs[0]) == 270
        assert seqs[0][0].shape == (7, 7, 245)
