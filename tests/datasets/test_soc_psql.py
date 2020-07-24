import os
import unittest
import pandas as pd
from unittest.mock import MagicMock
from soc import datasets
from soc.datasets import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


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

    def test_soc_psql_seq_dataset(self):
        dataset = datasets.SocPSQLSeqDataset({'no_db': True})

        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)

        seqs = dataset[0]
        assert len(seqs) == 2
        assert len(seqs[0]) == 297
        assert seqs[0][0].shape == (soc_data.STATE_SIZE, 7, 7)
        assert seqs[1][0].shape == (soc_data.ACTION_SIZE, 7, 7)

        seqs = dataset[1]
        assert len(seqs[0]) == 270
        assert seqs[0][0].shape == (soc_data.STATE_SIZE, 7, 7)
        assert seqs[1][0].shape == (soc_data.ACTION_SIZE, 7, 7)
