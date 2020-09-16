import os
import unittest
import torch
import pandas as pd
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from unittest.mock import MagicMock
from soc import datasets
from soc.datasets import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestSocPSQLDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLConfig)

        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_soc_psql_seq_dataset(self):
        with initialize():
            config = compose(config_name="config", overrides=["no_db=true", "psql_password=dummy"])
            dataset = datasets.SocPSQLSeqDataset(config)

            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)

            seqs = dataset[0]
            assert len(seqs) == 2
            assert len(seqs[0]) == 30
            assert seqs[0][0].shape == (soc_data.STATE_SIZE, 7, 7)
            assert seqs[1][0].shape == (soc_data.ACTION_SIZE, 7, 7)

            seqs = dataset[1]
            assert len(seqs[0]) == 30
            assert seqs[0][0].shape == (soc_data.STATE_SIZE, 7, 7)
            assert seqs[1][0].shape == (soc_data.ACTION_SIZE, 7, 7)
