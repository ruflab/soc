import os
import unittest
import pandas as pd
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from unittest.mock import MagicMock
import soc
from soc import datasets
from soc.datasets import utils as ds_utils
from soc.datasets import soc_data

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
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLConfig)

        states = [pd.read_csv(file) for file in cls.obs_files]
        actions = [pd.read_csv(file) for file in cls.actions_files]

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return states[idx]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return actions[idx]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_seq_data_loading_pipeline(self):
        with initialize():
            config = compose(
                config_name="config",
                overrides=["no_db=true"]
            )
            dataset = soc.datasets.SocPSQLSeqSAToSDataset(config)
            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
            dataset._get_length = MagicMock(return_value=2)

            dataloader = DataLoader(dataset, batch_size=2, collate_fn=ds_utils.pad_seq_sas)

            x = next(iter(dataloader))

            assert len(x) == 3
            assert x[0].shape == (2, 8, soc_data.STATE_SIZE + soc_data.ACTION_SIZE, 7, 7)
            assert x[1].shape == (2, 8, soc_data.STATE_SIZE, 7, 7)
            assert x[2].shape == (2, 8, soc_data.STATE_SIZE, 7, 7)
