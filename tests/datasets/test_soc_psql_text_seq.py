import os
import unittest
# import pandas as pd
# from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
# from unittest.mock import MagicMock
from soc import datasets

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestSocPSQLTextSeqDataset(unittest.TestCase):

    # states_df: pd.DataFrame
    # actions_df: pd.DataFrame

    # obs_files = [
    #     os.path.join(fixture_dir, 'small_obsgamestates_100.csv'),
    #     os.path.join(fixture_dir, 'small_obsgamestates_101.csv'),
    # ]
    # actions_files = [
    #     os.path.join(fixture_dir, 'small_gameactions_100.csv'),
    #     os.path.join(fixture_dir, 'small_gameactions_101.csv'),
    # ]

    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLConfig)

    #     states = [pd.read_csv(file) for file in cls.obs_files]
    #     actions = [pd.read_csv(file) for file in cls.actions_files]

    #     def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
    #         return states[idx]

    #     def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
    #         return actions[idx]

    #     cls._get_states_from_db_se_f = _get_states_from_db_se_f
    #     cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    # def test_dataset_index(self):
    #     with initialize():
    #         config = compose(config_name="config", overrides=["psql_password=dummy"])
    #         dataset = datasets.SocPSQLTextSeqDataset(config)

    #         data = dataset[0]
    #         # dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
    #         # dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)

    #         # data_df = self._get_states_from_db_se_f(0)
    #         # s = len(data_df)

    #         # input_size = dataset.get_input_size()
    #         # output_size = dataset.get_output_size()

    #         # out = dataset[0]

    #         # in_data = out[0]
    #         # out_data = out[1]

    #         # s_in = in_data.shape[0]
    #         # s_out = out_data.shape[0]

    #         # assert s_in == s_out
    #         # assert in_data.shape == (s - 1, input_size[0], input_size[1], input_size[2])
    #         # assert out_data.shape == (s - 1, output_size[0], output_size[1], output_size[2])
