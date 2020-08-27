import os
import unittest
import pandas as pd
import numpy as np
import torch
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from unittest.mock import MagicMock
from soc import datasets
from soc.datasets import utils as ds_utils
from soc.datasets import java_utils as ju

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestUtils(unittest.TestCase):

    states_df: pd.DataFrame
    actions_df: pd.DataFrame

    obs_files = [
        os.path.join(fixture_dir, 'small_obsgamestates_100.csv'),
        os.path.join(fixture_dir, 'small_obsgamestates_101.csv'),
    ]
    actions_files = [
        os.path.join(fixture_dir, 'small_gameactions_100.csv'),
        os.path.join(fixture_dir, 'small_gameactions_101.csv'),
    ]

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

    def test_pad_seq_sas(self):
        with initialize():
            config = compose(config_name="config", overrides=["no_db=true", "psql_password=dummy"])
            dataset = datasets.SocPSQLSeqSAToSDataset(config)

            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)

            data_df = self._get_states_from_db_se_f(0)
            s = len(data_df)

            input_size = dataset.get_input_size()
            output_size = dataset.get_output_size()

            out = ds_utils.pad_seq_sas([dataset[0], dataset[1]])
            in_data = out[0]
            out_data = out[1]

            s_in = in_data.shape[1]
            s_out = out_data.shape[1]

            assert s_in == s_out
            assert in_data.shape == (2, s - 1, input_size[0], input_size[1], input_size[2])
            assert out_data.shape == (2, s - 1, output_size[0], output_size[1], output_size[2])

    def test_normalize_hexlayout_np(self):
        seq_data = self._get_states_from_db_se_f(0)
        hexlayout = seq_data['hexlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d)[0]

        normed = ds_utils.normalize_hexlayout(hexlayout)
        hexlayout_reconstructed = ds_utils.unnormalize_hexlayout(normed)

        np.testing.assert_array_equal(hexlayout_reconstructed, hexlayout)

    def test_normalize_hexlayout_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        hexlayout = seq_data['hexlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d)[0]
        hexlayout_t = torch.tensor(hexlayout)

        normed = ds_utils.normalize_hexlayout(hexlayout_t)
        hexlayout_reconstructed = ds_utils.unnormalize_hexlayout(normed)

        np.testing.assert_array_equal(hexlayout_reconstructed, hexlayout_t)

    def test_normalize_numberlayout_np(self):
        seq_data = self._get_states_from_db_se_f(0)
        numberlayout = seq_data['numberlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d)[0]
        normed = ds_utils.normalize_numberlayout(numberlayout)
        numberlayout_reconstructed = ds_utils.unnormalize_numberlayout(normed)

        np.testing.assert_array_equal(numberlayout_reconstructed, numberlayout)

    def test_normalize_numberlayout_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        numberlayout = seq_data['numberlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d)[0]
        numberlayout_t = torch.tensor(numberlayout)

        normed = ds_utils.normalize_numberlayout(numberlayout_t)
        numberlayout_reconstructed = ds_utils.unnormalize_numberlayout(normed)

        np.testing.assert_array_equal(numberlayout_reconstructed, numberlayout_t)
