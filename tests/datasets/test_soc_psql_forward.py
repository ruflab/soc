import os
import unittest
import pandas as pd
import numpy as np
import torch
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from unittest.mock import MagicMock
from soc import datasets

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestSocPSQLForwardSAToSADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLForwardConfig)

        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(
            self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = data[table_id][0]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = data[table_id][1]
            return seq[start_row_id:end_row_id]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_dataset_index(self):
        with initialize():
            config = compose(
                config_name="config",
                overrides=[
                    "no_db=true",
                    "history_length=3",
                    "future_length=2",
                    "first_index=0",
                    "psql_password=dummy"
                ]
            )
            dataset = datasets.SocPSQLForwardSAToSAPolicyDataset(config)
            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
            dataset._get_trajectories_length = MagicMock(return_value=[9, 9])

            input_size = dataset.get_input_size()
            output_shape_spatial, output_shape, output_shape_actions = dataset.get_output_size()

            inputs, outputs = dataset[0]

            np.testing.assert_array_equal(inputs.shape, input_size)
            np.testing.assert_array_equal(outputs[0].shape, output_shape_spatial)
            np.testing.assert_array_equal(outputs[1].shape, output_shape)
            np.testing.assert_array_equal(outputs[2].shape, output_shape_actions)


class TestSocPSQLForwardSAToSAPolicyDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLForwardConfig)

        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(
            self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = data[table_id][0]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
            self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = data[table_id][1]
            return seq[start_row_id:end_row_id]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_dataset_index(self):
        with initialize():
            config = compose(
                config_name="config",
                overrides=[
                    "no_db=true",
                    "history_length=3",
                    "future_length=2",
                    "first_index=0",
                    "psql_password=dummy"
                ]
            )
            dataset = datasets.SocPSQLForwardSAToSAPolicyDataset(config)
            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
            dataset._get_trajectories_length = MagicMock(return_value=[9, 9])

            input_size = dataset.get_input_size()
            output_shape_spatial, output_shape, output_shape_actions = dataset.get_output_size()

            inputs, outputs = dataset[0]

            np.testing.assert_array_equal(inputs.shape, input_size)
            np.testing.assert_array_equal(outputs[0].shape, output_shape_spatial)
            np.testing.assert_array_equal(outputs[1].shape, output_shape)
            np.testing.assert_array_equal(outputs[2].shape, output_shape_actions)
