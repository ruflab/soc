import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from soc import datasets

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestSocPSQLSeqSAToSDataset(unittest.TestCase):

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

    @classmethod
    def setUpClass(cls):
        states = [pd.read_csv(file) for file in cls.obs_files]
        actions = [pd.read_csv(file) for file in cls.actions_files]

        def _get_states_from_db_se_f(
                self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = states[table_id]
            return seq[start_row_id:end_row_id]

        def _get_actions_from_db_se_f(
                self, table_id: int, start_row_id: int, end_row_id: int
        ) -> pd.DataFrame:
            seq = actions[table_id]
            return seq[start_row_id:end_row_id]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_dataset_index(self):
        config = {'no_db': True, 'history_length': 3, 'future_length': 2, 'first_index': 0}
        dataset = datasets.SocPSQLForwardSAToSADataset(config)
        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_nb_steps = MagicMock(return_value=[9, 9])

        input_size = dataset.get_input_size()
        output_size = dataset.get_output_size()

        inputs, outputs = dataset[0]

        np.testing.assert_array_equal(inputs.shape, input_size)
        np.testing.assert_array_equal(outputs.shape, output_size)
