import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import soc
from soc import datasets

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')

_DATASET_PATH = os.path.join(fixture_dir, 'soc_5_fullseq.pt')


class TestSocPreprocessedForwardSAToSADataset(unittest.TestCase):

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

        if not os.path.isfile(_DATASET_PATH):
            ds = soc.datasets.SocPSQLSeqDataset({})
            ds.dump_preprocessed_dataset(fixture_dir, 5)

    def test_dataset_index(self):
        config = {
            'no_db': True,
            'history_length': 3,
            'future_length': 2,
            'first_index': 0,
            'dataset_path': _DATASET_PATH
        }
        dataset = datasets.SocPreprocessedForwardSAToSADataset(config)
        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_nb_steps = MagicMock(return_value=[9, 9])

        input_size = dataset.get_input_size()
        output_shape = dataset.get_output_size()

        inputs, outputs = dataset[0]

        np.testing.assert_array_equal(inputs.shape, input_size)
        np.testing.assert_array_equal(outputs.shape, output_shape)

    def test_get_output_metadata(self):
        config = {
            'no_db': True,
            'history_length': 3,
            'future_length': 2,
            'first_index': 0,
            'dataset_path': _DATASET_PATH
        }
        dataset = datasets.SocPreprocessedForwardSAToSADataset(config)
        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_nb_steps = MagicMock(return_value=[9, 9])

        batch = dataset[0]
        y_true = batch[1]

        metadata = dataset.get_output_metadata()

        last_key = list(metadata.keys())[-1]
        assert metadata[last_key][1] == y_true.shape[1]


class TestSocPreprocessedForwardSAToSAPolicyDataset(unittest.TestCase):

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
        config = {
            'no_db': True,
            'history_length': 3,
            'future_length': 2,
            'first_index': 0,
            'dataset_path': _DATASET_PATH
        }
        dataset = datasets.SocPreprocessedForwardSAToSAPolicyDataset(config)
        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_nb_steps = MagicMock(return_value=[9, 9])

        input_size = dataset.get_input_size()
        output_shape_spatial, output_shape, output_shape_actions = dataset.get_output_size()

        inputs, outputs = dataset[0]

        np.testing.assert_array_equal(inputs.shape, input_size)
        np.testing.assert_array_equal(outputs[0].shape, output_shape_spatial)
        np.testing.assert_array_equal(outputs[1].shape, output_shape)
        np.testing.assert_array_equal(outputs[2].shape, output_shape_actions)

    def test_get_output_metadata(self):
        config = {
            'no_db': True,
            'history_length': 3,
            'future_length': 2,
            'first_index': 0,
            'dataset_path': _DATASET_PATH
        }
        dataset = datasets.SocPreprocessedForwardSAToSAPolicyDataset(config)
        dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
        dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
        dataset._get_nb_steps = MagicMock(return_value=[9, 9])

        batch = dataset[0]
        y_spatial_s_true_seq, y_s_true_seq, y_a_true_seq = batch[1]
        metadata = dataset.get_output_metadata()
        spatial_metadata, linear_metadata, actions_metadata = metadata

        last_spatial_key = list(spatial_metadata.keys())[-1]
        assert spatial_metadata[last_spatial_key][1] == y_spatial_s_true_seq.shape[1]
        last_linear_key = list(linear_metadata.keys())[-1]
        assert linear_metadata[last_linear_key][1] == y_s_true_seq.shape[1]
        last_action_key = list(actions_metadata.keys())[-1]
        assert actions_metadata[last_action_key][1] == y_a_true_seq.shape[1]
