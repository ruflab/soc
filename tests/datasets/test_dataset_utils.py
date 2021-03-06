import os
import unittest
import pandas as pd
import numpy as np
import torch
from typing import List
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from unittest.mock import MagicMock
from soc import datasets
from soc.datasets import utils as ds_utils
from soc.datasets import java_utils as ju

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestDatasetUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLConfig)

        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))
        text_bert_data = torch.load(os.path.join(fixture_dir, 'soc_text_bert_3_fullseq.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        def _get_text_bert_seq(self, idx: int) -> List[torch.Tensor]:
            return text_bert_data[idx]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f
        cls._get_text_bert_seq = _get_text_bert_seq

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
        hexlayout = seq_data['hexlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d).iloc[0]
        normed = ds_utils.normalize_hexlayout(hexlayout)
        hexlayout_reconstructed = ds_utils.unnormalize_hexlayout(normed)

        np.testing.assert_array_equal(hexlayout_reconstructed, hexlayout)

    def test_normalize_hexlayout_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        hexlayout = seq_data['hexlayout'].apply(ju.parse_layout).apply(ju.mapping_1d_2d).iloc[0]
        hexlayout_t = torch.tensor(hexlayout)

        normed = ds_utils.normalize_hexlayout(hexlayout_t)
        hexlayout_reconstructed = ds_utils.unnormalize_hexlayout(normed)

        np.testing.assert_array_equal(hexlayout_reconstructed, hexlayout_t)

    def test_normalize_numberlayout_np(self):
        seq_data = self._get_states_from_db_se_f(0)
        numberlayout = seq_data['numberlayout'].apply(ju.parse_layout)\
                                               .apply(ju.mapping_1d_2d).iloc[0]
        normed = ds_utils.normalize_numberlayout(numberlayout)
        numberlayout_reconstructed = ds_utils.unnormalize_numberlayout(normed)

        np.testing.assert_array_equal(numberlayout_reconstructed, numberlayout)

    def test_normalize_numberlayout_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        numberlayout = seq_data['numberlayout'].apply(ju.parse_layout)\
                                               .apply(ju.mapping_1d_2d).iloc[0]
        numberlayout_t = torch.tensor(numberlayout)

        normed = ds_utils.normalize_numberlayout(numberlayout_t)
        numberlayout_reconstructed = ds_utils.unnormalize_numberlayout(normed)

        np.testing.assert_array_equal(numberlayout_reconstructed, numberlayout_t)

    def test_normalize_gameturn_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        gameturn = seq_data['gameturn'].apply(ju.get_replicated_plan).iloc[0]
        gameturn_t = torch.tensor(gameturn)

        normed = ds_utils.normalize_gameturn(gameturn_t)
        gameturn_reconstructed = ds_utils.unnormalize_gameturn(normed)

        np.testing.assert_array_equal(gameturn_reconstructed, gameturn_t)

    def test_normalize_playersresources_torch(self):
        seq_data = self._get_states_from_db_se_f(0)
        playersresources = seq_data['playersresources'].apply(ju.parse_player_resources).iloc[0]
        playersresources_t = torch.tensor(playersresources)

        normed = ds_utils.normalize_playersresources(playersresources_t)
        playersresources_reconstructed = ds_utils.unnormalize_playersresources(normed)

        np.testing.assert_array_equal(playersresources_reconstructed, playersresources_t)

    def test_find_actions_idxs(self):
        data = self._get_text_bert_seq(0)
        sa_seq_t = data[0]
        batch_sa_seq_t = sa_seq_t.unsqueeze(0)
        idxs = ds_utils.find_actions_idxs(batch_sa_seq_t, 'TRADE')

        assert torch.all(
            torch.eq(
                idxs,
                torch.tensor([
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False
                ])
            )
        )
