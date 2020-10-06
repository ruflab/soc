import os
import unittest
import torch
import pandas as pd
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from unittest.mock import MagicMock
from soc import datasets

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestSocPSQLTextSeqDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cs = ConfigStore.instance()
        cs.store(name="config", node=datasets.PSQLTextConfig)

        data = torch.load(os.path.join(fixture_dir, 'soc_text_bert_3_raw_df.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        def _get_chats_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][2]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f
        cls._get_chats_from_db_se_f = _get_chats_from_db_se_f

    def test_dataset_index(self):
        with initialize():
            config = compose(config_name="config", overrides=["psql_password=dummy"])
            dataset = datasets.SocPSQLTextBertSeqDataset(config)

            dataset._get_states_from_db = MagicMock(side_effect=self._get_states_from_db_se_f)
            dataset._get_actions_from_db = MagicMock(side_effect=self._get_actions_from_db_se_f)
            dataset._get_chats_from_db = MagicMock(side_effect=self._get_chats_from_db_se_f)

            dataset[0]
