import os
import unittest
import pandas as pd
import numpy as np
import torch
from soc.datasets import utils, soc_data
from soc import losses

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestLosses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_hexlayout_loss(self):
        state = self._get_states_from_db_se_f(0)
        state = utils.preprocess_states(state)
        state = state.iloc[7]
        state_t = torch.tensor(
            np.concatenate([state[col] for col in soc_data.STATE_COLS.keys()], axis=0),
            dtype=torch.float32
        )  # yapf: ignore
        state_t = state_t.unsqueeze(0)

        indexes = [0, 2]

        loss = losses.hexlayout_loss(indexes, state_t, state_t)
        assert loss == 0

        loss = losses.hexlayout_loss(indexes, state_t + 0.2, state_t)
        assert loss != 0
