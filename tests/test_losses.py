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

    _get_states_from_db_se_f = None
    _get_actions_from_db_se_f = None

    @classmethod
    def setUpClass(cls):
        cls.states = [pd.read_csv(file) for file in cls.obs_files]
        cls.actions = [pd.read_csv(file) for file in cls.actions_files]

    def test_hexlayout_loss(self):
        state = self.states[0]
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
