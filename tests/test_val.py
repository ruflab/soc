import os
# import time
# import shutil
import unittest
import pandas as pd
import numpy as np
import torch
# from unittest.mock import MagicMock
from soc import val
from soc.datasets import soc_data, utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestVal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    # def setUp(self):
    #     self.folder = os.path.join(fixture_dir, str(int(time.time() * 1000000)))

    # def tearDown(self):
    #     return shutil.rmtree(self.folder)

    def test_compare_by_idx(self):
        a = torch.ones([1, 3, 4, 5])
        a[0, :, :, 3] = 2
        a[0, :, 1] = -1
        a[0, 0] = 0

        b = torch.ones([1, 3, 4, 5])
        b[0, :, :, 3] = 3
        b[0, :, 1] = -1
        b[0, 0] = 0

        try:
            x = val.compare_by_idx(a, a, 1, 1, torch.float32)
            assert False
        except Exception:
            pass

        x = val.compare_by_idx(a, b, 2, 3, torch.float32)
        assert x == 13 / 15

        x = val.compare_by_idx(a, b, 0, dtype=torch.float32)
        assert x == 54 / 60

        x = val.compare_by_idx(a, -1, 0, dtype=torch.float32)
        assert x == 10 / 60

    def test_get_stats(self):
        metadata = {'1': [0, 1], '2': [1, None]}

        a = torch.ones([1, 3, 4, 5])
        a[0, :, :, 3] = 2
        a[0, :, 1] = -1
        a[0, 0] = 0

        b = torch.ones([1, 3, 4, 5])
        b[0, :, :, 3] = 3
        b[0, :, 1] = -1
        b[0, 0] = 0

        stats_dict = val.get_stats(metadata, a, b)
        true_dict = {'1_acc': 13 / 15, '2_acc': 41 / 45}
        self.assertDictEqual(stats_dict, true_dict)

    def test_gamestate_acc(self):
        state = self._get_states_from_db_se_f(0)
        state = utils.preprocess_states(state)
        state = state.iloc[6]
        state_t = torch.tensor(
            np.concatenate([state[col] for col in soc_data.STATE_FIELDS], axis=0),
            dtype=torch.float32
        )  # yapf: ignore
        state_t = state_t.unsqueeze(0).unsqueeze(0)

        indexes = [75, 99]

        acc = val.gamestate_acc(indexes, state_t, state_t)
        assert acc == 1.

        new_state_t = torch.zeros_like(state_t)
        new_state_t[:, :, 0] = 0.6
        new_state_t[:, :, 5] = 0.4
        acc = val.gamestate_acc(indexes, new_state_t, state_t)
        assert acc == 0
