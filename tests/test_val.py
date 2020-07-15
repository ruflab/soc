import os
# import time
# import shutil
import unittest
# import pandas as pd
import torch
# from unittest.mock import MagicMock
from soc import val

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestVal(unittest.TestCase):

    # df_states: pd.DataFrame
    # df_actions: pd.DataFrame

    # obs_files = [
    #     os.path.join(fixture_dir, 'small_obsgamestates_100.csv'),
    #     os.path.join(fixture_dir, 'small_obsgamestates_101.csv'),
    # ]
    # actions_files = [
    #     os.path.join(fixture_dir, 'small_gameactions_100.csv'),
    #     os.path.join(fixture_dir, 'small_gameactions_101.csv'),
    # ]

    # _get_states_from_db_se_f = None
    # _get_actions_from_db_se_f = None

    # @classmethod
    # def setUpClass(cls):
    #     cls.states = [pd.read_csv(file) for file in cls.obs_files]
    #     cls.actions = [pd.read_csv(file) for file in cls.actions_files]

    # def setUp(self):
    #     self.folder = os.path.join(fixture_dir, str(int(time.time() * 1000000)))

    # def tearDown(self):
    #     return shutil.rmtree(self.folder)

    def test_mean_by_idx(self):
        a = torch.ones([3, 4, 5])
        a[:, :, 3] = 2
        a[:, 1] = -1
        a[0] = 0

        b = torch.ones([3, 4, 5])
        b[:, :, 3] = 3
        b[:, 1] = -1
        b[0] = 0

        try:
            x = val.mean_by_idx(a, a, 1, 1, torch.float32)
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
        metadata = {'1': [[0, 1]], '2': [[1, None]]}

        a = torch.ones([3, 4, 5])
        a[:, :, 3] = 2
        a[:, 1] = -1
        a[0] = 0

        b = torch.ones([3, 4, 5])
        b[:, :, 3] = 3
        b[:, 1] = -1
        b[0] = 0

        stats_dict = val.get_stats(metadata, a, b)
        true_dict = {'acc_1': 13 / 15, 'acc_2': 41 / 45}
        self.assertDictEqual(stats_dict, true_dict)
