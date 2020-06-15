import os
import unittest
import numpy as np
import pandas as pd
from soc import utils

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestUtils(unittest.TestCase):
    def test_parse_layout(self):
        data = '{1,2,3,4,54}'
        formatted_data = utils.parse_layout(data)

        np.testing.assert_array_equal(formatted_data, [1, 2, 3, 4, 54])

    def test_mapping_1d_2d(self):
        obs_file = os.path.join(fixture_dir, 'obsgamestates_100.csv')
        df_states = pd.read_csv(obs_file)
        hexlayout = df_states['hexlayout'].iloc[0]

        data_2d = utils.mapping_1d_2d(utils.parse_layout(hexlayout))
        x = np.array(data_2d)

        assert x.shape == (7, 7)

    def test_mapping_2d_1d(self):
        mat = np.vstack([
            np.concatenate([np.arange(0, 4), -1 * np.ones(3)]),
            np.concatenate([np.arange(4, 9), -1 * np.ones(2)]),
            np.concatenate([np.arange(9, 15), -1 * np.ones(1)]),
            np.arange(15, 22),
            np.concatenate([-1 * np.ones(1), np.arange(22, 28)]),
            np.concatenate([-1 * np.ones(2), np.arange(28, 33)]),
            np.concatenate([-1 * np.ones(3), np.arange(33, 37)]),
        ]).astype(np.int64)  # yapf: disable

        x = utils.mapping_2d_1d(mat)
        y = list(range(37))

        np.testing.assert_array_equal(x, y)

    def test_get_1d_2d(self):
        id_2d = (3, 5)
        x = utils.get_1d_id(id_2d)

        assert x == 20

    def test_get_one_hot_plan(self):
        id_2d = (3, 5)
        x = utils.get_one_hot_plan(id_2d)

        y = np.zeros([7, 7])
        y[3, 5] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_pieces(self):
        print('TODO!')
