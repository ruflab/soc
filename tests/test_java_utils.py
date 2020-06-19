import os
import unittest
import numpy as np
import pandas as pd
from soc import java_utils as ju

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, 'fixtures')


class TestJavaUtils(unittest.TestCase):
    def test_parse_layout(self):
        data = '{1,2,3,4,54}'
        formatted_data = ju.parse_layout(data)

        np.testing.assert_array_equal(formatted_data, [1, 2, 3, 4, 54])

    def test_mapping_1d_2d(self):
        obs_file = os.path.join(fixture_dir, 'obsgamestates_100.csv')
        df_states = pd.read_csv(obs_file)
        hexlayout = df_states['hexlayout'].iloc[0]

        data_2d = ju.mapping_1d_2d(ju.parse_layout(hexlayout))
        x = np.array(data_2d)

        assert x.shape == (7, 7, 1)

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

        x = ju.mapping_2d_1d(mat)
        y = list(range(37))

        np.testing.assert_array_equal(x, y)

    def test_get_1d_2d(self):
        id_2d = (3, 5)
        x = ju.get_1d_id(id_2d)

        assert x == 20

    def test_get_one_hot_plan(self):
        id_2d = (3, 5)
        x = ju.get_one_hot_plan(id_2d)

        y = np.zeros([7, 7, 1])
        y[3, 5, 0] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_pieces_empty(self):
        x = ju.parse_pieces('{}')
        y = np.zeros([7, 7, 4 * 18])

        np.testing.assert_array_equal(x, y)

    def test_parse_pieces(self):
        pieces = '{{2,137,0},{1,148,1},{0,167,2}}'

        x = ju.parse_pieces(pieces)

        y = np.zeros([7, 7, 4 * 18])
        # City player 0
        y[2, 3, 0 + 12 + 3] = 1
        y[3, 3, 0 + 12 + 1] = 1
        y[3, 4, 0 + 12 + 5] = 1
        # Settlement player 1
        # y[5, 3, 18 + 6 + 0] = 1  # Water tile not counted for now
        y[5, 4, 18 + 6 + 4] = 1
        y[5, 3, 18 + 6 + 2] = 1
        # Road player 2
        y[4, 4, 36 + 0 + 2] = 1
        y[5, 5, 36 + 0 + 5] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_player_infos(self):
        obs_file = os.path.join(fixture_dir, 'obsgamestates_100.csv')
        df = pd.read_csv(obs_file)
        p_infos = df['players'].iloc[50]

        players_plans = ju.parse_player_infos(p_infos)

        assert players_plans.shape == (7, 7, 4 * 41)
