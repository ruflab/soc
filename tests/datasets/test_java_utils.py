import os
import unittest
import torch
import numpy as np
import pandas as pd
from soc.datasets import java_utils as ju
from soc.datasets import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))
fixture_dir = os.path.join(cfd, '..', 'fixtures')


class TestJavaUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = torch.load(os.path.join(fixture_dir, 'soc_seq_3_raw_df.pt'))

        def _get_states_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][0]

        def _get_actions_from_db_se_f(self, idx: int) -> pd.DataFrame:
            return data[idx][1]

        cls._get_states_from_db_se_f = _get_states_from_db_se_f
        cls._get_actions_from_db_se_f = _get_actions_from_db_se_f

    def test_parse_layout(self):
        data = '{1,2,3,4,54}'
        formatted_data = ju.parse_layout(data)

        np.testing.assert_array_equal(formatted_data, [1, 2, 3, 4, 54])

    def test_mapping_1d_2d(self):
        states_df = self._get_states_from_db_se_f(0)
        hexlayout = states_df['hexlayout'].iloc[0]

        data_2d = ju.mapping_1d_2d(ju.parse_layout(hexlayout))
        x = np.array(data_2d)

        assert x.shape == (1, 7, 7)

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

        y = np.zeros([1, 7, 7])
        y[0, 3, 5] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_pieces_empty(self):
        x = ju.parse_pieces('{}')
        y = np.zeros([4 * 18, 7, 7])

        np.testing.assert_array_equal(x, y)

    def test_parse_pieces(self):
        pieces = r'{{2,137,0},{1,148,1},{0,167,2}}'

        x = ju.parse_pieces(pieces)

        y = np.zeros([4 * 18, 7, 7])
        # City player 0
        y[0 + 12 + 3, 2, 3] = 1
        y[0 + 12 + 1, 3, 3] = 1
        y[0 + 12 + 5, 3, 4] = 1
        # Settlement player 1
        # y[5, 3, 18 + 6 + 0] = 1  # Water tile not counted for now
        y[18 + 6 + 4, 5, 4] = 1
        y[18 + 6 + 2, 5, 3] = 1
        # Road player 2
        y[36 + 0 + 2, 4, 4] = 1
        y[36 + 0 + 5, 5, 5] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_player_infos(self):
        states_df = self._get_states_from_db_se_f(0)
        p_infos = states_df['players'].iloc[-1]

        players_plans = ju.parse_player_infos(p_infos)

        assert players_plans.shape == (4 * 41, 7, 7)

    def test_parse_actions(self):
        actions_df = self._get_actions_from_db_se_f(0)
        actions = actions_df['type'].iloc[[0, 1, 2, 5, -1]]

        x = np.stack(actions.apply(ju.parse_actions))

        y = np.zeros([5, soc_data.ACTION_SIZE, 7, 7])
        y[0, 1] = 1
        y[1, 5] = 1
        y[2, 4] = 1
        y[3, 5] = 1
        y[4, 5] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_dice_result(self):
        states_df = self._get_states_from_db_se_f(0)
        diceresults = states_df['diceresult'].iloc[[0, 1, 2, 5, -1]]

        x = np.stack(diceresults.apply(ju.parse_dice_result))

        y = np.zeros([5, soc_data.STATE_COLS_SIZE['diceresult'], 7, 7])

        y[0, -2] = 1
        y[1, -2] = 1
        y[2, -2] = 1
        y[3, -2] = 1
        y[4, -2] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_game_phases(self):
        states_df = self._get_states_from_db_se_f(0)
        gamestates = states_df['gamestate'].iloc[[0, 1, 2, 5, -1]]

        x = np.stack(gamestates.apply(ju.parse_game_phases))

        y = np.zeros([5, soc_data.STATE_COLS_SIZE['gamestate'], 7, 7])

        y[0, 5] = 1
        y[1, 6] = 1
        y[2, 5] = 1
        y[3, 6] = 1
        y[4, 6] = 1

        np.testing.assert_array_equal(x, y)

    def test_parse_current_player(self):
        states_df = self._get_states_from_db_se_f(0)
        currentplayers = states_df['currentplayer'].iloc[[0, 1, 2, 5, -1]]

        x = np.stack(currentplayers.apply(ju.parse_current_player))

        y = np.zeros([5, 4, 7, 7])

        y[0, 1] = 1
        y[1, 1] = 1
        y[2, 2] = 1
        y[3, 3] = 1
        y[4, 0] = 1

        np.testing.assert_array_equal(x, y)
