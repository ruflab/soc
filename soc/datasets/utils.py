import torch
import pandas as pd
from torch.nn.utils import rnn as rnn_utils
from ..typing import SocSeqList, SocSeqBatch
from .. import java_utils as ju


def pad_seq_sas(inputs: SocSeqList) -> SocSeqBatch:
    """
        Pad the different inputs

        inputs is a list of (state_seq, actions_seq)
    """
    xs_l = []
    ys_l = []
    mask_l = []
    for tuple_seq in inputs:
        x, y = tuple_seq

        xs_l.append(x)
        ys_l.append(y)
        mask_l.append(torch.ones_like(y))

    xs_t = rnn_utils.pad_sequence(xs_l, batch_first=True)
    ys_t = rnn_utils.pad_sequence(ys_l, batch_first=True)
    mask_t = rnn_utils.pad_sequence(mask_l, batch_first=True)

    return xs_t, ys_t, mask_t


def preprocess_states(df_states: pd.DataFrame) -> pd.DataFrame:
    """
        This function applies the preprocessing steps necessary to move from the raw
        observation to a spatial representation.

        The spatial representation is like this:
            - plan 0: Tile type (hexlayout)
            - plan 1: Tile number
            - plan 2: Robber position
            - plan 3: Game phase id
            - plan 4: Development card left
            - plan 5: Last dice result
            - plan 6: Starting player id
            - plan 7: Current player id
            - plan 8: Current player has played a developement card during its turn
            3 type of pieces, 6 way to put it around the hex
            - plan 9-26: Player 1 pieces
            - plan 27-44: Player 2 pieces
            - plan 45-62: Player 3 pieces
            - plan 63-80: Player 4 pieces
            see java_utils.parse_player_infos for more information
            - plan 81-121: Player 1 public info
            - plan 122-162: Player 2 public info
            - plan 163-203: Player 3 public info
            - plan 204-244: Player 4 public info

        State shape: 245x7x7
    """
    df_states = df_states.copy()
    del df_states['touchingnumbers']
    del df_states['name']
    del df_states['id']

    df_states['hexlayout'] = df_states['hexlayout'].apply(ju.parse_layout) \
                                                   .apply(ju.mapping_1d_2d)
    df_states['numberlayout'] = df_states['numberlayout'].apply(ju.parse_layout) \
                                                         .apply(ju.mapping_1d_2d)

    df_states['robberhex'] = df_states['robberhex'].apply(ju.get_1d_id_from_hex) \
                                                   .apply(ju.get_2d_id) \
                                                   .apply(ju.get_one_hot_plan)

    df_states['piecesonboard'] = df_states['piecesonboard'].apply(ju.parse_pieces)

    df_states['players'] = df_states['players'].apply(ju.parse_player_infos)

    df_states['gamestate'] = df_states['gamestate'].apply(ju.get_replicated_plan)
    df_states['devcardsleft'] = df_states['devcardsleft'].apply(ju.get_replicated_plan)
    df_states['diceresult'] = df_states['diceresult'].apply(ju.get_replicated_plan)
    df_states['startingplayer'] = df_states['startingplayer'].apply(ju.get_replicated_plan)
    df_states['currentplayer'] = df_states['currentplayer'].apply(ju.get_replicated_plan)
    df_states['playeddevcard'] = df_states['playeddevcard'].apply(ju.get_replicated_plan)

    return df_states


def preprocess_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    df_actions = df_actions.copy()
    del df_actions['id']
    del df_actions['beforestate']
    del df_actions['afterstate']
    del df_actions['value']

    df_actions['type'] = df_actions['type'].apply(ju.parse_actions)

    return df_actions
