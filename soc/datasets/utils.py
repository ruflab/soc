import torch
import pandas as pd
import numpy as np
from torch.nn.utils import rnn as rnn_utils
from typing import TypeVar
from ..typing import SocSeqList, SocSeqBatch, SocSeqPolicyBatch, SocSeqPolicyList
from . import java_utils as ju

DataTensor = TypeVar('DataTensor', np.ndarray, torch.Tensor)


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


def pad_seq_policy(inputs: SocSeqPolicyList) -> SocSeqPolicyBatch:
    """
        Pad the different inputs

        inputs is a list of (state_seq, actions_seq)
    """
    xs_l = []
    ys_spatial_l = []
    ys_linear_l = []
    ys_action_l = []
    mask_spatial_l = []
    mask_linear_l = []
    mask_action_l = []
    for tuple_seq in inputs:
        x, y = tuple_seq
        y_spatial, y_linear, y_action = y

        xs_l.append(x)

        ys_spatial_l.append(y_spatial)
        ys_linear_l.append(y_linear)
        ys_action_l.append(y_action)

        mask_spatial_l.append(torch.ones_like(y_spatial))
        mask_linear_l.append(torch.ones_like(y_linear))
        mask_action_l.append(torch.ones_like(y_action))

    xs_t = rnn_utils.pad_sequence(xs_l, batch_first=True)

    ys_spatial_t = rnn_utils.pad_sequence(ys_spatial_l, batch_first=True)
    ys_linear_t = rnn_utils.pad_sequence(ys_linear_l, batch_first=True)
    ys_action_t = rnn_utils.pad_sequence(ys_action_l, batch_first=True)

    mask_spatial_t = rnn_utils.pad_sequence(mask_spatial_l, batch_first=True)
    mask_linear_t = rnn_utils.pad_sequence(mask_linear_l, batch_first=True)
    mask_action_t = rnn_utils.pad_sequence(mask_action_l, batch_first=True)

    ys_t = (ys_spatial_t, ys_linear_t, ys_action_t)
    mask_t = (mask_spatial_t, mask_linear_t, mask_action_t)

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
                                                   .apply(ju.mapping_1d_2d) \
                                                   .apply(normalize_hexlayout)
    df_states['numberlayout'] = df_states['numberlayout'].apply(ju.parse_layout) \
                                                         .apply(ju.mapping_1d_2d) \
                                                         .apply(normalize_numberlayout)

    df_states['robberhex'] = df_states['robberhex'].apply(ju.get_1d_id_from_hex) \
                                                   .apply(ju.get_2d_id) \
                                                   .apply(ju.get_one_hot_plan)

    df_states['piecesonboard'] = df_states['piecesonboard'].apply(ju.parse_pieces)

    df_states['players'] = df_states['players'].apply(ju.parse_player_infos)

    df_states['gamestate'] = df_states['gamestate'].apply(ju.parse_game_phases)

    df_states['devcardsleft'] = df_states['devcardsleft'].apply(ju.get_replicated_plan) \
                                                         .apply(normalize_devcardsleft)

    df_states['diceresult'] = df_states['diceresult'].apply(ju.parse_dice_result)

    df_states['startingplayer'] = df_states['startingplayer'].apply(ju.parse_starting_player)

    df_states['currentplayer'] = df_states['currentplayer'].apply(ju.parse_current_player)

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


def normalize_hexlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone().type(torch.float32)  # type:ignore
        # We add 1 to replace the -1 values with zeros and avoid any other 0 in the data
        data += 1
        # We make sure all the values are between 1 and 256 so that
        # All log values are between 0 and 256
        val = torch.tensor(255 + 1 + 1, dtype=data.dtype)
        data = torch.log(data + 1) / torch.log(val)
    else:
        data = data.copy()
        data += 1
        data = np.log(data + 1) / np.log(255 + 1 + 1)

    return data


def unnormalize_hexlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone()
        val = torch.tensor(255 + 1 + 1, dtype=data.dtype)
        data = torch.exp(data * torch.log(val)) - 1
        data = torch.round(data).type(torch.int64)  # type:ignore
    else:
        data = data.copy()
        data = np.exp(data * np.log(255 + 1 + 1)) - 1
        data = np.round(data).astype(np.int64)

    data -= 1

    return data


def normalize_numberlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone().type(torch.float32)  # type:ignore
        # We replace -1 with 0 to avoid sending any signals to the model
        data[data == -1] = 0
        # # We make sure all the values are between 0 and 1
        data = data / 12.
    else:
        data = data.copy()
        data[data == -1] = 0
        data = data / 12.

    return data


def unnormalize_numberlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone()
        data = data * 12
        data[data == 0] = -1
        data = torch.round(data).type(torch.int64)  # type:ignore
    else:
        data = data.copy()
        data = data * 12
        data[data == 0] = -1
        data = np.round(data).astype(np.int64)

    return data


def normalize_devcardsleft(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone().type(torch.float32)  # type:ignore
        data = data / 25.
    else:
        data = data.copy()
        data = data / 25.

    return data


def unnormalize_devcardsleft(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data * 25.
        data = torch.round(data).type(torch.int64)  # type:ignore
    else:
        data = data.copy()
        data = np.round(data).astype(np.int64)

    return data
