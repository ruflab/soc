import re
import torch
from torch import nn
import pandas as pd
import numpy as np
from torch._six import container_abcs, string_classes, int_classes
from torch.nn.utils import rnn as rnn_utils
from typing import TypeVar, Dict, List, Tuple
from ..typing import SocSeqList, SocSeqBatch, SocSeqPolicyBatch, SocSeqPolicyList
from . import soc_data
from . import java_utils as ju

DataTensor = TypeVar('DataTensor', np.ndarray, torch.Tensor)
np_str_obj_array_pattern = re.compile(r'[SaUO]')


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


def pad_seq_text_policy(data_dict_l):
    max_text_lengh = 0
    for data_dict in data_dict_l:
        max_text_lengh = max(max_text_lengh, data_dict['chat_history_t'].shape[1])
    for i, data_dict in enumerate(data_dict_l):
        if data_dict['chat_history_t'].shape[1] < max_text_lengh:
            zeros_shape_history = [
                data_dict['chat_history_t'].shape[0],
                max_text_lengh - data_dict['chat_history_t'].shape[1],
                data_dict['chat_history_t'].shape[2]
            ]
            zeros = torch.zeros(zeros_shape_history, dtype=torch.float32)

            data_dict_l[i]['chat_history_t'] = torch.cat([data_dict['chat_history_t'], zeros],
                                                         dim=1)
            # yapf: disable
            data_dict_l[i]['chat_mask_history_t'] = torch.cat(
                [data_dict['chat_mask_history_t'], zeros[:, :, 0]],
                dim=1
            )
            # yapf: enable

            zeros_shape_future = [
                data_dict['chat_future_t'].shape[0],
                max_text_lengh - data_dict['chat_future_t'].shape[1],
                data_dict['chat_future_t'].shape[2]
            ]
            zeros_shape_future[1] = max_text_lengh - data_dict['chat_future_t'].shape[1]
            zeros = torch.zeros(zeros_shape_future, dtype=torch.float32)

            data_dict_l[i]['chat_future_t'] = torch.cat([data_dict['chat_future_t'], zeros], dim=1)
            # yapf: disable
            data_dict_l[i]['chat_mask_future_t'] = torch.cat(
                [data_dict['chat_mask_future_t'], zeros[:, :, 0]],
                dim=1
            )
            # yapf: enable

    return default_collate(data_dict_l)


def default_collate(batch):
    r"""
        Default Pytorch 1.6 collate function

        Puts each data field into a tensor with outer dimension batch size
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                default_collate_err_msg_format = (
                    "default_collate: batch must contain tensors, numpy arrays, numbers, "
                    "dicts or lists; found {}"
                )
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def preprocess_states(states_df: pd.DataFrame) -> pd.DataFrame:
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
    states_df = states_df.copy()
    del states_df['touchingnumbers']
    del states_df['name']
    del states_df['id']

    states_df['gameturn'] = states_df['gameturn'].apply(ju.get_replicated_plan)

    states_df['hexlayout'] = states_df['hexlayout'].apply(ju.parse_layout) \
                                                   .apply(ju.mapping_1d_2d) \
                                                   .apply(normalize_hexlayout)
    states_df['numberlayout'] = states_df['numberlayout'].apply(ju.parse_layout) \
                                                         .apply(ju.mapping_1d_2d) \
                                                         .apply(normalize_numberlayout)

    states_df['robberhex'] = states_df['robberhex'].apply(ju.get_1d_id_from_hex) \
                                                   .apply(ju.get_2d_id) \
                                                   .apply(ju.get_one_hot_plan)

    states_df['piecesonboard'] = states_df['piecesonboard'].apply(ju.parse_pieces)

    states_df['gamestate'] = states_df['gamestate'].apply(ju.parse_game_phases)

    states_df['devcardsleft'] = states_df['devcardsleft'].apply(ju.parse_devcardsleft)

    states_df['diceresult'] = states_df['diceresult'].apply(ju.parse_dice_result)

    states_df['startingplayer'] = states_df['startingplayer'].apply(ju.parse_starting_player)

    states_df['currentplayer'] = states_df['currentplayer'].apply(ju.parse_current_player)

    states_df['playeddevcard'] = states_df['playeddevcard'].apply(ju.get_replicated_plan)

    states_df['playersresources'] = states_df['playersresources'].apply(ju.parse_player_resources)

    states_df['players'] = states_df['players'].apply(ju.parse_player_infos)

    return states_df


def preprocess_actions(actions_df: pd.DataFrame) -> pd.DataFrame:
    actions_df = actions_df.copy()
    del actions_df['id']
    del actions_df['beforestate']
    del actions_df['afterstate']
    del actions_df['value']

    actions_df['type'] = actions_df['type'].apply(ju.parse_actions)
    # The first action is igniting the first state so we remove it
    actions_df = actions_df[1:]
    # and we duplicate the last one to keep the same numbers of state-actions
    actions_df = actions_df.append(actions_df.iloc[-1])

    return actions_df


def preprocess_chats(
    chats_df: pd.DataFrame, game_length: int, first_state_idx: int = 0
) -> pd.DataFrame:
    data: Dict[str, List] = {'message': [[] for i in range(game_length)]}

    for _, row in chats_df.iterrows():
        # Index start at 1 in the DB
        db_state = (row['current_state'] - 1) - first_state_idx
        mess = "{}: {}".format(row['sender'], row['message'])

        data['message'][db_state].append(mess)
    data['message'] = list(map(lambda x: '' if len(x) == 0 else '\n'.join(x), data['message']))
    chats_preproc_df = pd.DataFrame(data)

    return chats_preproc_df


def stack_states_df(states_df: pd.DataFrame) -> torch.Tensor:
    state_seq = []
    for i in range(len(states_df.index)):
        current_state_df = states_df.iloc[i]

        current_state_np = np.concatenate(
            [current_state_df[col] for col in soc_data.STATE_FIELDS], axis=0
        )  # yapf: ignore

        state_seq.append(torch.tensor(current_state_np, dtype=torch.float32))

    state_seq_t = torch.stack(state_seq)

    return state_seq_t


def stack_actions_df(actions_df: pd.DataFrame) -> torch.Tensor:
    action_seq = []
    for i in range(len(actions_df.index)):
        current_action_df = actions_df.iloc[i]
        current_action_np = current_action_df['type']
        action_seq.append(torch.tensor(current_action_np, dtype=torch.float32))

    action_seq_t = torch.stack(action_seq)

    return action_seq_t


def replace_firstnames(text, lm='bert'):
    if lm == 'bert':
        return text.replace('BayesBetty', 'Betty')\
            .replace('BayesFranck', 'Peter')\
            .replace('BayesJake', 'Jake')\
            .replace('DRLSam', 'Sam')\
            # .replace('\n', '[SEP]')  # TODO: are we sure we want to use a separtor?

    else:
        raise NotImplementedError('LM {} is not supported'.format(lm))


def compute_text_features(
    messages: List[str],
    tokenizer,
    text_model: nn.Module
) -> List[torch.Tensor]:
    encoded_inputs = tokenizer(
        messages, padding=True, truncation=True, return_tensors="pt"
    )

    empty_last_hidden_state = None
    empty_pooler_output = None
    last_hidden_state_list = []
    pooler_output_list = []
    # To speed things up, we compute only once the representation for the void sentences
    # This works because there are much more void sentences than actual sentences
    for i in range(len(messages)):
        if messages[i] == '':
            if empty_last_hidden_state is None:
                empty_last_hidden_state, empty_pooler_output = text_model(
                    input_ids=encoded_inputs['input_ids'][i:i + 1],
                    token_type_ids=encoded_inputs['token_type_ids'][i:i + 1],
                    attention_mask=encoded_inputs['attention_mask'][i:i + 1],
                )
            last_hidden_state_list.append(empty_last_hidden_state)
            pooler_output_list.append(empty_pooler_output)
        else:
            last_hidden_state, pooler_output = text_model(
                input_ids=encoded_inputs['input_ids'][i:i + 1],
                token_type_ids=encoded_inputs['token_type_ids'][i:i + 1],
                attention_mask=encoded_inputs['attention_mask'][i:i + 1],
            )
            last_hidden_state_list.append(last_hidden_state)
            pooler_output_list.append(pooler_output)
    last_hidden_state = torch.cat(last_hidden_state_list, dim=0)
    pooler_output = torch.cat(pooler_output_list, dim=0)

    mask = encoded_inputs['attention_mask'].to(torch.float32)

    return [last_hidden_state, pooler_output, mask]


def normalize_hexlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone().type(torch.float32)  # type:ignore
        # We add 1 to replace the -1 values with zeros and avoid any other 0 in the data
        data += 1
        # We make sure all the values are between 1 and 257 so that
        # All log values are between 0 and 257
        val = torch.tensor(255 + 1 + 1, dtype=data.dtype)
        data = torch.sqrt(data + 1) / torch.sqrt(val)
    else:
        data = data.copy()
        data += 1
        data = np.sqrt(data + 1) / np.sqrt(255 + 1 + 1)

    return data


def unnormalize_hexlayout(data: DataTensor) -> DataTensor:
    if isinstance(data, torch.Tensor):
        data = data.clone()
        val = torch.tensor(255 + 1 + 1, dtype=data.dtype)
        data = torch.square(data * torch.sqrt(val)) - 1
        data = torch.round(data).type(torch.int64)  # type:ignore
    else:
        data = data.copy()
        data = np.square(data * np.sqrt(255 + 1 + 1)) - 1
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
        data = torch.round(data).type(torch.int64)  # type:ignore
        data[data == 0] = -1
    else:
        data = data.copy()
        data = data * 12
        data = np.round(data).astype(np.int64)
        data[data == 0] = -1

    return data


def find_actions_idxs(batch_sa_seq_t: torch.Tensor, action_name: str) -> torch.Tensor:
    action_idx = soc_data.ACTIONS_NAMES.index(action_name)

    action_seq = torch.argmax(batch_sa_seq_t[:, :, -len(soc_data.ACTIONS):, 0, 0], dim=2)
    idxs = (action_seq == action_idx)

    return idxs


def separate_state_data(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_states_l = []
    linear_states_l = []
    last_idx = 0

    for field in soc_data.STATE_FIELDS:
        field_type = soc_data.STATE_FIELDS_TYPE[field]
        field_size = soc_data.STATE_FIELDS_SIZE[field]

        if field_type in [3, 4, 5]:
            sub_state = state[:, last_idx:last_idx + field_size]
            spatial_states_l.append(sub_state)
        else:
            sub_state = state[:, last_idx:last_idx + field_size, 0, 0]
            linear_states_l.append(sub_state)
        last_idx += field_size

    spatial_states_t = torch.cat(spatial_states_l, dim=1)
    lineat_states_t = torch.cat(linear_states_l, dim=1)

    return spatial_states_t, lineat_states_t
