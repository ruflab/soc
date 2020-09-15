import os
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict, Callable
from ..typing import SocDataMetadata
from . import soc_data
from .utils import separate_state_data, pad_seq_text_policy
from .soc_preprocessed_forward import PreprocessedForwardConfig

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', '..', 'data')

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class PreprocessedTextForwardConfig(PreprocessedForwardConfig):
    tokenizer_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    use_pooler_features: bool = True


class SocPreprocessedTextBertForwardSAToSADataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """
    def __init__(self, omegaConf, dataset_type: str = 'train'):
        super(SocPreprocessedTextBertForwardSAToSADataset, self).__init__()

        self.path = omegaConf['dataset_path']
        self.history_length = omegaConf['history_length']
        self.future_length = omegaConf['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self._inc_seq_steps: List[int] = []
        self._length = -1

        self.data = torch.load(self.path)

        self._set_props(omegaConf)

    def _set_props(self, omegaConf):
        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        n_bert_feature = self.data[0][0][1].shape[-1]
        text_input_shape = [self.history_length, None, n_bert_feature]
        self.input_shape = [game_input_shape, text_input_shape]

        self.output_shape = [
            self.future_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self) -> int:
        if self._length == -1:
            total_steps = 0
            nb_games = len(self.data)
            for i in range(nb_games):
                seq = self.data[i]
                if isinstance(seq, list):
                    seq = seq[0]
                total_steps += seq.shape[0]

            self._length = total_steps - nb_games * (self.seq_len_per_datum - 1)

        return self._length

    def _set_stats(self):
        nb_steps = self._get_nb_steps()
        for i, nb_step in enumerate(nb_steps):
            seq_nb_steps = nb_step - (self.seq_len_per_datum - 1)

            if i == 0:
                self._inc_seq_steps.append(seq_nb_steps)
            else:
                self._inc_seq_steps.append(seq_nb_steps + self._inc_seq_steps[-1])

    def _get_nb_steps(self) -> List[int]:
        nb_games = len(self.data)
        nb_steps = []
        for i in range(nb_games):
            seq = self.data[i]
            if isinstance(seq, list):
                seq = seq[0]
            nb_steps.append(seq.shape[0])

        return nb_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_t, chat_seq_t, chat_mask_seq_t = self._get_data(idx)

        _, _, H, W = x_t.shape
        history_t = x_t[:self.history_length]
        future_t = x_t[self.history_length:]
        chat_history_t = chat_seq_t[:self.history_length]
        chat_future_t = chat_seq_t[self.history_length:]
        chat_mask_history_t = chat_mask_seq_t[:self.history_length]
        chat_mask_future_t = chat_mask_seq_t[self.history_length:]

        data_dict = {
            'history_t': history_t,
            'chat_history_t': chat_history_t,
            'chat_mask_history_t': chat_mask_history_t,
            'future_t': future_t,
            'chat_future_t': chat_future_t,
            'chat_mask_future_t': chat_mask_future_t,
        }

        return data_dict

    def _get_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self._inc_seq_steps) == 0:
            self._set_stats()

        prev_seq_steps = 0
        table_id = 0
        for i, seq_steps in enumerate(self._inc_seq_steps):
            if idx < seq_steps:
                table_id = i
                break
            prev_seq_steps = seq_steps
        r = idx - prev_seq_steps
        start_row_id = r
        end_row_id = r + self.seq_len_per_datum

        game_seq_t = self.data[table_id][0][start_row_id:end_row_id]
        chat_seq_t = self.data[table_id][1][start_row_id:end_row_id]
        chat_mask_seq_t = self.data[table_id][2][start_row_id:end_row_id]

        return game_seq_t, chat_seq_t, chat_mask_seq_t

    def get_input_size(self) -> SOCShape:
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self) -> SOCShape:
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self) -> Callable:
        return pad_seq_text_policy

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata['mean_' + field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        metadata['mean_actions'] = [last_idx, last_idx + soc_data.ACTION_SIZE]

        return metadata


class SocPreprocessedTextBertForwardSAToSAPolicyDataset(SocPreprocessedTextBertForwardSAToSADataset
                                                        ):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S_h, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [S_f, C_ss, H, W], [S_f, C_ls], [S_f, C_actions] )
    """
    def _set_props(self, config):
        n_bert_feature = self.data[0][1].shape[-1]
        game_input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        text_input_shape = [self.history_length, None, n_bert_feature]
        self.input_shape = [game_input_shape, text_input_shape]

        output_shape_spatial = [
            self.future_length, soc_data.SPATIAL_STATE_SIZE
        ] + soc_data.BOARD_SIZE
        output_shape = [self.future_length, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [self.future_length, soc_data.ACTION_SIZE]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        data_dict = super(SocPreprocessedTextBertForwardSAToSAPolicyDataset, self).__getitem__(idx)

        future_t = data_dict['future_t']
        del data_dict['future_t']

        states_future_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        actions_future_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]

        spatial_states_future_t, lin_states_future_t = separate_state_data(states_future_t)

        data_dict['spatial_states_future_t'] = spatial_states_future_t
        data_dict['lin_states_future_t'] = lin_states_future_t
        data_dict['actions_future_t'] = actions_future_t

        return data_dict

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        spatial_metadata: SocDataMetadata = {}
        last_spatial_idx = 0

        linear_metadata: SocDataMetadata = {}
        last_linear_idx = 0

        for field in soc_data.STATE_FIELDS:
            field_type = soc_data.STATE_FIELDS_TYPE[field]
            if field_type in [3, 4, 5]:
                spatial_metadata[field] = [
                    last_spatial_idx, last_spatial_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_spatial_idx += soc_data.STATE_FIELDS_SIZE[field]
            else:
                linear_metadata[field] = [
                    last_linear_idx, last_linear_idx + soc_data.STATE_FIELDS_SIZE[field]
                ]
                last_linear_idx += soc_data.STATE_FIELDS_SIZE[field]

        actions_metadata: SocDataMetadata = {
            'actions': [0, soc_data.ACTION_SIZE],
        }

        return (spatial_metadata, linear_metadata, actions_metadata)
