import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import Tuple, List, Union
from . import utils as ds_utils
from . import soc_data
from ..typing import SocDataMetadata

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class FileForwardConfig:
    name: str = MISSING
    dataset_path: str = MISSING
    history_length: int = MISSING
    future_length: int = MISSING

    shuffle: bool = True


class SocFileForwardSAToSAPolicyDataset(Dataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    _length: int = -1
    _inc_seq_steps: List[int] = []
    history_length: int
    future_length: int
    input_shape: SOCShape
    output_shape: SOCShape

    def __init__(self, omegaConf: DictConfig, dataset_type: str = 'train'):
        super(SocFileForwardSAToSAPolicyDataset, self).__init__()

        self.path = omegaConf['dataset_path']

        self.history_length = omegaConf['history_length']
        self.future_length = omegaConf['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

        self.data = torch.load(self.path)
        self._set_props(omegaConf)

    def _set_props(self, omegaConf):
        self.input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

        output_shape_spatial = [
            self.future_length, soc_data.SPATIAL_STATE_SIZE
        ] + soc_data.BOARD_SIZE
        output_shape = [self.future_length, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [self.future_length, soc_data.ACTION_SIZE]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self) -> int:
        if self._length == -1:
            total_steps = 0
            nb_games = len(self.data)
            for i in range(nb_games):
                total_steps += len(self.data[i][0])

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
            nb_steps.append(len(self.data[i][0]))

        return nb_steps

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        states_df, actions_df = self._get_data(idx)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)

        to_concat = []
        for i in range(len(states_df)):
            current_state_df = states_df.iloc[i]
            current_action_df = actions_df.iloc[i]

            current_state_np = np.concatenate(
                [current_state_df[col] for col in soc_data.STATE_FIELDS], axis=0
            )  # yapf: ignore
            current_action_np = current_action_df['type']

            to_concat.append(current_state_np)
            to_concat.append(current_action_np)

        history_l = to_concat[:self.history_length * 2]
        future_l = to_concat[self.history_length * 2:]

        input_shape = self.get_input_size()
        # yapf: disable
        output_shape = [self.future_length, ] + input_shape[1:]  # type:ignore
        # yapf: enable
        history_np = np.concatenate(history_l, axis=0).reshape(input_shape)
        future_np = np.concatenate(future_l, axis=0).reshape(output_shape)

        history_t = torch.tensor(history_np, dtype=torch.float32)
        future_t = torch.tensor(future_np, dtype=torch.float32)

        future_states_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        future_actions_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        future_spatial_states_t = torch.cat([future_states_t[:, 0:3], future_states_t[:, 9:81]],
                                            dim=1)  # [S, C_ss, H, W]
        future_lin_states_t = torch.cat(
            [future_states_t[:, 3:9, 0, 0], future_states_t[:, 81:, 0, 0]], dim=1
        )  # [S, C_ls]

        return (history_t, [future_spatial_states_t, future_lin_states_t, future_actions_t])

    def _get_data(self, idx: int):
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

        states_df, actions_df = self.data[table_id]

        return states_df[start_row_id:end_row_id], actions_df[start_row_id:end_row_id]

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

    def get_collate_fn(self):
        return None

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
