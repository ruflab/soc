import argparse
from argparse import ArgumentParser
import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from ..typing import SocDatasetItem, SocConfig, SocDataMetadata

cfd = os.path.dirname(os.path.realpath(__file__))

SOCShape = Union[Tuple[List[int], ...], List[int]]


class SocPreprocessedForwardSAToSADataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """

    _length: int = -1
    _n_states: int = 245
    _n_spatial_states: int = 2 + 1 + 4 * 18
    _n_spatial_states_wo_map: int = 1 + 4 * 18
    _n_actions: int = 17
    _inc_seq_steps: List[int] = []
    history_length: int
    future_length: int
    input_shape: SOCShape
    output_shape: SOCShape

    def __init__(self, config: SocConfig):
        super(SocPreprocessedForwardSAToSADataset, self).__init__()

        default_path = os.path.join(cfd, '..', '..', 'data', '50_seq_sasa.pt')
        self.path = config.get('dataset_path', default_path)
        data = torch.load(self.path)
        self.seq_data = []
        for x_t, y_t in data:
            new_x_t = torch.cat([x_t, y_t[-1:]], dim=0)
            self.seq_data.append(new_x_t)

        assert 'history_length' in config
        assert 'future_length' in config

        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

        self._set_props(config)

    def _set_props(self, config: SocConfig):
        S, C, H, W = self.seq_data[0].shape
        self.input_shape = [self.history_length, C, H, W]
        self.output_shape = [self.future_length, C, H, W]

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--dataset_path',
            type=str,
            default=argparse.SUPPRESS,
        )
        parser.add_argument('history_length', type=int, default=8)
        parser.add_argument('future_length', type=int, default=1)

        return parser

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self) -> int:
        if self._length == -1:
            total_steps = 0
            nb_games = len(self.seq_data)
            for i in range(nb_games):
                total_steps += self.seq_data[i].shape[0]

            self._length = total_steps - nb_games * self.seq_len_per_datum

        return self._length

    def _set_stats(self):
        nb_steps = self._get_nb_steps()
        for i, nb_step in enumerate(nb_steps):
            seq_nb_steps = nb_step - self.seq_len_per_datum

            if i == 0:
                self._inc_seq_steps.append(seq_nb_steps)
            else:
                self._inc_seq_steps.append(seq_nb_steps + self._inc_seq_steps[-1])

    def _get_nb_steps(self) -> List[int]:
        nb_games = len(self.seq_data)
        nb_steps = []
        for i in range(nb_games):
            nb_steps.append(self.seq_data[i].shape[0])

        return nb_steps

    def __getitem__(self, idx: int) -> SocDatasetItem:
        x_t = self._get_data(idx)

        _, _, H, W = x_t.shape
        history_t = x_t[:self.history_length]
        future_t = x_t[self.history_length:]

        return history_t, future_t

    def _get_data(self, idx: int) -> torch.Tensor:
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

        return self.seq_data[table_id][start_row_id:end_row_id]

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

    def get_collate_fn(self) -> None:
        return None

    def get_training_type(self) -> str:
        return 'supervised_forward'

    def get_output_metadata(self) -> SocDataMetadata:
        metadata: SocDataMetadata = {
            'map': [0, 2],
            'robber': [2, 3],
            'properties': [3, 9],
            'pieces': [9, 81],
            'infos': [81, 245],
            'action': [245, 262],
        }

        return metadata


class SocPreprocessedForwardSAToSAPolicyDataset(SocPreprocessedForwardSAToSADataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S_h, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [S_f, C_ss, H, W], [S_f,  C_ls], [S_f,  C_actions] )
    """
    def _set_props(self, config: SocConfig):
        _, C, H, W = self.seq_data[0].shape
        self.input_shape = [self.history_length, C, H, W]

        output_shape_spatial = [self.future_length, self._n_spatial_states, H, W]
        output_shape = [self.future_length, self._n_states - self._n_spatial_states]
        output_shape_actions = [self.future_length, self._n_actions]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        history_t, future_t = super(
            SocPreprocessedForwardSAToSAPolicyDataset, self
        ).__getitem__(idx)
        _, _, H, W = history_t.shape

        future_states_t = future_t[:, :-self._n_actions]  # [S, C_s, H, W]
        future_actions_t = future_t[:, -self._n_actions:, 0, 0]  # [S, C_a]
        future_spatial_states_t = torch.cat([future_states_t[:, 0:3], future_states_t[:, 9:81]],
                                            dim=1)  # [S, C_ss, H, W]
        future_lin_states_t = torch.cat(
            [future_states_t[:, 3:9, 0, 0], future_states_t[:, 81:, 0, 0]], dim=1
        )  # [S, C_ls]

        return (history_t, (future_spatial_states_t, future_lin_states_t, future_actions_t))

    def get_training_type(self) -> str:
        return 'resnet18policy'

    def get_output_metadata(self) -> SocDataMetadata:
        metadata: SocDataMetadata = {
            'map': [0, 2],
            'robber': [2, 3],
            'pieces': [3, self._n_spatial_states],
        }

        return metadata
