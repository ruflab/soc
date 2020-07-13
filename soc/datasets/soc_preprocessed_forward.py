import argparse
from argparse import ArgumentParser
import os
import torch
from torch.utils.data import Dataset
from typing import List
from ..typing import SocDatasetItem, SocConfig

cfd = os.path.dirname(os.path.realpath(__file__))


class SocPreprocessedForwardSAToSADataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S * (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S * (C_states + C_actions), H, W]
    """

    _length: int = -1
    _n_actions: int = 17
    _inc_seq_steps: List[int] = []
    history_length: int
    future_length: int

    def __init__(self, config: SocConfig):
        super(SocPreprocessedForwardSAToSADataset, self).__init__()

        default_path = os.path.join(cfd, '..', '..', 'data', '50_seq_sas.pt')
        self.path = config.get('dataset_path', default_path)
        data = torch.load(self.path)
        self.seq_data = []
        n_action_channels = data[0][0].shape[1] - data[0][1].shape[1]
        for x_t, y_t in data:
            # create a None action
            last_a = torch.zeros([1, n_action_channels, x_t.shape[-2], x_t.shape[-1]])
            last_sa = torch.cat([y_t[-1:], last_a], dim=1)
            new_x_t = torch.cat([x_t, last_sa], dim=0)
            self.seq_data.append(new_x_t)

        assert 'history_length' in config
        assert 'future_length' in config

        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

        self._set_props(config)

    def _set_props(self, config: SocConfig):
        self.input_shape = list(self.seq_data[0].shape[1:])
        self.input_shape[0] *= self.history_length
        self.output_shape = list(self.seq_data[0].shape[1:])
        self.output_shape[0] *= self.future_length

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--dataset_path',
            type=str,
            default=argparse.SUPPRESS, )
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

        return history_t.view(-1, H, W), future_t.view(-1, H, W)

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

    def get_input_size(self) -> List[int]:
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self) -> List[int]:
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self) -> None:
        return None

    def get_training_type(self) -> str:
        return 'supervised_forward'


# class SocPreprocessedForwardSAToSAPolicyDataset(SocPreprocessedForwardSAToSADataset):
#     """
#         Returns a completely formatted dataset:

#         Input: Concatenation of state and actions representation
#         in Sequence.
#             Dims: [S * C_states + C_actions), H, W]

#         Output: Tuple of next state and next actions
#             Dims: ( [S * C_states, H, W], [S * C_actions] )
#     """
#     def _set_props(self, config: SocConfig):
#         self.input_shape = list(self.seq_data[0].shape[1:])
#         self.input_shape[0] *= self.history_length

#         output_shape_1 = list(self.seq_data[0].shape[1:])
#         output_shape_1[0] -= self._n_actions
#         output_shape_1[0] *= self.future_length
#         self.output_shape = (output_shape_1[0], [self._n_actions * self.future_length, 1])

#     def __getitem__(self, idx: int) -> SocDatasetItem:
#         x_t = self._get_data(idx)

#         _, _, H, W = x_t.shape
#         history_t = x_t[:self.history_length]
#         future_t = x_t[self.history_length:]
#         future_states_t = future_t[:, :-self._n_actions]
#         future_actions_t = future_t[:, -self._n_actions:, 0, 0]

#         return (
#             history_t.view(-1, H, W), (future_states_t.view(-1, H, W), future_actions_t.view(-1))
#         )