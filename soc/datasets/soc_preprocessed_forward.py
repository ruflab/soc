import os
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import List, Tuple, Union
from ..typing import SocDatasetItem, SocDataMetadata
from . import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', '..', 'data')

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class PreprocessedForwardConfig(DictConfig):
    name: str = MISSING
    dataset_path: str = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
    history_length: int = MISSING
    future_length: int = MISSING

    shuffle: bool = True


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
    _inc_seq_steps: List[int] = []
    history_length: int
    future_length: int
    input_shape: SOCShape
    output_shape: SOCShape

    def __init__(self, config: PreprocessedForwardConfig):
        super(SocPreprocessedForwardSAToSADataset, self).__init__()

        self.path = config['dataset_path']
        self.data = torch.load(self.path)

        assert 'history_length' in config
        assert 'future_length' in config

        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

        self._set_props(config)

    def _set_props(self, config):
        self.input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
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
                total_steps += self.data[i].shape[0]

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
        nb_games = len(self.data)
        nb_steps = []
        for i in range(nb_games):
            nb_steps.append(self.data[i].shape[0])

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

        return self.data[table_id][start_row_id:end_row_id]

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
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'mean_robberhex': [2, 3],
            'mean_piecesonboard': [3, 75],
            'mean_gamestate': [75, 99],
            'mean_diceresult': [99, 112],
            'mean_startingplayer': [112, 116],
            'mean_currentplayer': [116, 120],
            'devcardsleft': [120, 121],
            'mean_playeddevcard': [121, 122],
            'players': [122, 286],
            'mean_actions': [286, 303],
        }

        return metadata


class SocPreprocessedForwardSAToSAPolicyDataset(SocPreprocessedForwardSAToSADataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S_h, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [S_f, C_ss, H, W], [S_f, C_ls], [S_f, C_actions] )
    """
    def _set_props(self, config):
        self.input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

        output_shape_spatial = [
            self.future_length, soc_data.SPATIAL_STATE_SIZE
        ] + soc_data.BOARD_SIZE
        output_shape = [self.future_length, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [self.future_length, soc_data.ACTION_SIZE]
        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        history_t, future_t = super(
            SocPreprocessedForwardSAToSAPolicyDataset, self
        ).__getitem__(idx)

        future_states_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        future_actions_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        future_spatial_states_t = torch.cat([future_states_t[:, 0:3], future_states_t[:, 9:81]],
                                            dim=1)  # [S, C_ss, H, W]
        future_lin_states_t = torch.cat(
            [future_states_t[:, 3:9, 0, 0], future_states_t[:, 81:, 0, 0]], dim=1
        )  # [S, C_ls]

        return (history_t, [future_spatial_states_t, future_lin_states_t, future_actions_t])

    def get_training_type(self) -> str:
        return 'resnet18policy'

    def get_output_metadata(self):
        spatial_metadata: SocDataMetadata = {
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'robberhex': [2, 3],
            'piecesonboard': [3, 75],
        }

        linear_metadata: SocDataMetadata = {
            'gamestate': [0, 24],
            'diceresult': [24, 37],
            'startingplayer': [37, 41],
            'currentplayer': [41, 45],
            'devcardsleft': [45, 46],
            'playeddevcard': [46, 47],
            'players': [47, 211],
        }

        actions_metadata: SocDataMetadata = {
            'actions': [0, soc_data.ACTION_SIZE],
        }

        return (spatial_metadata, linear_metadata, actions_metadata)
