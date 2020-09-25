import os
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import List, Callable, Union, Tuple
from . import utils as ds_utils
from ..typing import SocDatasetItem, SocDataMetadata
from . import soc_data

cfd = os.path.dirname(os.path.realpath(__file__))
_DATA_FOLDER = os.path.join(cfd, '..', '..', 'data')

OutputShape = Union[List[int], Tuple[List[int], ...]]


@dataclass
class PreprocessedSeqConfig:
    name: str = MISSING
    dataset_path: str = os.path.join(_DATA_FOLDER, 'soc_50_fullseq.pt')
    max_seq_length: int = -1

    shuffle: bool = True


class SocPreprocessedSeqSAToSDataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, C_states, H, W]
    """

    output_shape: OutputShape

    def __init__(self, config: DictConfig):
        super(SocPreprocessedSeqSAToSDataset, self).__init__()

        self.path = config['dataset_path']

        self.data = torch.load(self.path)
        self._set_props(config)

    def _set_props(self, config):
        self.input_shape = [-1, soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        self.output_shape = [-1, soc_data.STATE_SIZE] + soc_data.BOARD_SIZE

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> SocDatasetItem:
        seq = self._get_data(idx)
        x_t = seq[:-1]
        y_t = seq[1:].clone()[:, :soc_data.STATE_SIZE]

        return x_t, y_t

    def _get_data(self, idx: int) -> torch.Tensor:
        seq = self.data[idx]
        if isinstance(seq, list):
            seq = seq[0]
        return seq

    def get_input_size(self) -> List[int]:
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self) -> OutputShape:
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self) -> Callable:
        return ds_utils.pad_seq_sas

    def get_input_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            # field_type = soc_data.STATE_FIELDS_TYPE[field]
            metadata[field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        return metadata

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata[field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        return metadata


class SocPreprocessedSeqSAToSADataset(SocPreprocessedSeqSAToSDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """
    def _set_props(self, config):
        self.input_shape = [-1, soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        self.output_shape = [-1, soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE

    def __getitem__(self, idx: int) -> SocDatasetItem:
        seq = self._get_data(idx)
        x_t = seq[:-1]
        y_t = seq[1:].clone()

        return x_t, y_t

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata['mean_' + field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        metadata['mean_actions'] = [last_idx, last_idx + soc_data.ACTION_SIZE]

        return metadata


class SocPreprocessedSeqSAToSAPolicyDataset(SocPreprocessedSeqSAToSADataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [-1, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [-1, C_ss, H, W], [-1, C_ls], [-1, C_actions] )
    """
    def _set_props(self, config):
        self.input_shape = [-1, soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE

        output_shape_spatial = [-1, soc_data.SPATIAL_STATE_SIZE] + soc_data.BOARD_SIZE
        output_shape = [-1, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [-1, soc_data.ACTION_SIZE]

        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        x_t, y_t = super(SocPreprocessedSeqSAToSAPolicyDataset, self).__getitem__(idx)

        y_states_t = y_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        y_actions_t = y_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        y_spatial_states_t = torch.cat([y_states_t[:, 0:3], y_states_t[:, 9:81]],
                                       dim=1)  # [S, C_ss, H, W]
        y_lin_states_t = torch.cat([y_states_t[:, 3:9, 0, 0], y_states_t[:, 81:, 0, 0]],
                                   dim=1)  # [S, C_ls]

        return (y_t, [y_spatial_states_t, y_lin_states_t, y_actions_t])

    def get_collate_fn(self) -> Callable:
        return ds_utils.pad_seq_policy

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


class SocPreprocessedTruncSeqSAToSAPolicyDataset(SocPreprocessedSeqSAToSAPolicyDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [-1, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [-1, C_ss, H, W], [-1, C_ls], [-1, C_actions] )
    """
    def _set_props(self, config):
        self.max_seq_length = config['max_seq_length']

        self.input_shape = [-1, soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE

        output_shape_spatial = [-1, soc_data.SPATIAL_STATE_SIZE] + soc_data.BOARD_SIZE
        output_shape = [-1, soc_data.STATE_SIZE - soc_data.SPATIAL_STATE_SIZE]
        output_shape_actions = [-1, soc_data.ACTION_SIZE]

        self.output_shape = (output_shape_spatial, output_shape, output_shape_actions)

    def __getitem__(self, idx: int):
        x_t, y_t = super(SocPreprocessedSeqSAToSAPolicyDataset, self).__getitem__(idx)

        y_states_t = y_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        y_actions_t = y_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        y_spatial_states_t = torch.cat([y_states_t[:, 0:3], y_states_t[:, 9:81]],
                                       dim=1)  # [S, C_ss, H, W]
        y_lin_states_t = torch.cat([y_states_t[:, 3:9, 0, 0], y_states_t[:, 81:, 0, 0]],
                                   dim=1)  # [S, C_ls]

        return (y_t, [y_spatial_states_t, y_lin_states_t, y_actions_t])

    def get_collate_fn(self) -> Callable:
        return ds_utils.pad_seq_policy
