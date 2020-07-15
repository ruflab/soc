import argparse
from argparse import ArgumentParser
import os
import torch
from torch.utils.data import Dataset
from typing import List, Callable
from . import utils as ds_utils
from typing import Tuple
from ..typing import SocDatasetItem

cfd = os.path.dirname(os.path.realpath(__file__))


class SocPreprocessedSeqSAToSDataset(Dataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, C_states, H, W]
    """
    def __init__(self, config={}):
        super(SocPreprocessedSeqSAToSDataset, self).__init__()

        default_path = os.path.join(cfd, '..', '..', 'data', '50_seq_sas.pt')
        self.path = config.get('dataset_path', default_path)
        self.data = torch.load(self.path)

        self.input_shape = self.data[0][0].shape[1:]
        self.output_shape = self.data[0][1].shape[1:]

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--dataset_path',
            type=str,
            default=argparse.SUPPRESS,
        )

        return parser

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> SocDatasetItem:
        x_t, y_t = self._get_data(idx)

        return x_t, y_t

    def _get_data(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

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

    def get_collate_fn(self) -> Callable:
        return ds_utils.pad_seq_sas

    def get_training_type(self) -> str:
        return 'supervised_seq'

    def get_output_metadata(self):
        return {
            'map': [[0, 2]],
            'robber': [[2, 3]],
            'properties': [[3, 9]],
            'pieces': [[9, 81]],
            'infos': [[81, 245]],
        }
