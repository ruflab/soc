import os
import zipfile
import shutil
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from omegaconf import MISSING, DictConfig
from typing import Tuple, List, Union, Dict, Optional
from . import utils as ds_utils
from .. import utils
from . import soc_data

SOCShape = Union[Tuple[List[int], ...], List[int]]


@dataclass
class FileSeqConfig:
    name: str = MISSING
    dataset_path: str = MISSING

    shuffle: bool = True


class SocFileSeqDataset(Dataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    def __init__(self, omegaConf: DictConfig, dataset_type: str = 'train'):
        super(SocFileSeqDataset, self).__init__()

        self.path = omegaConf['dataset_path']

        self.data = torch.load(self.path)
        self._set_props(omegaConf)

    def _set_props(self, config):
        state_shape = [soc_data.STATE_SIZE] + soc_data.BOARD_SIZE
        action_shape = [soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE

        self.input_shape = [state_shape, action_shape]
        self.output_shape = [state_shape, action_shape]

        self.infix = 'seq'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        states_df, actions_df = self._get_data(idx)

        game_length = len(states_df)
        assert game_length == len(states_df.index) == len(actions_df.index)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)

        state_seq_t = ds_utils.stack_states_df(states_df)
        action_seq_t = ds_utils.stack_actions_df(actions_df)

        data_dict = {
            'state_seq_t': state_seq_t,
            'action_seq_t': action_seq_t,
        }

        return data_dict

    def _get_data(self, idx: int):
        return self.data[idx]

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

    def dump_preprocessed_dataset(
        self, folder: str, testing: bool = False, separate_seq: bool = False
    ):
        sec_trunc_idx: Optional[int] = None
        if testing is True:
            nb_games = 3
            sec_trunc_idx = 20
        else:
            nb_games = len(self)

        prefix = "{}/soc_{}_{}_fullseq".format(folder, self.infix, nb_games)
        if separate_seq:
            folder = prefix

        utils.check_folder(folder)

        seqs = []
        for i in range(nb_games):
            print('processing input {}'.format(i))
            inputs_l = self._load_input_seq(i)
            for input_idx, input_t in enumerate(inputs_l):
                inputs_l[input_idx] = input_t[:sec_trunc_idx]

            if separate_seq:
                path = "{}/{}.pt".format(folder, i)
                torch.save(inputs_l, path)
            else:
                seqs.append(inputs_l)

        if separate_seq:

            def zipdir(path, zip_filename):
                ziph = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        ziph.write(file_path, os.path.basename(file_path))

                ziph.close()

            zip_filename = "{}.zip".format(prefix)
            zipdir(folder, zip_filename)
            shutil.rmtree(folder)
        else:
            path = "{}.pt".format(prefix)
            torch.save(seqs, path)

    def _load_input_seq(self, idx: int) -> List[torch.Tensor]:
        data = self[idx]

        state_seq_t = data['state_seq_t']  # SxC_sxHxW
        action_seq_t = data['action_seq_t']  # SxC_axHxW

        input_seq_t = [torch.cat([state_seq_t, action_seq_t], dim=1)]

        return input_seq_t
