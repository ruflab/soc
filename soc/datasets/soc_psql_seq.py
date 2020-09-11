import os
import zipfile
import shutil
import sqlalchemy
import pandas as pd
import torch
from torch import Tensor
from typing import List, Optional, Union, Tuple
from .soc_psql import SocPSQLDataset
from . import utils as ds_utils
from . import soc_data
from .. import utils
from ..typing import SocDatasetItem, SocDataMetadata, SocSize


class SocPSQLSeqDataset(SocPSQLDataset):
    """
        Defines a Settlers of Catan postgresql dataset for sequence models.
        One datapoint is a tuple (states, actions):
        - states is the full sequence of game states
        - actions is the full sequence of actions

        Args:
            psql_username: (str) username
            psql_host: (str) host
            psql_port: (int) port
            psql_db_name: (str) database name

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """
    def _set_props(self, config):
        state_shape = [soc_data.STATE_SIZE] + soc_data.BOARD_SIZE
        action_shape = [soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        self.input_shape = [state_shape, action_shape]
        self.output_shape = [state_shape, action_shape]

        self.infix = 'seq'

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self):
        if self._length == -1 and self.engine is not None:
            query = r"""
                SELECT count(id)
                FROM simulation_games
            """
            res = self.engine.execute(sqlalchemy.text(query))
            self._length = res.scalar()

        return self._length

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        states_df = self._get_states_from_db(idx)
        actions_df = self._get_actions_from_db(idx)

        assert len(states_df.index) == len(actions_df.index)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)

        state_seq_t = ds_utils.stack_states_df(states_df)
        action_seq_t = ds_utils.stack_actions_df(actions_df)

        return state_seq_t, action_seq_t

    def _get_states_from_db(self, idx: int) -> pd.DataFrame:
        db_id = self._first_index + idx
        query = """
            SELECT *
            FROM obsgamestates_{}
        """.format(db_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                states_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return states_df

    def _get_actions_from_db(self, idx: int) -> pd.DataFrame:
        db_id = self._first_index + idx
        query = """
            SELECT *
            FROM gameactions_{}
        """.format(db_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                actions_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return actions_df

    def get_input_size(self) -> SocSize:
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self) -> SocSize:
        """
            Return the output dimension
        """

        return self.output_shape

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

    def _load_input_seq(self, idx: int) -> List[Tensor]:
        data = self[idx]

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW

        input_seq_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        return [input_seq_t]

    def dump_raw_dataset(
        self,
        folder: str,
        testing: bool = False,
    ):
        utils.check_folder(folder)

        if testing is True:
            nb_games = 3
        else:
            nb_games = len(self)

        data = []
        for i in range(nb_games):
            df_list = self._load_input_df_list(i, testing)
            data.append(df_list)

        path = "{}/soc_{}_{}_raw_df.pt".format(folder, self.infix, nb_games)
        torch.save(data, path)

    def _load_input_df_list(self, idx: int, testing: bool = False) -> List[pd.DataFrame]:
        states_df = self._get_states_from_db(idx)
        actions_df = self._get_actions_from_db(idx)

        sec_trunc_idx: Optional[int] = None
        if testing is True:
            sec_trunc_idx = 20
            df_list = [states_df[10:10 + sec_trunc_idx], actions_df[10:10 + sec_trunc_idx]]
        else:
            df_list = [states_df, actions_df]

        return df_list


class SocPSQLSeqSAToSDataset(SocPSQLSeqDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: S x (C_states + C_actions) x H x W

        Output: Next state
            Dims: S x C_states x H x W
    """
    def _set_props(self, config):
        self.input_shape = [soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        self.output_shape = [soc_data.STATE_SIZE] + soc_data.BOARD_SIZE

    def __getitem__(self, idx: int) -> SocDatasetItem:
        data = super(SocPSQLSeqSAToSDataset, self).__getitem__(idx)

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        input_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        x_t = input_t[:-1]
        y_t = state_seq_t[1:]

        return x_t, y_t

    def get_collate_fn(self):
        return ds_utils.pad_seq_sas

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata[field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        return metadata


class SocPSQLSeqSAToSADataset(SocPSQLSeqDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """
    def _set_props(self, config):
        self.input_shape = [soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE
        self.output_shape = [soc_data.STATE_SIZE + soc_data.ACTION_SIZE] + soc_data.BOARD_SIZE

    def __getitem__(self, idx: int) -> SocDatasetItem:
        data = super(SocPSQLSeqSAToSADataset, self).__getitem__(idx)

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        cat_seq = torch.cat([state_seq_t, action_seq_t], dim=1)

        x_t = cat_seq[:-1]
        y_t = cat_seq[1:]

        return x_t, y_t

    def get_collate_fn(self):
        return ds_utils.pad_seq_sas

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata[field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        metadata['actions'] = [last_idx, last_idx + soc_data.ACTION_SIZE]

        return metadata
