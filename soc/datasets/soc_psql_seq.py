import os
import zipfile
import shutil
import sqlalchemy
import numpy as np
import pandas as pd
import torch
from typing import List
from .soc_psql import SocPSQLDataset
from . import utils as ds_utils
from . import soc_data
from .. import utils
from ..typing import SocDatasetItem, SocDataMetadata


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

    def __getitem__(self, idx: int) -> SocDatasetItem:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        states_df = self._get_states_from_db(idx)
        actions_df = self._get_actions_from_db(idx)

        assert len(states_df.index) == len(actions_df.index)
        game_length = len(states_df)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)

        state_seq = []
        action_seq = []
        for i in range(game_length):
            current_state_df = states_df.iloc[i]
            current_action_df = actions_df.iloc[i]

            current_state_np = np.concatenate(
                [current_state_df[col] for col in soc_data.STATE_COLS.keys()], axis=0
            )  # yapf: ignore
            current_action_np = current_action_df['type']

            state_seq.append(torch.tensor(current_state_np, dtype=torch.float32))
            action_seq.append(torch.tensor(current_action_np, dtype=torch.float32))

        state_seq_t = torch.stack(state_seq)
        action_seq_t = torch.stack(action_seq)

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

    def dump_preprocessed_dataset(
        self, folder: str, testing: bool = False, separate_seq: bool = False
    ):
        if testing is True:
            limit = 5
        else:
            limit = len(self)

        if separate_seq:
            folder = "{}/soc_{}_fullseq".format(folder, limit)

        utils.check_folder(folder)

        seqs = []
        for i in range(limit):
            input_seq_t = self._load_input_seq(i)

            if testing is True:
                input_seq_t = input_seq_t[:8]

            if separate_seq:
                path = "{}/{}.pt".format(folder, i)
                torch.save(input_seq_t, path)
            else:
                seqs.append(input_seq_t)

        if separate_seq:

            def zipdir(path, zip_filename):
                ziph = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        ziph.write(file_path, os.path.basename(file_path))

                ziph.close()

            zip_filename = "{}/../soc_{}_fullseq.zip".format(folder, limit)
            zipdir(folder, zip_filename)
            shutil.rmtree(folder)
        else:
            path = "{}/soc_{}_fullseq.pt".format(folder, limit)
            torch.save(seqs, path)

    def _load_input_seq(self, idx: int):
        data = self[idx]

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW

        input_seq_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        return input_seq_t

    def dump_raw_dataset(self, folder: str):
        utils.check_folder(folder)

        limit = len(self)

        data = []
        for i in range(limit):
            states_df = self._get_states_from_db(i)
            actions_df = self._get_actions_from_db(i)

            data.append([states_df, actions_df])

        path = "{}/soc_{}_raw.pt".format(folder, limit)
        torch.save(data, path)


class SocPSQLSeqSAToSDataset(SocPSQLSeqDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: S x (C_states + C_actions) x H x W

        Output: Next state
            Dims: S x C_states x H x W
    """
    def __getitem__(self, idx: int) -> SocDatasetItem:
        data = super(SocPSQLSeqSAToSDataset, self).__getitem__(idx)

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        input_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        x_t = input_t[:-1]
        y_t = state_seq_t[1:]

        return x_t, y_t

    def get_input_size(self) -> List[int]:
        """
            Return the input dimension
        """

        return [
            soc_data.STATE_SIZE + soc_data.ACTION_SIZE,
        ] + soc_data.BOARD_SIZE

    def get_output_size(self) -> List[int]:
        """
            Return the output dimension
        """

        return [
            soc_data.STATE_SIZE,
        ] + soc_data.BOARD_SIZE

    def get_collate_fn(self):
        return ds_utils.pad_seq_sas

    def get_output_metadata(self) -> SocDataMetadata:
        return {
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'robberhex': [2, 3],
            'piecesonboard': [3, 75],
            'gamestate': [75, 99],
            'diceresult': [99, 111],
            'startingplayer': [111, 115],
            'currentplayer': [115, 118],
            'devcardsleft': [118, 119],
            'playeddevcard': [119, 120],
            'players': [120, 284],
        }


class SocPSQLSeqSAToSADataset(SocPSQLSeqDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S, (C_states + C_actions), H, W]

        Output: Next state
            Dims: [S, (C_states + C_actions), H, W]
    """
    def __getitem__(self, idx: int) -> SocDatasetItem:
        data = super(SocPSQLSeqSAToSADataset, self).__getitem__(idx)

        state_seq_t = data[0]  # SxC_sxHxW
        action_seq_t = data[1]  # SxC_axHxW
        cat_seq = torch.cat([state_seq_t, action_seq_t], dim=1)

        x_t = cat_seq[:-1]
        y_t = cat_seq[1:]

        return x_t, y_t

    def get_input_size(self) -> List[int]:
        """
            Return the input dimension
        """
        return [
            soc_data.STATE_SIZE + soc_data.ACTION_SIZE,
        ] + soc_data.BOARD_SIZE

    def get_output_size(self) -> List:
        """
            Return the output dimension
        """

        return [
            soc_data.STATE_SIZE + soc_data.ACTION_SIZE,
        ] + soc_data.BOARD_SIZE

    def get_collate_fn(self):
        return ds_utils.pad_seq_sas

    def get_output_metadata(self) -> SocDataMetadata:
        metadata: SocDataMetadata = {
            'hexlayout': [0, 1],
            'numberlayout': [1, 2],
            'robberhex': [2, 3],
            'piecesonboard': [3, 75],
            'gamestate': [75, 99],
            'diceresult': [99, 111],
            'startingplayer': [111, 115],
            'currentplayer': [115, 118],
            'devcardsleft': [118, 119],
            'playeddevcard': [119, 120],
            'players': [120, 284],
            'actions': [284, 301],
        }

        return metadata
