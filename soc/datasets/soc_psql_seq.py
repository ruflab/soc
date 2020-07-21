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
        df_states = self._get_states_from_db(idx)
        df_actions = self._get_actions_from_db(idx)

        assert len(df_states.index) == len(df_actions.index)
        game_length = len(df_states)

        df_states = ds_utils.preprocess_states(df_states)
        df_actions = ds_utils.preprocess_actions(df_actions)

        state_seq = []
        action_seq = []
        for i in range(game_length):
            current_state_df = df_states.iloc[i]
            current_action_df = df_actions.iloc[i]

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

        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_actions_from_db(self, idx: int) -> pd.DataFrame:
        db_id = self._first_index + idx
        query = """
            SELECT *
            FROM gameactions_{}
        """.format(db_id)

        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states


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

        state_seq_t = data[0]  # SxFsxHxW
        action_seq_t = data[1]  # SxFaxHxW
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

    def get_training_type(self):
        return 'supervised_seq'

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

    def dump_preprocessed_dataset(self, folder: str):
        utils.check_folder(folder)

        path = "{}/50_seq_sas.pt".format(folder)
        all_data = [self[i] for i in range(len(self))]

        torch.save(all_data, path)


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

        state_seq_t = data[0]  # SxFsxHxW
        action_seq_t = data[1]  # SxFaxHxW
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

    def get_training_type(self):
        return 'supervised_seq'

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

    def dump_preprocessed_dataset(self, folder: str):
        utils.check_folder(folder)

        path = "{}/50_seq_sasa.pt".format(folder)
        all_data = [self[i] for i in range(len(self))]

        torch.save(all_data, path)
