import sqlalchemy
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, List
from .soc_psql import SocPSQLDataset, PSQLConfig
from . import utils as ds_utils
from . import soc_data
from ..typing import SocDatasetItem, SocDataMetadata


@dataclass
class PSQLForwardConfig(PSQLConfig):
    history_length: int = MISSING
    future_length: int = MISSING


class SocPSQLForwardSAToSADataset(SocPSQLDataset):
    """
        Defines a Settlers of Catan postgresql dataset for forward models.
        One datapoint is a tuple (past, future)

        Args:
            config: (Dict) The dataset configuration

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    _inc_seq_steps: List = []
    history_length: int
    future_length: int

    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

        self.input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        self.output_shape = [
            self.future_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

    def __len__(self) -> int:
        return self._get_length()

    def _get_length(self):
        if self._length == -1 and self.engine is not None:
            query = r"""
                SELECT count(id), sum(nbsteps)
                FROM simulation_games
            """
            res = self.engine.execute(sqlalchemy.text(query))
            nb_games, total_steps = res.first()

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

    def _get_nb_steps(self) -> List:
        if self.engine is None:
            raise Exception('No engine found')

        query = r"""
            SELECT nbsteps
            FROM simulation_games
        """
        res = self.engine.execute(sqlalchemy.text(query))
        data = res.fetchall()
        nb_steps = list(map(lambda x: x[0], data))

        return nb_steps

    def __getitem__(self, idx: int) -> SocDatasetItem:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        states_df, actions_df = self._get_data_from_db(idx)
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
        seq_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        history_t = seq_t[:self.history_length]
        future_t = seq_t[self.history_length:]

        return history_t, future_t

    def _get_data_from_db(self, idx: int) -> Tuple:
        if len(self._inc_seq_steps) == 0:
            self._set_stats()

        prev_seq_steps = 0
        table_id = 0
        for i, seq_steps in enumerate(self._inc_seq_steps):
            if idx < seq_steps:
                table_id = self._first_index + i
                break
            prev_seq_steps = seq_steps
        r = idx - prev_seq_steps
        start_row_id = r + 1  # We add 1 because indices in the PosGreSQL DB start at 1 and not 0
        end_row_id = start_row_id + self.seq_len_per_datum

        states = self._get_states_from_db(table_id, start_row_id, end_row_id)
        actions = self._get_actions_from_db(table_id, start_row_id, end_row_id)

        return states, actions

    def _get_states_from_db(
        self, table_id: int, start_row_id: int, end_row_id: int
    ) -> pd.DataFrame:
        query = """
            SELECT *
            FROM obsgamestates_{}
            WHERE id >= {} AND id < {}
        """.format(table_id, start_row_id, end_row_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                states_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return states_df

    def _get_actions_from_db(
        self, table_id: int, start_row_id: int, end_row_id: int
    ) -> pd.DataFrame:
        query = """
            SELECT *
            FROM gameactions_{}
            WHERE id >= {} AND id < {}
        """.format(table_id, start_row_id, end_row_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                states_df = pd.read_sql_query(query, con=conn)
        else:
            raise Exception('No engine detected')

        return states_df

    def get_input_size(self):
        """
            Return the input dimension
        """

        return self.input_shape

    def get_output_size(self):
        """
            Return the output dimension
        """

        return self.output_shape

    def get_collate_fn(self):
        return None

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


class SocPSQLForwardSAToSAPolicyDataset(SocPSQLForwardSAToSADataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: [S_h, (C_states + C_actions), H, W]

        Output: Tuple of next state and next actions
            Dims: ( [S_f, C_ss, H, W], [S_f, C_ls], [S_f, C_actions] )
    """
    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

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
        history_t, future_t = super(SocPSQLForwardSAToSAPolicyDataset, self).__getitem__(idx)

        future_states_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        future_actions_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]
        future_spatial_states_t = torch.cat([future_states_t[:, 0:3], future_states_t[:, 9:81]],
                                            dim=1)  # [S, C_ss, H, W]
        future_lin_states_t = torch.cat(
            [future_states_t[:, 3:9, 0, 0], future_states_t[:, 81:, 0, 0]], dim=1
        )  # [S, C_ls]

        return (history_t, [future_spatial_states_t, future_lin_states_t, future_actions_t])

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
