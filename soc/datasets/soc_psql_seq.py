import sqlalchemy
import numpy as np
import pandas as pd
from typing import List
from .soc_psql import SocPSQLDataset
from . import utils
from ..typing import SOCSeq


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

    def __getitem__(self, idx: int) -> SOCSeq:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """
        df_states = self._get_states_from_db(idx)
        df_actions = self._get_actions_from_db(idx)

        assert len(df_states.index) == len(df_actions.index)
        game_length = len(df_states)

        df_states = utils.preprocess_states(df_states)
        df_actions = utils.preprocess_actions(df_actions)

        state_seq = []
        action_seq = []
        for i in range(game_length):
            current_state_df = df_states.iloc[i]
            current_action_df = df_actions.iloc[i]

            current_state_np = np.concatenate([current_state_df[col] for col in self._obs_columns],
                                              axis=0)
            current_action_np = current_action_df['type']

            state_seq.append(current_state_np)
            action_seq.append(current_action_np)

        return np.array(state_seq), np.array(action_seq)

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

    def get_collate_fn(self):
        return utils.pad_seq


class SocPSQLSeqSAToSDataset(SocPSQLSeqDataset):
    """
        Returns a completely formatted dataset:

        Input: Concatenation of state and actions representation
        in Sequence.
            Dims: S x (C_states + C_actions) x H x W

        Output: Next state
            Dims: S x C_states x H x W
    """
    def __getitem__(self, idx: int) -> SOCSeq:
        data = super(SocPSQLSeqSAToSDataset, self).__getitem__(idx)
        input_np = np.concatenate([data[0], data[1]], axis=1)

        return input_np[:-1], data[0][1:]

    def get_input_size(self) -> List:
        """
            Return the input dimension
        """
        size = self._state_size.copy()
        size[0] += self._action_size[0]

        return size

    def get_output_size(self) -> List:
        """
            Return the output dimension
        """

        return self._state_size

    def get_collate_fn(self):
        return utils.pad_seq_sas

    def get_training_type(self):
        return 'supervised'
