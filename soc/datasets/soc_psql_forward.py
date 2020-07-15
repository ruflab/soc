import sqlalchemy
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List
from .soc_psql import SocPSQLDataset
from . import utils as ds_utils
from .. import utils
from ..typing import SocDatasetItem, SocConfig, SocDataMetadata


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

    def _set_props(self, config: SocConfig):
        assert 'history_length' in config
        assert 'future_length' in config

        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = super(SocPSQLForwardSAToSADataset, cls).add_argparse_args(parent_parser)

        parser.add_argument('history_length', type=int, default=8)
        parser.add_argument('future_length', type=int, default=1)

        return parser

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
        df_states, df_actions = self._get_data_from_db(idx)
        assert len(df_states.index) == len(df_actions.index)

        df_states = ds_utils.preprocess_states(df_states)
        df_actions = ds_utils.preprocess_actions(df_actions)

        to_concat = []
        for i in range(len(df_states)):
            current_state_df = df_states.iloc[i]
            current_action_df = df_actions.iloc[i]

            current_state_np = np.concatenate([current_state_df[col] for col in self._obs_columns],
                                              axis=0)
            current_action_np = current_action_df['type']

            to_concat.append(current_state_np)
            to_concat.append(current_action_np)

        history_l = to_concat[:self.history_length * 2]
        future_l = to_concat[self.history_length * 2:]

        history_np = np.concatenate(history_l, axis=0)
        future_np = np.concatenate(future_l, axis=0)

        history_t = torch.tensor(history_np, dtype=torch.float32)
        future_t = torch.tensor(future_np, dtype=torch.float32)

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
        start_row_id = r
        end_row_id = r + self.seq_len_per_datum

        states = self._get_states_from_db(table_id, start_row_id, end_row_id)
        actions = self._get_actions_from_db(table_id, start_row_id, end_row_id)

        return states, actions

    def _get_states_from_db(
        self, table_id: int, start_row_id: int, end_row_id: int
    ) -> pd.DataFrame:
        query = """
            SELECT *
            FROM obsgamestates_{}
            WHERE id > {} AND id < {}
        """.format(table_id, start_row_id, end_row_id)

        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_actions_from_db(
        self, table_id: int, start_row_id: int, end_row_id: int
    ) -> pd.DataFrame:
        query = """
            SELECT *
            FROM gameactions_{}
            WHERE id > {} AND id < {}
        """.format(table_id, start_row_id, end_row_id)

        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def get_input_size(self) -> List:
        """
            Return the input dimension
        """
        size = self._state_size.copy()
        size[0] += self._action_size[0]
        size[0] *= self.history_length

        return size

    def get_output_size(self) -> List:
        """
            Return the output dimension
        """
        size = self._state_size.copy()
        size[0] += self._action_size[0]
        size[0] *= self.future_length

        return size

    def get_collate_fn(self):
        return None

    def get_training_type(self):
        return 'supervised_forward'

    def get_output_metadata(self) -> SocDataMetadata:
        metadata: SocDataMetadata = {
            'map': [],
            'properties': [],
            'pieces': [],
            'infos': [],
            'action': [],
        }
        for i in range(self.future_length):
            start_i = i * (self._state_size[0] + self._action_size[0])
            metadata['map'].append([start_i + 0, start_i + 2])
            metadata['properties'].append([start_i + 2, start_i + 9])
            metadata['pieces'].append([start_i + 9, start_i + 81])
            metadata['infos'].append([start_i + 81, start_i + 245])
            metadata['action'].append([start_i + 245, start_i + 262])

        return metadata

    def dump_preprocessed_dataset(self, folder: str):
        utils.check_folder(folder)

        path = "{}/50_forward_sasa.pt".format(folder)
        all_data = [self[i] for i in range(len(self))]

        torch.save(all_data, path)
