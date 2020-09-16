import sqlalchemy
import pandas as pd
import torch
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple, List, Union
from .soc_psql import SocPSQLDataset, PSQLConfig
from . import utils as ds_utils
from . import soc_data
from ..typing import SocDataMetadata


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
    def _set_props(self, config):
        self.history_length = config['history_length']
        self.future_length = config['future_length']
        self.seq_len_per_datum = self.history_length + self.future_length
        self._inc_seq_steps: List[int] = []
        self._length = -1

        self.input_shape = [
            self.history_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE
        self.output_shape = [
            self.future_length, soc_data.STATE_SIZE + soc_data.ACTION_SIZE
        ] + soc_data.BOARD_SIZE

    def _get_length(self):
        if self._length == -1 and self.engine is not None:
            query = r"""
                SELECT count(id), sum(nbsteps)
                FROM simulation_games
            """
            res = self.engine.execute(sqlalchemy.text(query))
            nb_games, total_steps = res.first()

            self._length = total_steps - nb_games * (self.seq_len_per_datum - 1)

        return self._length

    def _set_stats(self):
        nb_steps = self._get_nb_steps()
        for i, nb_step in enumerate(nb_steps):
            seq_nb_steps = nb_step - (self.seq_len_per_datum - 1)

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

    def __getitem__(self, idx: int):
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """

        table_id, start_row_id, end_row_id = self._get_db_idxs(idx)
        states_df = self._get_states_from_db(table_id, start_row_id, end_row_id)
        actions_df = self._get_actions_from_db(table_id, start_row_id, end_row_id)

        assert len(states_df.index) == len(actions_df.index)

        states_df = ds_utils.preprocess_states(states_df)
        actions_df = ds_utils.preprocess_actions(actions_df)

        state_seq_t = ds_utils.stack_states_df(states_df)
        action_seq_t = ds_utils.stack_actions_df(actions_df)
        seq_t = torch.cat([state_seq_t, action_seq_t], dim=1)

        history_t = seq_t[:self.history_length]
        future_t = seq_t[self.history_length:]

        return history_t, future_t

    def _get_db_idxs(self, idx: int) -> Tuple:
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

        return table_id, start_row_id, end_row_id

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
            WHERE beforestate >= {} AND beforestate < {}
        """.format(table_id, start_row_id, end_row_id)

        if self.engine is not None:
            with self.engine.connect() as conn:
                actions_df = pd.read_sql_query(query, con=conn)
                if len(actions_df) < (end_row_id - start_row_id):
                    # At the end of the trajectory, there is no action after the last state
                    # In this special case, we add it again
                    actions_df = actions_df.append(actions_df.iloc[-1])
        else:
            raise Exception('No engine detected')

        return actions_df

    def get_output_metadata(self) -> Union[SocDataMetadata, Tuple[SocDataMetadata, ...]]:
        metadata: SocDataMetadata = {}
        last_idx = 0

        for field in soc_data.STATE_FIELDS:
            metadata[field] = [last_idx, last_idx + soc_data.STATE_FIELDS_SIZE[field]]
            last_idx += soc_data.STATE_FIELDS_SIZE[field]

        metadata['actions'] = [last_idx, last_idx + soc_data.ACTION_SIZE]

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
        self._inc_seq_steps: List[int] = []
        self._length = -1

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

        states_future_t = future_t[:, :-soc_data.ACTION_SIZE]  # [S, C_s, H, W]
        actions_future_t = future_t[:, -soc_data.ACTION_SIZE:, 0, 0]  # [S, C_a]

        spatial_states_future_t, lin_states_future_t = ds_utils.separate_state_data(states_future_t)

        return (history_t, [spatial_states_future_t, lin_states_future_t, actions_future_t])

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
