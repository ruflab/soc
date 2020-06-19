from sqlalchemy import create_engine
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple
from . import java_utils as ju


class SocPSQLDataset(Dataset):
    """
        Defines a Settlers of Catan postgresql dataset.

        Args:
            psql_username: (str) username
            psql_host: (str) host
            psql_port: (int) port
            psql_db_name: (str) database name

        Returns:
            dataset: (Dataset) A pytorch Dataset giving access to the data

    """

    _length: int

    _obs_columns = [
        'hexlayout',
        'numberlayout',
        'robberhex',
        'gamestate',
        'devcardsleft',
        'diceresult',
        'startingplayer',
        'currentplayer',
        'playeddevcard',
        'piecesonboard',
        'players',
    ]

    def __init__(
            self,
            no_db: bool = False,
            psql_username: str = 'deepsoc',
            psql_host: str = 'localhost',
            psql_port: int = 5432,
            psql_db_name: str = 'soc'
    ) -> None:
        super(SocPSQLDataset, self).__init__()

        self._length = -1

        if no_db:
            self.engine = None
        else:
            self.engine = create_engine(
                'postgresql://{}@{}:{}/{}'.format(
                    psql_username, psql_host, psql_port, psql_db_name
                )
            )

    def __len__(self) -> int:
        return self._get_length()

    def __getitem__(self, idx: int) -> Tuple:
        """
            Return one datapoint from the dataset

            A datapoint is a complete trajectory (s_t, a_t, s_t+1, etc.)

        """

        df_states = self._get_states_from_db(idx)
        df_actions = self._get_actions_from_db(idx)

        assert len(df_states.index) == len(df_actions.index)
        game_length = len(df_states)

        df_states = self._preprocess_states(df_states)
        df_actions = self._preprocess_actions(df_actions)

        states_seq = []
        actions_seq = []
        for i in range(game_length):
            current_state_df = df_states.iloc[i]
            current_action_df = df_actions.iloc[i]

            current_state_np = np.concatenate([current_state_df[col] for col in self._obs_columns],
                                              axis=2)
            current_action_np = current_action_df['type']

            states_seq.append(current_state_np)
            actions_seq.append(current_action_np)

        return np.array(states_seq), np.array(actions_seq)

    def _get_states_from_db(self, idx: int) -> pd.DataFrame:
        db_id = 100 + idx  # The first row in the DB starts at 100 in the JAVA app

        query = "SELECT * from obsgamestates_{}".format(db_id)
        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_actions_from_db(self, idx: int) -> pd.DataFrame:
        query = "SELECT * from gameactions_{}".format(idx)
        df_states = pd.read_sql_query(query, con=self.engine)

        return df_states

    def _get_length(self) -> int:
        if self._length == -1 and self.engine is not None:
            query = r"SELECT count(tablename) FROM pg_catalog.pg_tables WHERE \
                schemaname != 'pg_catalog' AND \
                schemaname != 'information_schema' AND \
                tablename SIMILAR TO 'obsgamestates_%\d\d\d+%'"

            self._length = self.engine.execute(query)

        return self._length

    def _preprocess_states(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
            This function applies the preprocessing steps necessary to move from the raw
            observation to a spatial representation.

            The spatial representation is like this:
                - plan 0: Tile type (hexlayout)
                - plan 1: Tile number
                - plan 2: Robber position
                - plan 3: Game phase id
                - plan 4: Development card left
                - plan 5: Last dice result
                - plan 6: Starting player id
                - plan 7: Current player id
                - plan 8: Current player has played a developement card during its turn
                3 type of pieces, 6 way to put it around the hex
                - plan 9-26: Player 1 pieces
                - plan 27-44: Player 2 pieces
                - plan 45-62: Player 3 pieces
                - plan 63-80: Player 4 pieces
                see ju.parse_player_infos for more information
                - plan 81-121: Player 1 public info
                - plan 122-162: Player 2 public info
                - plan 163-203: Player 3 public info
                - plan 204-244: Player 4 public info

            State shape: 7x7x245
        """
        del df_states['touchingnumbers']
        del df_states['name']
        del df_states['id']

        df_states['hexlayout'] = df_states['hexlayout'].apply(ju.parse_layout) \
                                                       .apply(ju.mapping_1d_2d)
        df_states['numberlayout'] = df_states['numberlayout'].apply(ju.parse_layout) \
                                                             .apply(ju.mapping_1d_2d)
        df_states['robberhex'] = df_states['robberhex'].apply(ju.get_1d_id_from_hex) \
                                                       .apply(ju.get_2d_id) \
                                                       .apply(ju.get_one_hot_plan)

        df_states['piecesonboard'] = df_states['piecesonboard'].apply(ju.parse_pieces)

        df_states['players'] = df_states['players'].apply(ju.parse_player_infos)

        df_states['gamestate'] = df_states['gamestate'].apply(ju.get_replicated_plan)
        df_states['devcardsleft'] = df_states['devcardsleft'].apply(ju.get_replicated_plan)
        df_states['diceresult'] = df_states['diceresult'].apply(ju.get_replicated_plan)
        df_states['startingplayer'] = df_states['startingplayer'].apply(ju.get_replicated_plan)
        df_states['currentplayer'] = df_states['currentplayer'].apply(ju.get_replicated_plan)
        df_states['playeddevcard'] = df_states['playeddevcard'].apply(ju.get_replicated_plan)

        return df_states

    def _preprocess_actions(self, df_actions: pd.DataFrame) -> pd.DataFrame:
        del df_actions['id']
        del df_actions['beforestate']
        del df_actions['afterstate']
        del df_actions['value']

        df_actions['type'] = df_actions['type'].apply(ju.parse_actions)

        return df_actions
