from sqlalchemy import create_engine
from torch.utils.data import Dataset
import pandas as pd
from . import utils


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

    def __getitem__(self, idx: int):
        df_states = self._get_states_from_db(idx)
        df_actions = self._get_actions_from_db(idx)

        assert len(df_states.index) == len(df_actions.index)
        game_length = len(df_states)

        df_states = self._preprocess_states(df_states)
        # import pdb;pdb.set_trace()
        seq = []
        for i in range(game_length):
            seq.append((df_states.iloc[i].values, df_actions.iloc[i].values))

        return seq

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
        if self._length is None:
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
                - plan 0: Tile type
                - plan 1: Tile number
                - plan 2: Robber position
                (3 type of pieces, 6 way to put it around the hex)
                - plan 3-20: Player 1 pieces
                - plan 21-38: Player 2 pieces
                - plan 39-56: Player 3 pieces
                - plan 57-74: Player 4 pieces
                (id, public VP, roads, settlements, cities, soldiers, ressources, dev cards)
                - plan 75-82: Player 1 public info
                - plan 83-90: Player 2 public info
                - plan 91-98: Player 3 public info
                - plan 99-106: Player 4 public info
                - plan 107: Game phase id
                - plan 108: Development card left
                - plan 109-113: Ressources card left
                - plan 114: Starting player id
                - plan 115: Current player id
                - plan 116: Last dice result
                - plan 117: Current player has played a developement card during its turn

            State dimensions: 7x7x118
        """
        df_states['hexlayout'] = df_states['hexlayout'].apply(utils.parse_layout) \
                                                       .apply(utils.mapping_1d_2d)
        df_states['numberlayout'] = df_states['numberlayout'].apply(utils.parse_layout) \
                                                             .apply(utils.mapping_1d_2d)
        df_states['robberhex'] = df_states['robberhex'].apply(utils.get_1d_id_from_hex) \
                                                       .apply(utils.get_2d_id) \
                                                       .apply(utils.get_one_hot_plan)
        df_states['gamestate'] = df_states['gamestate'].apply(utils.get_replicated_plan)
        df_states['devcardsleft'] = df_states['devcardsleft'].apply(utils.get_replicated_plan)
        df_states['diceresult'] = df_states['diceresult'].apply(utils.get_replicated_plan)
        df_states['startingplayer'] = df_states['startingplayer'].apply(utils.get_replicated_plan)
        df_states['currentplayer'] = df_states['currentplayer'].apply(utils.get_replicated_plan)
        df_states['playeddevcard'] = df_states['playeddevcard'].apply(utils.get_replicated_plan)

        import pdb
        pdb.set_trace()

        return df_states
